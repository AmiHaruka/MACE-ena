"""Usage: python -m mace.modules.envhead.train --help."""
import argparse
import importlib
import json
import logging
from pathlib import Path
from time import perf_counter

import torch


from ..adapters import foundation_adapter
from ..data.loading import load_dataset
from ..modules.config import EnvHeadConfig
from ..modules.model import EnvMACE
from .split import split_dataset
from .training import EnvHeadLoss, ProgressDataLoader, fit


def log_model_summary(base, model):
    """Report foundation architecture and the exact feature interface to EnvHead."""
    adapter = foundation_adapter(model)
    interaction = adapter.interaction(model)
    logging.info("Foundation: %s.%s", type(base).__module__, type(base).__name__)
    logging.info(
        "Precision: foundation=%s; training=%s; device=%s",
        next(base.parameters()).dtype,
        next(model.parameters()).dtype,
        next(model.parameters()).device,
    )
    logging.info(
        "Foundation adapter=%s; heads=%s; atomic_numbers=%s; cutoff=%g Angstrom",
        adapter.name,
        model.heads,
        model.atomic_numbers.tolist(),
        float(model.r_max),
    )
    logging.info(
        "Foundation interaction layers=%d; frozen parameters=%d; trainable head parameters=%d",
        len(model.interactions),
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if not n.startswith("env_head.")
        ),
        sum(p.numel() for p in model.env_head.parameters() if p.requires_grad),
    )
    logging.info(
        "Foundation interactions: %s", [type(m).__name__ for m in model.interactions]
    )
    logging.info("Foundation readouts: %s", [type(m).__name__ for m in model.readouts])
    logging.info(
        "Foundation E0s (atomic_numbers order): %s",
        model.atomic_energies_fn.atomic_energies.detach().cpu().tolist(),
    )
    if hasattr(model, "scale_shift"):
        logging.info("Foundation scale/shift: %s", model.scale_shift)
    logging.info(
        "Feature tap: final interaction, before product; irreps=%s; flattened dimension=%d",
        interaction.target_irreps,
        interaction.target_irreps.dim,
    )
    logging.info(
        "Feature layout: [QM atoms, channels, spherical components]; edge irreps=%s; radial irreps=%s",
        interaction.edge_attrs_irreps,
        interaction.edge_feats_irreps,
    )
    logging.info(
        "EnvHead geometry projection: %s -> %s",
        model.env_head.geometry.irreps_in,
        model.env_head.geometry.irreps_out,
    )
    logging.info("EnvHead configuration: %s", model.env_head.config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation",
        required=True,
        help="Trusted single-head MACE/ScaleShiftMACE/MACELES foundation file",
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--head-config", help="JSON object file with EnvHeadConfig fields"
    )
    parser.add_argument(
        "--fusion", choices=["additive", "product", "message"], default="product"
    )
    parser.add_argument(
        "--charge-route", choices=["fukui", "eem", "sqe"], default="fukui"
    )
    parser.add_argument("--density-max-l", type=int, choices=[1, 2], default=2)
    parser.add_argument("--mmc", action="store_true")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--gaussian-width", type=float, default=1.0)
    parser.add_argument("--mm-chunk-size", type=int, default=1024)
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument("--total-charge-key", default="total_charge")
    parser.add_argument("--dipole-key", default="REF_dipole")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=1.0)
    parser.add_argument("--mm-forces-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--enable-cueq", action="store_true",
        help="Accelerate MACE/MACELES backbone and EnvHead equivariant operators with cuEq",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Default: follow foundation; float64 to float32 allowed, float32 to float64 rejected",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--adapter-module",
        action="append",
        default=[],
        help="Import a trusted Python module that registers a foundation adapter",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    directory = Path(args.output_dir)
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(
            "Output directory is nonempty; select a new directory or resume a version-4 run"
        )
    if args.resume:
        saved = directory / "arguments.json"
        if (
            not saved.exists()
            or json.loads(saved.read_text()).get("format_version") != 4
        ):
            raise ValueError("Only direct-delta version-4 runs can be resumed")
    torch.manual_seed(args.seed)
    for module in args.adapter_module:
        importlib.import_module(module)
    settings = {
        key: getattr(args, key)
        for key in (
            "fusion",
            "charge_route",
            "density_max_l",
            "mmc",
            "channels",
            "num_interactions",
            "correlation",
            "gaussian_width",
            "mm_chunk_size",
        )
    }
    if args.head_config:
        with open(args.head_config, encoding="utf8") as stream:
            settings.update(json.load(stream))
    logging.info("Loading foundation: %s", args.foundation)
    base = torch.load(args.foundation, map_location="cpu", weights_only=False)
    dtype = getattr(torch, args.dtype) if args.dtype else next(base.parameters()).dtype
    model = EnvMACE(base, EnvHeadConfig(**settings), dtype=dtype).to(args.device)
    if args.enable_cueq:
        from ..backends import enable_cueq

        enable_cueq(model)
    torch.set_default_dtype(dtype)
    log_model_summary(base, model)
    logging.info("Loading training data: %s", args.train_file)
    train_graphs = load_dataset(
        args.train_file,
        model,
        args.energy_key,
        args.forces_key,
        args.total_charge_key,
        args.dipole_key,
    )
    if args.valid_file:
        valid_graphs = load_dataset(
            args.valid_file,
            model,
            args.energy_key,
            args.forces_key,
            args.total_charge_key,
            args.dipole_key,
        )
        train_ids = {g for g in train_graphs.group_ids if g}
        if train_ids.intersection(g for g in valid_graphs.group_ids if g):
            raise ValueError("group_id overlap between training and validation")
    else:
        started = perf_counter()
        logging.info("Splitting dataset indices")
        train_graphs, valid_graphs = split_dataset(
            train_graphs, valid_fraction=args.valid_fraction, seed=args.seed
        )
        logging.info("Dataset index split finished in %.3f s", perf_counter() - started)
    logging.info(
        "Data ready: train=%d; validation=%d; batch_size=%d; training batches/epoch=%d",
        len(train_graphs),
        len(valid_graphs),
        args.batch_size,
        (len(train_graphs) + args.batch_size - 1) // args.batch_size,
    )
    # Record the exact split without materializing all HDF5 atom/MM arrays.
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    split = dict(
        train_indices=getattr(train_graphs, "indices", list(range(len(train_graphs)))),
        valid_indices=getattr(valid_graphs, "indices", list(range(len(valid_graphs)))),
        train_samples=train_graphs.sample_ids,
        valid_samples=valid_graphs.sample_ids,
        train_groups=train_graphs.group_ids,
        valid_groups=valid_graphs.group_ids,
        train_file=str(Path(args.train_file).resolve()),
        valid_file=str(Path(args.valid_file).resolve()) if args.valid_file else None,
    )
    manifest = directory / "split.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != split:
            raise ValueError("Existing run has a different split or requires --resume")
    else:
        manifest.write_text(json.dumps(split, indent=2), encoding="utf-8")
    arguments = dict(vars(args), format_version=4, resolved_head_config=settings)
    # Preserve the original run metadata on continuation.
    if not args.resume:
        (directory / "arguments.json").write_text(
            json.dumps(arguments, indent=2), encoding="utf-8"
        )
    fit(
        model,
        ProgressDataLoader(
            train_graphs,
            phase="Training",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        {
            "valid": ProgressDataLoader(
                valid_graphs,
                phase="Validation",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        },
        args.output_dir,
        args.epochs,
        args.learning_rate,
        args.resume,
        EnvHeadLoss(args.energy_weight, args.forces_weight, args.mm_forces_weight),
    )


if __name__ == "__main__":
    main()
