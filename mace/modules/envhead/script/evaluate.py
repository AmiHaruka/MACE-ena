"""Evaluate EnvHead models and optionally export environment and charge responses."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mace.modules.envhead.backends import enable_cueq
from mace.modules.envhead.data.loading import load_dataset
from mace.modules.envhead.modules.electrostatics import field_backend
from mace.modules.envhead.modules.graph import resolve_neighbor_backend
from mace.modules.envhead.train.checkpoint import METRICS_VERSION, load_inference
from mace.modules.envhead.train.training import EnvHeadLoss, epoch
from mace.tools.torch_geometric.dataloader import DataLoader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Complete .model export or full EnvHead checkpoint",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--execution-mode", choices=["reference", "optimized"], default="optimized"
    )
    parser.add_argument(
        "--neighbor-backend", choices=["auto", "reference", "pyg"], default="reference"
    )
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--cueq-conv-fusion", action="store_true")
    parser.add_argument("--compare-environment", action="store_true")
    parser.add_argument(
        "--mm-forces", action="store_true",
        help="Evaluate full-energy forces on background sites against available labels",
    )
    parser.add_argument("--strengths", type=float, nargs="*", default=[])
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {target}")
    saved = load_inference(args.checkpoint, args.device)
    model = saved["model"].eval()
    if args.enable_cueq or args.cueq_conv_fusion:
        model = enable_cueq(model, conv_fusion=args.cueq_conv_fusion)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(next(model.parameters()).dtype)
        return evaluate(model, saved, args, target)
    finally:
        torch.set_default_dtype(previous_dtype)


def evaluate(model, saved, args, target):
    dataset = load_dataset(args.data, model)
    if not len(dataset):
        raise ValueError("Evaluation data is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size)
    training_args = saved.get("training_args", {})
    stage = saved.get("stage", 1)
    prefix = "stage_two_" if stage == 2 else ""
    defaults = (
        (100, 10, training_args.get("charges_weight", 10))
        if stage == 2 else (10, 100, 10)
    )
    weights = saved.get("loss_weights", {})
    criterion = EnvHeadLoss(
        *(
            weights.get(key, training_args.get(f"{prefix}{key}_weight", default))
            for key, default in zip(("energy", "forces", "charges"), defaults)
        )
    )
    neighbor = resolve_neighbor_backend(args.neighbor_backend, args.device)
    options = dict(execution_mode=args.execution_mode, neighbor_backend=neighbor)
    rows, scans, atomic = [], [], []
    offset = 0
    mm_error_sq, mm_components = 0.0, 0

    def export(batch, prediction):
        nonlocal offset, mm_error_sq, mm_components
        data = batch.to_dict()
        # Transfer each export tensor once; atom-wise scalar CUDA reads synchronize.
        prediction = {key: value.detach().cpu() for key, value in prediction.items()}
        reference = {
            key: data[key].detach().cpu()
            for key in (
                "ptr",
                "mm_count",
                "energy",
                "energy_weight",
                "forces",
                "forces_weight",
                "charges",
                "charges_weight",
                "positions",
            )
        }
        if args.mm_forces:
            reference.update({
                key: data[key].detach().cpu()
                for key in ("mm_forces", "mm_forces_weight")
            })
        numbers = model.atomic_numbers[batch.node_attrs.argmax(-1)].detach().cpu()
        compared = (
            model.compare_environment(data, **options)
            if args.compare_environment
            else None
        )
        if compared is not None:
            compared = {key: value.detach().cpu() for key, value in compared.items()}
        mm_start = 0
        for g, (lo, hi) in enumerate(
            zip(reference["ptr"][:-1].tolist(), reference["ptr"][1:].tolist())
        ):
            index = offset + g
            energy = float(prediction["energy"][g].detach())
            reference_energy = (
                float(reference["energy"][g])
                if reference["energy_weight"][g] > 0 else None
            )
            error = energy - reference_energy if reference_energy is not None else None
            labelled = bool(reference["charges_weight"][g] > 0)
            qerror = (
                (prediction["charges"][lo:hi] - reference["charges"][lo:hi].flatten())
                if labelled
                else None
            )
            row = dict(
                index=index,
                sample_id=dataset.sample_ids[index],
                group_id=dataset.group_ids[index],
                qm_atoms=hi - lo,
                mm_sources=int(reference["mm_count"][g]),
                energy=energy,
                reference_energy=reference_energy,
                energy_error=error,
                energy_error_per_atom=error / (hi - lo) if error is not None else None,
                energy_ele=float(prediction["energy_ele"][g].detach()),
                forces_rmse=float(
                    (prediction["forces"][lo:hi] - reference["forces"][lo:hi])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ) if reference["forces_weight"][g] > 0 else None,
                charges_rmse=float(qerror.square().mean().sqrt().detach())
                if labelled
                else None,
            )
            if args.mm_forces:
                mm_stop = mm_start + row["mm_sources"]
                valid = reference["mm_forces_weight"][mm_start:mm_stop].flatten() > 0
                errors = (
                    prediction["mm_forces"][mm_start:mm_stop][valid].double()
                    - reference["mm_forces"][mm_start:mm_stop][valid].double()
                )
                squared = float(errors.square().sum())
                count = errors.numel()
                row["mm_forces_rmse"] = (squared / count) ** 0.5 if count else None
                mm_error_sq += squared
                mm_components += count
                mm_start = mm_stop
            if compared is not None:
                for key in ("energy_vac", "energy_response"):
                    row[key] = float(compared[key][g].detach())
                row["delta_charge_rms"] = float(
                    compared["delta_charges"][lo:hi].square().mean().sqrt().detach()
                )
            rows.append(row)
            for local, atom in enumerate(range(lo, hi)):
                item = dict(
                    index=index,
                    atom_index=local,
                    sample_id=dataset.sample_ids[index],
                    group_id=dataset.group_ids[index],
                    atomic_number=int(numbers[atom]),
                    x=float(reference["positions"][atom, 0]),
                    y=float(reference["positions"][atom, 1]),
                    z=float(reference["positions"][atom, 2]),
                    charge=float(prediction["charges"][atom].detach()),
                    reference_charge=float(reference["charges"][atom])
                    if labelled
                    else None,
                )
                if compared is not None:
                    item["charge_vacuum"] = float(
                        compared["charges_vacuum"][atom].detach()
                    )
                    item["delta_charge"] = float(
                        compared["delta_charges"][atom].detach()
                    )
                atomic.append(item)
        del compared
        for strength in args.strengths:
            changed = dict(data, mm_charges=strength * data["mm_charges"])
            out = model(changed, compute_force=False, compute_response=True, **options)
            start = 0
            for g, count in enumerate(batch.mm_count.reshape(-1).tolist()):
                lo, hi = int(batch.ptr[g]), int(batch.ptr[g + 1])
                derivative = (
                    out["response_potential"][start : start + count]
                    * data["mm_charges"][start : start + count].flatten()
                ).sum()
                scans.append(
                    dict(
                        index=offset + g,
                        strength=strength,
                        energy=float(out["energy"][g].detach()),
                        energy_ele=float(out["energy_ele"][g].detach()),
                        dE_ds=float(derivative.detach()),
                        charge_sum=float(out["charges"][lo:hi].sum().detach()),
                    )
                )
                start += count
            del out
        offset += len(batch.ptr) - 1

    metrics = epoch(
        model, loader, criterion, on_batch=export,
        compute_mm_force=args.mm_forces, **options,
    )
    if args.mm_forces:
        metrics["mm_forces_rmse"] = (
            (mm_error_sq / mm_components) ** 0.5 if mm_components else None
        )
    parameter = next(model.parameters())
    summary = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_epoch=saved.get("epoch"),
        checkpoint_stage=stage,
        data=str(Path(args.data).resolve()),
        metrics=metrics,
        metrics_version=METRICS_VERSION,
        structures=len(dataset),
        units=dict(
            energy="eV",
            energy_rmse_per_atom="eV/atom",
            forces="eV/Angstrom",
            charges="e",
        ),
        group_ids=sorted(set(dataset.group_ids)),
        model_config=asdict(model.config),
        loss_weights=dict(
            energy=criterion.energy_weight,
            forces=criterion.forces_weight,
            charges=criterion.charges_weight,
        ),
        execution=dict(
            **options,
            field_backend=field_backend(parameter.device, parameter.dtype, model.config)
            if args.execution_mode == "optimized"
            else "torch",
            cueq=bool(getattr(model, "_cueq_enabled", False)),
            cueq_execution=("cuda" if parameter.is_cuda else "torch_reference")
            if getattr(model, "_cueq_enabled", False) else None,
            cueq_conv_fusion=all(
                bool(getattr(layer, "conv_fusion", False))
                for layer in model.foundation.interactions
            ),
            dtype=str(parameter.dtype),
            device=str(parameter.device),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(parameter.device)
            if parameter.is_cuda
            else None,
        ),
        diagnostics=dict(
            compare_environment=args.compare_environment, strengths=args.strengths
        ),
        scope="Held-out labels as supplied; comparison is a model-defined response; strength sweeps need independent reference for validation",
    )
    if args.mm_forces:
        summary["units"]["mm_forces"] = "eV/Angstrom"
        summary["diagnostics"]["mm_forces"] = True
    target.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("predictions", rows),
        ("atomic_charges", atomic),
        ("strength_scan", scans),
    ):
        if records:
            with (target / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (target / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
