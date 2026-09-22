"""Train EnvHead v2 with native MACE datasets and full-model checkpoints."""

import argparse
import json
import sys
from pathlib import Path
import torch
from mace.tools.torch_geometric.dataloader import DataLoader
from ..modules.model import EnvMACE
from ..modules.config import EnvHeadConfig
from ..data.loading import load_dataset
from .split import split_dataset
from .training import EnvHeadLoss, epoch
from .checkpoint import save_checkpoint, load_checkpoint


def is_cuda_oom(error):
    # TorchScript can wrap the original CUDA OOM in a plain RuntimeError.
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-config", help="EnvHeadConfig JSON")
    parser.add_argument("--resume", help="Full v2 checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--energy-weight", type=float)
    parser.add_argument("--forces-weight", type=float)
    parser.add_argument("--charges-weight", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--valid-fraction", type=float)
    args = parser.parse_args()
    saved = load_checkpoint(args.resume, args.device) if args.resume else None
    defaults = dict(
        batch_size=1,
        learning_rate=0.01,
        energy_weight=1.0,
        forces_weight=1.0,
        charges_weight=0.0,
        seed=0,
        valid_fraction=0.1,
    )
    previous = saved.get("training_args", {}) if saved else {}
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, previous.get(key, default))
    torch.manual_seed(args.seed)
    if saved:
        model = saved["model"]
        if args.dtype and next(model.parameters()).dtype != getattr(torch, args.dtype):
            raise ValueError("Resume dtype must match the checkpoint")
    else:
        if not args.foundation:
            parser.error("Supply --foundation or --resume")
        base = torch.load(args.foundation, map_location=args.device, weights_only=False)
        if args.dtype:
            base = base.to(dtype=getattr(torch, args.dtype))
        torch.set_default_dtype(next(base.parameters()).dtype)
        config = (
            EnvHeadConfig(**json.loads(Path(args.head_config).read_text()))
            if args.head_config
            else EnvHeadConfig()
        )
        if args.enable_cueq:
            config.cueq = True
        model = EnvMACE(base, config).to(args.device)
    if saved and args.enable_cueq and not getattr(model, "_cueq_enabled", False):
        raise ValueError(
            "Enable cuEq at the start of a new run, before optimizer creation"
        )
    torch.set_default_dtype(next(model.parameters()).dtype)
    train = load_dataset(args.train_file, model)
    if args.valid_file:
        valid = load_dataset(args.valid_file, model)
        overlap = set(g for g in train.group_ids if g) & set(
            g for g in valid.group_ids if g
        )
        if overlap:
            raise ValueError("Training/validation source groups overlap")
    else:
        if not all(train.group_ids):
            raise ValueError(
                "Automatic validation split requires group_id for every sample"
            )
        train, valid = split_dataset(train, args.valid_fraction, args.seed)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    first = 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first = saved["epoch"]
        torch.set_rng_state(saved["rng_state"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng_state"]])
    train_loader, valid_loader = (
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        DataLoader(valid, batch_size=args.batch_size),
    )
    loss = EnvHeadLoss(args.energy_weight, args.forces_weight, args.charges_weight)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "last.pt").exists() and not saved:
        raise FileExistsError("Output contains a previous run; use --resume")
    (directory / "arguments.json").write_text(json.dumps(vars(args), indent=2))
    for step in range(first, args.epochs):
        try:
            training = epoch(model, train_loader, loss, optimizer)
            validation = epoch(model, valid_loader, loss)
        except RuntimeError as error:
            if not str(args.device).startswith("cuda") or not is_cuda_oom(error):
                raise
            print(str(error).splitlines()[-1], file=sys.stderr, flush=True)
            print(f"ENVHEAD_CUDA_OOM batch_size={args.batch_size}", file=sys.stderr, flush=True)
            raise SystemExit(86) from None
        scheduler.step(validation["loss"])
        record = dict(epoch=step + 1, train=training, valid=validation)
        print(json.dumps(record), flush=True)
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(
            directory / "last.pt",
            model,
            optimizer,
            scheduler,
            step + 1,
            training_args=vars(args),
        )
