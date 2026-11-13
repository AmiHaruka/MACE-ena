"""Public EnvHead training: staged EMA evaluation and full-state recovery."""

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys

import torch
from torch.nn.parallel import DistributedDataParallel

from ..modules.model import EnvMACE
from ..modules.config import EnvHeadConfig
from ..data.loading import load_dataset
from .checkpoint import METRICS_VERSION, export_model, load_checkpoint, save_checkpoint
from .distributed import DistributedContext, make_loaders, seed_all
from .optimization import build_optimizer, make_ema
from .settings import RunPaths, build_parser, parse_args
from .split import save_validation_indices, split_dataset
from .training import EnvHeadLoss, TrainingObjective, epoch


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and any(
            marker in str(error).lower()
            for marker in (
                "cuda out of memory",
                "cuda error: out of memory",
                "cudaerrormemoryallocation",
                "cuda_error_out_of_memory",
            )
        )
    )


def report(paths, context, message):
    if context.is_main:
        print(message, flush=True)
        with paths.log.open("a") as stream:
            stream.write(message + "\n")


def main(argv=None):
    bootstrap = build_parser().parse_args(argv)
    context = DistributedContext.initialize(bootstrap.device)
    try:
        saved = (
            load_checkpoint(bootstrap.resume, context.device)
            if bootstrap.resume
            else None
        )
        if saved and saved.get("training_state_version") != 1:
            raise ValueError(
                "Checkpoint lacks named optimizer/EMA training state; start a new training run"
            )
        args = parse_args(argv, saved.get("training_args") if saved else None)
        if saved:
            for key in (
                "weight_decay",
                "amsgrad",
                "foundation_lr_factor",
                "environment_lr_factor",
                "ema",
                "ema_decay",
                "lr_factor",
                "scheduler_patience",
                "stage_two",
                "stage_two_start",
                "stage_two_lr",
            ):
                if getattr(args, key) != saved["training_args"].get(key):
                    raise ValueError(
                        f"Resume must retain {key}; start a new run to change optimizer/EMA/schedule settings"
                    )
        args.device, args.world_size = str(context.device), context.world_size
        seed_all(args.seed)
        paths = RunPaths(args)
        if context.is_main:
            if not saved:
                paths.ensure_new()
            paths.create()
        context.barrier()
        if saved:
            model = saved["model"]
            if args.dtype and next(model.parameters()).dtype != getattr(
                torch, args.dtype
            ):
                raise ValueError("Resume dtype must match the checkpoint")
        else:
            if not args.foundation:
                raise ValueError("Supply --foundation or --resume")
            base = torch.load(
                args.foundation, map_location=args.device, weights_only=False
            )
            if args.dtype:
                base = base.to(dtype=getattr(torch, args.dtype))
            torch.set_default_dtype(next(base.parameters()).dtype)
            config = (
                EnvHeadConfig(**json.loads(Path(args.head_config).read_text()))
                if args.head_config
                else EnvHeadConfig()
            )
            if args.enable_cueq or args.cueq_conv_fusion:
                config.cueq = True
            model = EnvMACE(base, config).to(args.device)
        if args.cueq_conv_fusion and model.config.environment_mode != "once":
            raise ValueError("QM convolution fusion requires a once model")
        if (
            saved
            and (args.enable_cueq or args.cueq_conv_fusion)
            and not getattr(model, "_cueq_enabled", False)
        ):
            raise ValueError(
                "Enable cuEq at the start of a new run, before optimizer creation"
            )
        if getattr(model, "_cueq_enabled", False) and next(model.parameters()).is_cuda:
            from ..backends import enable_cueq

            enable_cueq(model, conv_fusion=args.cueq_conv_fusion)
        torch.set_default_dtype(next(model.parameters()).dtype)
        from ..modules.graph import resolve_neighbor_backend
        from ..modules.electrostatics import field_backend

        neighbors = resolve_neighbor_backend(args.neighbor_backend, args.device)
        if neighbors == "pyg":
            from ..modules.graph import _pyg_builder

            _pyg_builder(model)
        fields = (
            field_backend(
                next(model.parameters()).device,
                next(model.parameters()).dtype,
                model.config,
            )
            if args.execution_mode == "optimized"
            else "torch"
        )
        report(
            paths,
            context,
            f"EnvHead field_backend={fields}; neighbors={neighbors}; execution={args.execution_mode}",
        )
        report(
            paths,
            context,
            f"Batch per rank={args.batch_size}; global batch={args.batch_size * context.world_size}; workers per rank={args.num_workers}",
        )
        train = load_dataset(args.train_file, model)
        if args.valid_file:
            valid = load_dataset(args.valid_file, model)
            overlap = set(g for g in train.group_ids if g) & set(
                g for g in valid.group_ids if g
            )
            if overlap:
                raise ValueError("Training/validation source groups overlap")
            args.split_mode = "explicit"
        else:
            args.split_mode = "group" if any(train.group_ids) else "random"
            train, valid = split_dataset(train, args.valid_fraction, args.seed)
            if context.is_main:
                save_validation_indices(valid, paths.validation_indices)
        args.validation_indices = getattr(valid, "indices", None)
        args.train_indices = getattr(train, "indices", None)
        args.data_split = dict(
            train_samples=len(train),
            valid_samples=len(valid),
            train_groups=sorted(set(g for g in train.group_ids if g)),
            valid_groups=sorted(set(g for g in valid.group_ids if g)),
        )
        report(
            paths,
            context,
            "Data split: " + json.dumps(dict(mode=args.split_mode, **args.data_split)),
        )
        train_loader, valid_loader = make_loaders(train, valid, args, context)
        optimizer = build_optimizer(
            model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foundation_lr_factor=args.foundation_lr_factor,
            environment_lr_factor=args.environment_lr_factor,
            amsgrad=args.amsgrad,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_factor, patience=args.scheduler_patience
        )
        ema = make_ema(model, args.ema_decay) if args.ema else None
        if saved:
            expected = [g["param_names"] for g in optimizer.param_groups]
            actual = [g.get("param_names") for g in saved["optimizer"]["param_groups"]]
            if expected != actual:
                raise ValueError(
                    "Checkpoint optimizer parameter groups differ from this model"
                )
            if bool(saved.get("ema")) != bool(ema):
                raise ValueError("Resume EMA configuration must match the checkpoint")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            if ema is not None:
                ema.load_state_dict(saved["ema"])
                ema.to(
                    device=next(model.parameters()).device,
                    dtype=next(model.parameters()).dtype,
                )
            if args.explicit_learning_rate:
                for group in optimizer.param_groups:
                    factor = (
                        args.foundation_lr_factor
                        if group["name"].startswith("foundation_")
                        else args.environment_lr_factor
                    )
                    group["lr"] = args.learning_rate * factor
                scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        loss = EnvHeadLoss(
            args.energy_weight, args.forces_weight, args.charges_weight, args.mm_forces_weight
        )
        options = dict(
            neighbor_backend=args.neighbor_backend, execution_mode=args.execution_mode
        )
        distributed_model = (
            DistributedDataParallel(
                TrainingObjective(model, loss, options, ddp=True),
                device_ids=[context.local_rank]
                if context.device.type == "cuda"
                else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if context.enabled
            else None
        )
        if saved:
            context.restore_rng(saved["rank_states"])
        if context.is_main:
            paths.arguments.write_text(json.dumps(vars(args), indent=2))
        run_training(
            model,
            train_loader,
            valid_loader,
            loss,
            optimizer,
            scheduler,
            args,
            saved,
            ema=ema,
            context=context,
            distributed_model=distributed_model,
        )
    finally:
        context.close()


def _preserve_best(paths, args, saved):
    """Copy an available matching best into a new output location."""
    target = paths.checkpoint("best")
    candidates = [target]
    if args.resume:
        candidates.append(Path(args.resume).with_name(target.name))
        candidates.append(
            Path(args.resume).with_name(
                Path(args.resume).name.replace("_last.pt", "_best.pt")
            )
        )
    keys = (
        "train_file",
        "valid_file",
        "validation_indices",
        "energy_weight",
        "forces_weight",
        "charges_weight",
        "mm_forces_weight",
        "stage_two_energy_weight",
        "stage_two_forces_weight",
        "stage_two_charges_weight",
        "stage_two_mm_forces_weight",
    )
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        with torch.random.fork_rng(devices=[]):
            candidate = torch.load(path, map_location="cpu", weights_only=False)
        valid = (
            candidate.get("epoch") == saved.get("best_epoch")
            and candidate.get("best_loss") == saved.get("best_loss")
            and candidate.get("stage", 1) == saved.get("stage", 1)
            and candidate.get("metrics_version") == METRICS_VERSION
            and all(
                candidate.get("training_args", {}).get(k, 0.0 if k.endswith("mm_forces_weight") else None)
                == getattr(args, k, None)
                for k in keys
            )
        )
        if valid:
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            from .checkpoint import load_model

            with torch.random.fork_rng(devices=[]):
                export_model(
                    paths.model(saved.get("stage", 1)),
                    load_model(path),
                    metadata=dict(
                        training_args=vars(args),
                        epoch=candidate["epoch"],
                        stage=candidate.get("stage", 1),
                    ),
                )
            if saved.get("stage", 1) == 2:
                original_args = candidate.get("training_args", {})
                source = Path(
                    original_args.get("model_dir")
                    or original_args.get("work_dir")
                    or path.resolve().parent.parent
                )
                original = source / f"{original_args.get('name', args.name)}.model"
                if (
                    original.is_file()
                    and original.resolve() != paths.model(1).resolve()
                ):
                    shutil.copy2(original, paths.model(1))
            return True
    return False


def run_training(
    model,
    train_loader,
    valid_loader,
    loss,
    optimizer,
    scheduler,
    args,
    saved=None,
    *,
    ema=None,
    context=None,
    distributed_model=None,
):
    context = context or DistributedContext(next(model.parameters()).device)
    paths = RunPaths(args)
    if context.is_main:
        paths.create()
    first = saved["epoch"] if saved else 0
    steps = saved.get("optimizer_steps", 0) if saved else 0
    stage = saved.get("stage", 1) if saved else 1
    active_batch_size, active_phase = args.batch_size, "training"
    keys = (
        "energy_weight",
        "forces_weight",
        "charges_weight",
        "mm_forces_weight",
        "stage_two_energy_weight",
        "stage_two_forces_weight",
        "stage_two_charges_weight",
        "stage_two_mm_forces_weight",
        "ema",
        "ema_decay",
        "stage_two",
        "stage_two_start",
        "train_file",
        "valid_file",
        "validation_indices",
    )
    comparable = bool(
        saved
        and saved.get("metrics_version") == METRICS_VERSION
        and saved.get("best_loss") is not None
        and all(
            saved.get("training_args", {}).get(k, 0.0 if k.endswith("mm_forces_weight") else None)
            == getattr(args, k, None)
            for k in keys
        )
    )
    comparable = context.broadcast(
        _preserve_best(paths, args, saved) if comparable and context.is_main else False
    )
    best_loss = saved["best_loss"] if comparable else float("inf")
    best_epoch = saved["best_epoch"] if comparable else None

    def set_weights():
        prefix = "stage_two_" if stage == 2 else ""
        for key in ("energy_weight", "forces_weight", "charges_weight", "mm_forces_weight"):
            setattr(loss, key, getattr(args, prefix + key))

    def loss_weights():
        result = dict(energy=loss.energy_weight, forces=loss.forces_weight, charges=loss.charges_weight)
        if args.mm_forces_weight > 0 or args.stage_two_mm_forces_weight > 0:
            result["mm_forces"] = loss.mm_forces_weight
        return result

    def validate():
        nonlocal active_batch_size, active_phase
        active_batch_size, active_phase = args.valid_batch_size, "validation"
        with ema.average_parameters() if ema is not None else nullcontext():
            return epoch(model, valid_loader, loss, **options)

    def record(completed, training, validation, rates, event=None):
        row = dict(
            epoch=completed,
            stage=stage,
            train=training,
            valid=validation,
            learning_rate=rates[0],
            learning_rates=rates,
            loss_weights=loss_weights(),
            evaluation_weights="ema" if ema is not None else "instantaneous",
            optimizer_steps=steps,
            metrics_version=METRICS_VERSION,
        )
        if event:
            row["event"] = event
        if context.is_main:
            report(paths, context, json.dumps(row))
            with paths.metrics.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def persist(completed, improved):
        rank_states = context.capture_rng()
        if context.is_main:
            kwargs = dict(
                epoch=completed,
                training_args=vars(args),
                best_loss=best_loss,
                best_epoch=best_epoch,
                optimizer_steps=steps,
                ema=ema,
                stage=stage,
                rank_states=rank_states,
            )
            save_checkpoint(
                paths.checkpoint("last"), model, optimizer, scheduler, **kwargs
            )
            metadata = dict(
                training_args=vars(args),
                epoch=completed,
                stage=stage,
                loss_weights=loss_weights(),
            )
            export_model(paths.model("last"), model, ema=ema, metadata=metadata)
            if improved:
                save_checkpoint(
                    paths.checkpoint("best"), model, optimizer, scheduler, **kwargs
                )
                export_model(paths.model(stage), model, ema=ema, metadata=metadata)
        context.barrier()

    options = dict(
        neighbor_backend=args.neighbor_backend,
        execution_mode=args.execution_mode,
        context=context,
    )
    set_weights()
    try:
        if not saved:
            # Establish recovery before the first (potentially oversized) force
            # validation. No best or inference model exists until it completes.
            initial_rng = context.capture_rng()
            if context.is_main:
                save_checkpoint(
                    paths.checkpoint("last"),
                    model,
                    optimizer,
                    scheduler,
                    epoch=0,
                    training_args=vars(args),
                    optimizer_steps=0,
                    ema=ema,
                    stage=stage,
                    rank_states=initial_rng,
                )
            context.barrier()
        if not comparable:
            validation = validate()
            best_loss, best_epoch = validation["loss"], first
            if saved:
                scheduler.best, scheduler.num_bad_epochs, scheduler.cooldown_counter = (
                    float("inf"),
                    0,
                    0,
                )
            record(first, None, validation, [g["lr"] for g in optimizer.param_groups])
            persist(first, True)
        for completed in range(first, args.epochs):
            if args.stage_two and stage == 1 and completed >= args.stage_two_start:
                stage = 2
                set_weights()
                for group in optimizer.param_groups:
                    factor = (
                        args.foundation_lr_factor
                        if group.get("name", "").startswith("foundation_")
                        else args.environment_lr_factor
                    )
                    group["lr"] = min(group["lr"], args.stage_two_lr * factor)
                scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
                scheduler.best, scheduler.num_bad_epochs, scheduler.cooldown_counter = (
                    float("inf"),
                    0,
                    0,
                )
                validation = validate()
                best_loss, best_epoch = validation["loss"], completed
                record(
                    completed,
                    None,
                    validation,
                    [g["lr"] for g in optimizer.param_groups],
                    "stage_start",
                )
                persist(completed, True)
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(completed)
            rates = [g["lr"] for g in optimizer.param_groups]
            active_batch_size, active_phase = args.batch_size, "training"
            training = epoch(
                model,
                train_loader,
                loss,
                optimizer,
                ema=ema,
                clip_grad=args.clip_grad,
                distributed_model=distributed_model,
                **options,
            )
            steps += len(train_loader)
            validation = validate()
            if stage == 1:
                scheduler.step(validation["loss"])
            improved = validation["loss"] < best_loss
            if improved:
                best_loss, best_epoch = validation["loss"], completed + 1
            record(completed + 1, training, validation, rates)
            persist(completed + 1, improved)
    except RuntimeError as error:
        if torch.device(args.device).type != "cuda" or not is_cuda_oom(error):
            raise
        print(str(error).splitlines()[-1], file=sys.stderr, flush=True)
        print(
            f"ENVHEAD_CUDA_OOM batch_size={active_batch_size} phase={active_phase}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(86) from None
