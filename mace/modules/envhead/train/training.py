"""Thin single-process adapter to the native MACE training loop."""
from pathlib import Path
import logging
from time import perf_counter
from functools import update_wrapper
from types import FunctionType

import torch

from mace.modules.loss import WeightedEnergyForcesLoss
from mace.tools.checkpoint import CheckpointState
from mace.tools.train import MACELoss as NativeMACELoss
from mace.tools.train import evaluate as native_evaluate, train as native_train
from mace.tools.utils import MetricsLogger
from mace.tools.torch_geometric.dataloader import DataLoader

from .checkpoint import EnvCheckpointHandler, save_head


class EnvValidationMetrics(NativeMACELoss):
    """Native pooled metrics plus vacuum/PC diagnostics, all partition independent."""

    def __init__(self, loss_fn):
        super().__init__(loss_fn)
        self.add_state(
            "loss_components", default=torch.zeros(3, 2), dist_reduce_fx="sum"
        )
        self.add_state(
            "phase_components", default=torch.zeros(2, 3, 2), dist_reduce_fx="sum"
        )
        self.add_state("phase_errors", default=torch.zeros(2, 4), dist_reduce_fx="sum")
        self.add_state(
            "response_maxima", default=torch.zeros(2, 4), dist_reduce_fx="max"
        )

    def update(self, batch, output):
        self.loss_components += self.loss_fn.validation_components(batch, output)
        n = batch.ptr[1:] - batch.ptr[:-1]
        phases = batch.mm_count.reshape(-1) > 0
        for phase in range(2):
            graphs = phases == bool(phase)
            if not graphs.any():
                continue
            self.phase_components[phase] += self.loss_fn.validation_components(
                batch, output, graphs
            )
            atoms = torch.repeat_interleave(graphs, n)
            e = ((output["energy"].detach() - batch.energy) / n)[graphs]
            f = (output["forces"].detach() - batch.forces)[atoms]
            e = e[(batch.weight * batch.energy_weight)[graphs] != 0]
            f = f[
                torch.repeat_interleave(batch.weight * batch.forces_weight, n)[atoms]
                != 0
            ]
            self.phase_errors[phase] += torch.stack(
                (
                    e.square().sum(),
                    e.new_tensor(e.numel()),
                    f.square().sum(),
                    e.new_tensor(f.numel()),
                )
            )
            maxima = torch.stack(
                [
                    output[key].detach().reshape(-1)[graphs].amax()
                    for key in (
                        "max_charge",
                        "max_dipole",
                        "max_quadrupole",
                        "charge_error",
                    )
                ]
            )
            self.response_maxima[phase] = torch.maximum(
                self.response_maxima[phase], maxima
            )
        super().update(batch, output)

    def compute(self):
        _, metrics = super().compute()
        loss = self.loss_fn.validation_loss(self.loss_components).detach().cpu().item()
        metrics["loss"] = loss
        weights = torch.stack(
            (
                self.loss_fn.energy_weight,
                self.loss_fn.forces_weight,
                self.loss_fn.mm_forces_weight,
            )
        )
        for phase, name in enumerate(("vacuum", "pc")):
            comp = self.phase_components[phase]
            if comp[0, 1] == 0:
                continue
            means = comp[:, 0] / torch.maximum(
                comp[:, 1], comp.new_tensor([1.0, 1.0, 3.0])
            )
            values = (means * weights).detach().cpu().tolist()
            errors = self.phase_errors[phase]
            rmse = (
                torch.sqrt(errors[[0, 2]] / errors[[1, 3]].clamp_min(1))
                .detach()
                .cpu()
                .tolist()
            )
            maxima = self.response_maxima[phase].detach().cpu().tolist()
            metrics[name + "_loss"] = sum(values)
            for key, value in zip(
                ("energy_loss", "forces_loss", "mm_forces_loss"), values
            ):
                metrics[name + "_" + key] = value
            metrics[name + "_rmse_e_per_atom"], metrics[name + "_rmse_f"] = rmse
            for key, value in zip(
                ("max_charge", "max_dipole", "max_quadrupole", "charge_error"), maxima
            ):
                metrics[name + "_" + key] = value
            logging.info(
                "%s: loss(E,F,MMF)=%s; RMSE E/atom,F=%s eV,eV/A; max q,mu,Theta,charge_error=%s",
                name,
                values,
                rmse,
                maxima,
            )
        return loss, metrics


def _bind(function, replacements):
    """Bind native functions privately; never mutate their module globals.

    Native train/evaluate lack evaluator/metric factory arguments. Reusing
    their function bodies keeps optimizer and checkpoint logic unchanged.
    """
    for name in replacements:
        if name not in function.__code__.co_names:
            raise RuntimeError(
                f"Native {function.__name__} no longer uses {name}; update EnvHead training adapter"
            )
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    bound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(bound, function)


evaluate = _bind(native_evaluate, {"MACELoss": EnvValidationMetrics})
train = _bind(native_train, {"evaluate": evaluate})


class ProgressDataLoader(DataLoader):
    """Native batching with phase and elapsed time around each consumed batch."""

    def __init__(self, *args, phase, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            # Workers must never inherit the training process CUDA context.
            kwargs.setdefault("multiprocessing_context", "spawn")
        super().__init__(*args, **kwargs)
        self.phase = phase

    def __iter__(self):
        for index, batch in enumerate(super().__iter__(), 1):
            started = perf_counter()
            logging.info("%s batch %d/%d: started", self.phase, index, len(self))
            yield batch
            logging.info(
                "%s batch %d/%d: finished in %.2f s",
                self.phase,
                index,
                len(self),
                perf_counter() - started,
            )


class EnvHeadLoss(WeightedEnergyForcesLoss):
    def validation_components(self, ref, pred, graphs=None):
        """Weighted sums/counts for partition-independent validation."""
        n = ref.ptr[1:] - ref.ptr[:-1]
        e = (
            ref.weight
            * ref.energy_weight
            * ((pred["energy"] - ref.energy) / n).square()
        )
        w = torch.repeat_interleave(ref.weight * ref.forces_weight, n)[:, None]
        f = w * (pred["forces"] - ref.forces).square()
        if graphs is not None:
            e = e[graphs]
            f = f[torch.repeat_interleave(graphs, n)]
        mm_sum, mm_count = e.new_zeros(()), e.new_zeros(())
        if getattr(ref, "mm_forces", None) is not None and ref.mm_forces.numel():
            w = ref.mm_forces_weight
            error = (pred["mm_forces"] - ref.mm_forces).square()
            if graphs is not None:
                sources = torch.repeat_interleave(graphs, ref.mm_count.reshape(-1))
                w, error = w[sources], error[sources]
            mm_sum = (error * w).sum()
            mm_count = 3 * w.sum()
        return torch.stack(
            (
                torch.stack((e.sum(), e.new_tensor(e.numel()))),
                torch.stack((f.sum(), e.new_tensor(f.numel()))),
                torch.stack((mm_sum, mm_count)),
            )
        ).detach()

    def validation_loss(self, components):
        floor = components.new_tensor([1.0, 1.0, 3.0])
        means = components[:, 0] / torch.maximum(components[:, 1], floor)
        return (
            torch.stack((self.energy_weight, self.forces_weight, self.mm_forces_weight))
            * means
        ).sum()

    def __init__(self, energy_weight=1.0, forces_weight=1.0, mm_forces_weight=1.0):
        super().__init__(energy_weight, forces_weight)
        self.register_buffer("mm_forces_weight", torch.tensor(float(mm_forces_weight)))

    def forward(self, ref, pred, ddp=None):
        if ddp:
            raise NotImplementedError("EnvHead training supports a single process")
        loss = super().forward(ref, pred, ddp)
        if getattr(ref, "mm_forces", None) is not None and ref.mm_forces.numel():
            weights = ref.mm_forces_weight
            loss = loss + self.mm_forces_weight * (
                (pred["mm_forces"] - ref.mm_forces).square() * weights
            ).sum() / (3 * weights.sum().clamp_min(1.0))
        return loss


def head_optimizer(model, learning_rate):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    expected = {id(p) for p in model.env_head.parameters()}
    actual = [id(p) for _, p in named]
    if (
        set(actual) != expected
        or len(actual) != len(set(actual))
        or any(not n.startswith("env_head.") for n, _ in named)
    ):
        raise ValueError("Trainable parameters must be exactly the EnvHead parameters")
    return torch.optim.Adam([p for _, p in named], lr=learning_rate)


class EnvTrainingLogger(MetricsLogger):
    """Record the optimizer's live LR without changing native loop scheduling."""

    def __init__(self, directory, optimizer):
        super().__init__(str(directory), "metrics")
        self.optimizer = optimizer

    def log(self, metrics):
        metrics = dict(metrics)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        if metrics.get("mode") == "eval":
            epoch = metrics.get("epoch")
            label = "Initial" if epoch is None else f"Epoch {epoch}"
            logging.info("%s: lr=%.6e", label, metrics["lr"])
        super().log(metrics)


def configure_plateau(scheduler):
    """Apply the current policy, including after an older checkpoint is loaded."""
    policy = dict(factor=0.5, patience=3, threshold=0.01,
                  threshold_mode="rel", cooldown=1, mode="min")
    minimum = [1e-5] * len(scheduler.optimizer.param_groups)
    changed = any(getattr(scheduler, key) != value for key, value in policy.items())
    changed = changed or scheduler.min_lrs != minimum
    for key, value in policy.items():
        setattr(scheduler, key, value)
    scheduler.min_lrs = minimum
    if hasattr(scheduler, "default_min_lr"):
        scheduler.default_min_lr = 1e-5
    if changed:
        # Old waiting counters used a different improvement criterion. Retain
        # the best loss, optimizer LR and epoch, but begin a fresh patience window.
        scheduler.num_bad_epochs = 0
        scheduler.cooldown_counter = 0


def fit(
    model,
    train_loader,
    valid_loaders,
    directory,
    max_epochs=10,
    learning_rate=1e-3,
    resume=False,
    loss_fn=None,
):
    if not valid_loaders:
        raise ValueError("At least one validation loader is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and (
        any(directory.glob("envmace_epoch-*.pt"))
        or (directory / "head.pt").exists()
        or any(directory.glob("metrics*"))
    ):
        raise FileExistsError("Existing run: use resume=True or a new directory")
    optimizer = head_optimizer(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, threshold=0.01,
        threshold_mode="rel", cooldown=1, min_lr=1e-5,
    )
    handler = EnvCheckpointHandler(directory=str(directory), tag="envmace", keep=True)
    state = CheckpointState(model, optimizer, scheduler)
    epoch = (
        handler.load_latest(state, device=model.r_max.device, strict=True)
        if resume
        else None
    )
    if resume and epoch is None:
        raise FileNotFoundError("No recovery checkpoint found")
    configure_plateau(scheduler)
    objective = (loss_fn or EnvHeadLoss()).to(model.r_max.device)
    output_args = {"forces": True, "virials": False, "stress": False}
    if epoch is not None and epoch + 1 < max_epochs:
        # Native train skips scheduler.step at start_epoch. A continuation needs
        # the update that an uninterrupted run would perform at this boundary.
        # Preserve RNG around this extra validation pass.
        rng = torch.get_rng_state()
        for loader in valid_loaders.values():
            valid_loss, _ = evaluate(
                model, objective, loader, output_args, model.r_max.device
            )
        torch.set_rng_state(rng)
        scheduler.step(valid_loss)
    logging.info(
        "Initial validation over %d batches; evaluates energy/forces without optimizer updates. Training batches follow.",
        sum(len(loader) for loader in valid_loaders.values()),
    )
    train(
        model=model,
        loss_fn=objective,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        start_epoch=0 if epoch is None else epoch + 1,
        max_num_epochs=max_epochs,
        patience=max_epochs + 1,
        checkpoint_handler=handler,
        logger=EnvTrainingLogger(directory, optimizer),
        eval_interval=1,
        output_args=output_args,
        device=model.r_max.device,
        log_errors="PerAtomRMSE",
        save_all_checkpoints=True,
    )
    save_head(directory / "head.pt", model)
    return model
