"""E/F training with optional atomic charge and background force supervision."""

import torch
import torch.distributed as dist


def mm_force_errors(ref, pred):
    """Weighted component MSE per labelled PC structure, plus raw component errors."""
    counts = ref.mm_count.flatten()
    owners = torch.repeat_interleave(
        torch.arange(len(counts), device=counts.device), counts,
        output_size=len(ref.mm_forces),
    )
    good = ref.mm_forces_weight.flatten() > 0
    selected = owners[good]
    errors = pred["mm_forces"][good] - ref.mm_forces[good]
    weights = ref.weight[selected] * ref.mm_forces_weight.flatten()[good]
    square = errors.new_zeros(len(counts)).index_add(
        0, selected, weights * errors.square().mean(-1)
    )
    number = counts.new_zeros(len(counts)).index_add(
        0, selected, torch.ones_like(selected)
    )
    return (square / number.clamp_min(1))[number > 0], errors


class EnvHeadLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=10.0, forces_weight=100.0, charges_weight=10.0,
        mm_forces_weight=0.0,
    ):
        super().__init__()
        self.energy_weight, self.forces_weight, self.charges_weight = (
            energy_weight,
            forces_weight,
            charges_weight,
        )
        self.mm_forces_weight = mm_forces_weight

    def forward(self, ref, pred, return_components=False, *, ddp=False):
        n = ref.ptr[1:] - ref.ptr[:-1]
        e = (
            ref.weight
            * ref.energy_weight
            * ((pred["energy"] - ref.energy) / n).square()
        ).mean()
        w = (ref.weight * ref.forces_weight)[ref.batch, None]
        f = (w * (pred["forces"] - ref.forces).square()).mean()
        charge_loss = e.new_zeros(())
        charge_errors = e.new_empty(0)
        good = ref.charges_weight[ref.batch] > 0
        if good.any():
            charge_errors = pred["charges"][good] - ref.charges.reshape(-1)[good]
            q = (ref.weight * ref.charges_weight)[ref.batch][
                good
            ] * charge_errors.square()
            charge_loss = self.charges_weight * q.mean()
        components = dict(
            energy_loss=self.energy_weight * e,
            forces_loss=self.forces_weight * f,
            charges_loss=charge_loss,
        )
        mm_count = 0
        if self.mm_forces_weight > 0:
            mm_mse, _ = mm_force_errors(ref, pred)
            mm_count = mm_mse.numel()
            components["mm_forces_loss"] = (
                self.mm_forces_weight * mm_mse.sum() / max(mm_count, 1)
            )
        loss = components["energy_loss"] + components["forces_loss"] + charge_loss
        if self.mm_forces_weight > 0:
            loss = loss + components["mm_forces_loss"]
        if ddp:
            # DDP averages rank gradients. Correct each target separately so a
            # rank with fewer atoms/labels has its proper global contribution.
            denominators = [ref.energy.numel(), ref.forces.numel(), charge_errors.numel()]
            if self.mm_forces_weight > 0:
                denominators.append(mm_count)
            counts = e.new_tensor(denominators)
            global_counts = counts.clone()
            dist.all_reduce(global_counts)
            scales = dist.get_world_size() * counts / global_counts.clamp_min(1)
            loss = sum(
                value * scale for value, scale in zip(components.values(), scales)
            )
        if return_components:
            return loss, components, charge_errors.detach()
        return loss


class TrainingObjective(torch.nn.Module):
    """Expose only the scalar objective's graph to DDP unused-parameter tracing."""

    def __init__(self, model, criterion, options, ddp=False):
        super().__init__()
        self.model, self.criterion = model, criterion
        self.options, self.ddp = options, ddp

    def forward(self, batch):
        prediction = self.model(
            batch.to_dict(), training=True, compute_charges=False,
            compute_mm_force=self.criterion.mm_forces_weight > 0, **self.options
        )
        loss, components, charge_errors = self.criterion(
            batch, prediction, return_components=True, ddp=self.ddp
        )
        return (
            loss,
            {k: v.detach() for k, v in components.items()},
            charge_errors,
            {k: v.detach() for k, v in prediction.items()},
        )


def epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    *,
    neighbor_backend="reference",
    execution_mode=None,
    on_batch=None,
    ema=None,
    clip_grad=None,
    distributed_model=None,
    context=None,
    compute_mm_force=False,
):
    train = optimizer is not None
    model.train(train)
    from ..modules.graph import resolve_neighbor_backend

    backend = resolve_neighbor_backend(
        neighbor_backend, next(model.parameters()).device
    )
    neighbor_options = dict(neighbor_backend=backend)
    mm_supervised = criterion.mm_forces_weight > 0
    mm_requested = compute_mm_force or mm_supervised
    if mm_requested:
        if distributed_model is not None and not mm_supervised:
            raise ValueError("Request MM-force diagnostics in evaluation, outside DDP training")
        neighbor_options["compute_mm_force"] = True
    if execution_mode is not None:
        neighbor_options["execution_mode"] = execution_mode
    totals = dict(
        energy_sq=0.0,
        forces_sq=0.0,
        energy_n=0,
        forces_n=0,
        energy_label_n=0,
        forces_label_n=0,
        energy_per_atom_sq=0.0,
        energy_loss=0.0,
        forces_loss=0.0,
        charges_loss=0.0,
        charges_sq=0.0,
        charges_n=0,
    )
    if mm_supervised:
        totals.update(mm_forces_loss=0.0, mm_structures=0, mm_forces_sq=0.0, mm_components=0)
    for batch in loader:
        batch = batch.to(next(model.parameters()).device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        if train and distributed_model is not None:
            loss, components, charge_errors, pred = distributed_model(batch)
        else:
            pred = model(
                batch.to_dict(),
                training=train,
                compute_charges=False,  # Always compute charges; skip extra diagnostics.
                **neighbor_options,
            )
            loss, components, charge_errors = criterion(
                batch, pred, return_components=True
            )
        if train:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if ema is not None:
                ema.update()
        counts = dict(
            energy_loss=batch.energy.numel(),
            forces_loss=batch.forces.numel(),
            charges_loss=charge_errors.numel(),
        )
        if mm_supervised:
            with torch.no_grad():
                mm_mse, mm_errors = mm_force_errors(batch, pred)
                counts["mm_forces_loss"] = mm_mse.numel()
                totals["mm_structures"] += mm_mse.numel()
                totals["mm_forces_sq"] += float(mm_errors.double().square().sum())
                totals["mm_components"] += mm_errors.numel()
            del mm_mse, mm_errors
        for name, value in components.items():
            totals[name] += float(value.detach()) * counts[name]
        totals["charges_sq"] += float(charge_errors.square().sum())
        totals["charges_n"] += charge_errors.numel()
        energy_mask = batch.energy_weight > 0
        force_mask = batch.forces_weight[batch.batch] > 0
        e = (pred["energy"].detach() - batch.energy)[energy_mask]
        f = (pred["forces"].detach() - batch.forces)[force_mask]
        totals["energy_sq"] += float(e.double().square().sum())
        totals["energy_per_atom_sq"] += float(
            (e.double() / (batch.ptr[1:] - batch.ptr[:-1])[energy_mask]).square().sum()
        )
        totals["forces_sq"] += float(f.square().sum())
        totals["energy_n"] += counts["energy_loss"]
        totals["forces_n"] += counts["forces_loss"]
        totals["energy_label_n"] += e.numel()
        totals["forces_label_n"] += f.numel()
        # Export evaluation details from the same predictions used for metrics.
        # Observers must detach retained tensors; ordinary training has no observer.
        if on_batch is not None:
            on_batch(batch, pred)
        # Evaluation forces retain an energy graph. Drop the previous batch's
        # outputs before allocating the next forward, including component losses.
        del pred, loss, components, value, charge_errors, e, f
    if context is not None and context.enabled:
        packed = torch.tensor(
            list(totals.values()),
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        dist.all_reduce(packed)
        totals = dict(zip(totals, packed.cpu().tolist()))
    components = {
        "energy_loss": totals["energy_loss"] / max(totals["energy_n"], 1),
        "forces_loss": totals["forces_loss"] / max(totals["forces_n"], 1),
        "charges_loss": totals["charges_loss"] / max(totals["charges_n"], 1),
    }
    if mm_supervised:
        components["mm_forces_loss"] = totals["mm_forces_loss"] / max(totals["mm_structures"], 1)
    metrics = dict(
        loss=sum(components.values()),
        **components,
        energy_rmse_per_atom=(totals["energy_per_atom_sq"] / totals["energy_label_n"])
        ** 0.5
        if totals["energy_label_n"]
        else None,
        charges_rmse=(totals["charges_sq"] / totals["charges_n"]) ** 0.5
        if totals["charges_n"]
        else None,
        energy_rmse=(totals["energy_sq"] / totals["energy_label_n"]) ** 0.5
        if totals["energy_label_n"]
        else None,
        forces_rmse=(totals["forces_sq"] / totals["forces_label_n"]) ** 0.5
        if totals["forces_label_n"]
        else None,
    )
    if mm_supervised:
        metrics["mm_forces_rmse"] = (
            (totals["mm_forces_sq"] / totals["mm_components"]) ** 0.5
            if totals["mm_components"] else None
        )
    return metrics
