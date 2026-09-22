"""E/F training with optional derivative-based atomic charge supervision."""

import torch


class EnvHeadLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, charges_weight=0.0):
        super().__init__()
        self.energy_weight, self.forces_weight, self.charges_weight = (
            energy_weight,
            forces_weight,
            charges_weight,
        )

    def forward(self, ref, pred):
        n = ref.ptr[1:] - ref.ptr[:-1]
        e = (
            ref.weight
            * ref.energy_weight
            * ((pred["energy"] - ref.energy) / n).square()
        ).mean()
        w = (ref.weight * ref.forces_weight)[ref.batch, None]
        f = (w * (pred["forces"] - ref.forces).square()).mean()
        loss = self.energy_weight * e + self.forces_weight * f
        if self.charges_weight:
            good = pred["charges_valid"][ref.batch] & (
                ref.charges_weight[ref.batch] > 0
            )
            if good.any():
                q = (ref.weight * ref.charges_weight)[ref.batch][good] * (
                    pred["charges"][good] - ref.charges.reshape(-1)[good]
                ).square()
                loss = loss + self.charges_weight * q.mean()
        return loss


def epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = dict(
        energy_sq=0.0, forces_sq=0.0, energy_n=0, forces_n=0, loss=0.0, batches=0
    )
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(
            batch.to_dict(),
            training=train,
            compute_charges=criterion.charges_weight > 0,
        )
        loss = criterion(batch, pred)
        if train:
            loss.backward()
            optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["batches"] += 1
        e = pred["energy"].detach() - batch.energy
        f = pred["forces"].detach() - batch.forces
        totals["energy_sq"] += float(e.square().sum())
        totals["forces_sq"] += float(f.square().sum())
        totals["energy_n"] += e.numel()
        totals["forces_n"] += f.numel()
    return dict(
        loss=totals["loss"] / max(totals["batches"], 1),
        energy_rmse=(totals["energy_sq"] / max(totals["energy_n"], 1)) ** 0.5,
        forces_rmse=(totals["forces_sq"] / max(totals["forces_n"], 1)) ** 0.5,
    )
