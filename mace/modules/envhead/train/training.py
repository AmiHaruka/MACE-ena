"""E/F training with optional reconstructed atomic charge supervision."""

import torch


class EnvHeadLoss(torch.nn.Module):
    def __init__(self, energy_weight=10.0, forces_weight=100.0, charges_weight=10.0):
        super().__init__()
        self.energy_weight, self.forces_weight, self.charges_weight = (
            energy_weight,
            forces_weight,
            charges_weight,
        )

    def forward(self, ref, pred, return_components=False):
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
        loss = components["energy_loss"] + components["forces_loss"] + charge_loss
        if return_components:
            return loss, components, charge_errors.detach()
        return loss


def epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = dict(
        energy_sq=0.0,
        forces_sq=0.0,
        energy_n=0,
        forces_n=0,
        energy_per_atom_sq=0.0,
        energy_loss=0.0,
        forces_loss=0.0,
        charges_loss=0.0,
        charges_sq=0.0,
        charges_n=0,
    )
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(
            batch.to_dict(),
            training=train,
            compute_charges=False,  # charges are always computed; skip optional ESP diagnostics
        )
        loss, components, charge_errors = criterion(batch, pred, return_components=True)
        if train:
            loss.backward()
            optimizer.step()
        counts = dict(
            energy_loss=batch.energy.numel(),
            forces_loss=batch.forces.numel(),
            charges_loss=charge_errors.numel(),
        )
        for name, value in components.items():
            totals[name] += float(value.detach()) * counts[name]
        totals["charges_sq"] += float(charge_errors.square().sum())
        totals["charges_n"] += charge_errors.numel()
        e = pred["energy"].detach() - batch.energy
        f = pred["forces"].detach() - batch.forces
        totals["energy_sq"] += float(e.double().square().sum())
        totals["energy_per_atom_sq"] += float(
            (e.double() / (batch.ptr[1:] - batch.ptr[:-1])).square().sum()
        )
        totals["forces_sq"] += float(f.square().sum())
        totals["energy_n"] += e.numel()
        totals["forces_n"] += f.numel()
        # Evaluation forces retain an energy graph. Drop the previous batch's
        # outputs before allocating the next forward, including component losses.
        del pred, loss, components, value, charge_errors, e, f
    components = {
        "energy_loss": totals["energy_loss"] / max(totals["energy_n"], 1),
        "forces_loss": totals["forces_loss"] / max(totals["forces_n"], 1),
        "charges_loss": totals["charges_loss"] / max(totals["charges_n"], 1),
    }
    return dict(
        loss=sum(components.values()),
        **components,
        energy_rmse_per_atom=(totals["energy_per_atom_sq"] / max(totals["energy_n"], 1))
        ** 0.5,
        charges_rmse=(totals["charges_sq"] / totals["charges_n"]) ** 0.5
        if totals["charges_n"]
        else None,
        energy_rmse=(totals["energy_sq"] / max(totals["energy_n"], 1)) ** 0.5,
        forces_rmse=(totals["forces_sq"] / max(totals["forces_n"], 1)) ** 0.5,
    )
