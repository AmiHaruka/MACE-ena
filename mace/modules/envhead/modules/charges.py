"""Total-charge-conserving allocation and fixed-metric projection."""

import math
import torch
from .electrostatics import COULOMB


def gaussian_coulomb(r, width):
    eye = torch.eye(len(r), device=r.device, dtype=r.dtype)
    d = torch.sqrt((r[:, None] - r[None]).square().sum(-1) + eye)
    diagonal = COULOMB / (math.sqrt(math.pi) * width)
    return COULOMB * torch.erf(d / (2 * width)) / d * (1 - eye) + diagonal * eye


class ChargeReconstruction(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(config.channels, config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(
                config.channels, 2 if config.charge_mode == "fukui_projection" else 1
            ),
        )
        if config.charge_mode == "fukui_projection":
            with torch.no_grad():
                self.readout[-1].weight[1].zero_()
                self.readout[-1].bias[1].fill_(1)

    def forward(self, h, data, diagnostics=False):
        if self.config.charge_mode == "charge_projection":
            return self.project_batched(h, data)
        return self.allocate(h, data, diagnostics)

    def prepare(self, data):
        """Return each atom's normalized fixed-metric correction weight."""
        weights = []
        r = data["positions"]
        for lo, hi in zip(data["ptr"][:-1].tolist(), data["ptr"][1:].tolist()):
            rg = r[lo:hi].double()
            j = gaussian_coulomb(rg, self.config.projection_width)
            hardness = self.config.projection_hardness
            h = (j + hardness * torch.eye(len(rg), device=r.device, dtype=rg.dtype)) / (
                j[0, 0] + hardness
            )
            z = torch.linalg.solve(
                h, torch.ones(len(rg), device=r.device, dtype=rg.dtype)
            )
            weights.append(z / z.sum())
        return torch.cat(weights)

    def allocate(self, h, data, diagnostics=False):
        """Nonnegative, normalized allocation of the total-charge correction.

        All atomwise NN work is batched; double reductions retain conservation
        accuracy without constructing any pair matrix or solving a system.
        """
        raw, source = self.readout(h[:, : self.config.channels]).double().unbind(-1)
        batch = data["batch"]
        total = data["total_charge"].reshape(-1).double()
        f = source.square() + 1e-6
        sums = raw.new_zeros(len(total), 2).index_add(
            0, batch, torch.stack((raw, f), dim=-1)
        )
        weights = f / sums[batch, 1]
        q = raw + weights * (total - sums[:, 0])[batch]
        result = {"charges": q.to(h)}
        if diagnostics:
            result["charge_weights"] = weights.to(h)
        return result

    def project_batched(self, h, data):
        """Same fixed-metric projection; one NN call and device-side reduction."""
        z = self.prepare(data)
        raw = self.readout(h[:, : self.config.channels]).flatten().double()
        batch = data["batch"]
        total = data["total_charge"].reshape(-1).double()
        sums = raw.new_zeros(len(total)).index_add(0, batch, raw)
        return {"charges": (raw + z * (total - sums)[batch]).to(h)}
