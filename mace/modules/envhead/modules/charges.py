"""QM-only charge reconstruction; objectives are not physical energy terms."""

import math
import numpy as np
import torch
from e3nn import o3
from ase.data import vdw_radii
from .encoders import COULOMB
from .graph import envelope


HARTREE_EV = 27.211386245988


def gaussian_coulomb(r, width):
    eye = torch.eye(len(r), device=r.device, dtype=r.dtype)
    d = torch.sqrt((r[:, None] - r[None]).square().sum(-1) + eye)
    diagonal = COULOMB / (math.sqrt(math.pi) * width)
    return COULOMB * torch.erf(d / (2 * width)) / d * (1 - eye) + diagonal * eye


def metric(r, width, hardness):
    j = gaussian_coulomb(r, width)
    diagonal = COULOMB / (math.sqrt(math.pi) * width) + hardness
    return (j + hardness * torch.eye(len(r), device=r.device, dtype=r.dtype)) / diagonal


def constrained_solve(a, rhs, total):
    """SPD Schur complement, enforcing one QM total-charge constraint."""
    ones = torch.ones_like(rhs)
    both = torch.linalg.solve(a, torch.stack((rhs, ones), -1))
    x, z = both.unbind(-1)
    return x + z * ((total - x.sum()) / z.sum())


class ChargeReconstruction(torch.nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.hidden, self.config = hidden, config
        if config.charge_mode in (
            "charge_projection",
            "iterative_response",
            "fukui_projection",
        ):
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(config.channels, config.channels),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    config.channels,
                    2 if config.charge_mode == "fukui_projection" else 1,
                ),
            )
            if config.charge_mode == "fukui_projection":
                with torch.no_grad():
                    self.readout[-1].weight[1].zero_()
                    self.readout[-1].bias[1].fill_(1)
        if config.charge_mode == "iterative_response":
            self.response = torch.nn.Sequential(
                torch.nn.Linear(config.channels + 3, config.channels),
                torch.nn.SiLU(),
                torch.nn.Linear(config.channels, 1),
            )
            torch.nn.init.zeros_(self.response[-1].weight)
            torch.nn.init.zeros_(self.response[-1].bias)
        if config.charge_mode == "esp_reconstruction":
            from scipy.integrate import lebedev_rule

            points, weights = lebedev_rule(config.esp_order)
            if np.any(weights <= 0):
                raise ValueError("ESP fitting requires positive quadrature weights")
            self.register_buffer("directions", torch.tensor(points.T.copy()))
            self.register_buffer(
                "quadrature", torch.tensor(weights.copy() / weights.sum())
            )
            self.register_buffer("vdw", torch.tensor(vdw_radii.copy()))
            self.query = torch.nn.Sequential(
                torch.nn.Linear(hidden.num_irreps, config.channels),
                torch.nn.SiLU(),
                torch.nn.Linear(config.channels, 1, bias=False),
            )
            self.sh_irreps = o3.Irreps.spherical_harmonics(config.max_l)

    def samples(self, r, z):
        radius = self.vdw[z].to(r)
        if not torch.isfinite(radius).all():
            raise ValueError("ESP grid needs tabulated vdW radii for all QM elements")
        shells = r.new_tensor(self.config.esp_shells)
        offsets = (
            radius[:, None, None, None]
            * shells[None, :, None, None]
            * self.directions.to(r)[None, None]
        )
        points = (r[:, None, None] + offsets).reshape(-1, 3)
        weights = self.quadrature.to(r).repeat(len(r) * len(shells)) / (
            len(r) * len(shells)
        )
        return points, weights, radius

    def grid_block(self, points, weights, radius, r):
        dr = points[:, None] - r[None]
        # Distances below vdW radii have zero fitting weight. A small interior
        # floor prevents a coincident, excluded point from generating NaNs.
        d = dr.square().sum(-1).clamp_min(1e-12).sqrt()
        exclusion = 1 - envelope((d / radius[None] - 1) / 0.2)
        w = weights * exclusion.prod(-1)
        return dr, d, w

    def target_potential(self, h, dr, d):
        direction = dr / d[..., None]
        sh = o3.spherical_harmonics(
            self.sh_irreps, direction, normalize=False, normalization="component"
        )
        features, offset = [], 0
        for (mul, ir), sl in zip(self.hidden, self.hidden.slices()):
            block = h[:, sl].reshape(len(h), mul, ir.dim)
            angular = (block[None] * sh[:, :, None, offset : offset + ir.dim]).sum(-1)
            features.append((angular / d[..., None] ** (ir.l + 1)).sum(1))
            offset += ir.dim
        return self.query(torch.cat(features, -1)).squeeze(-1)

    def prepare(self, data):
        """Build QM-only geometry once for vacuum and embedded reconstruction."""
        geometry = []
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
            geometry.append((j, z / z.sum()))
        return geometry

    @staticmethod
    def project(raw, z, total):
        return raw + z * (total - raw.sum())

    def forward(self, h, data, diagnostics=False, potential=None, geometry=None):
        if self.config.charge_mode == "fukui_projection":
            return self.allocate(h, data, diagnostics)
        r = data["positions"]
        totals = data["total_charge"].reshape(-1)
        outputs, residuals, conditions, steps = [], [], [], []
        projection = self.config.charge_mode in (
            "charge_projection",
            "iterative_response",
        )
        if projection and geometry is None:
            geometry = self.prepare(data)
        if potential is None:
            potential = h.new_zeros(len(h))
        for g in range(len(totals)):
            lo, hi = int(data["ptr"][g]), int(data["ptr"][g + 1])
            rg, hg, total = r[lo:hi].double(), h[lo:hi], totals[g].double()
            n = hi - lo
            if projection:
                j, z = geometry[g]
                raw = self.readout(hg[:, : self.config.channels]).flatten().double()
                q = self.project(raw, z, total)
                if self.config.charge_mode == "iterative_response":
                    initial = q
                    last_step = q.new_zeros(())
                    for _ in range(self.config.charge_iterations):
                        features = torch.cat(
                            (
                                hg[:, : self.config.channels],
                                q.to(hg)[:, None],
                                ((j @ q) / HARTREE_EV).to(hg)[:, None],
                                (potential[lo:hi] / HARTREE_EV).to(hg).reshape(-1, 1),
                            ),
                            dim=-1,
                        )
                        proposal = self.project(
                            initial + self.response(features).flatten().double(),
                            z,
                            total,
                        )
                        updated = q + self.config.charge_mixing * (proposal - q)
                        last_step = (updated - q).abs().max()
                        q = updated
                    steps.append(last_step)
            elif n == 1:
                q = total.reshape(1) + hg.sum() * 0
                residuals.append(total.new_zeros(()))
                conditions.append(total.new_ones(()))
            else:
                prior = metric(
                    rg, self.config.projection_width, self.config.projection_hardness
                )
                points, weights, radii = self.samples(rg, data["atomic_numbers"][lo:hi])
                a, rhs, wsum = rg.new_zeros(n, n), rg.new_zeros(n), rg.new_zeros(())
                for start in range(0, len(points), self.config.esp_chunk_size):
                    dr, d, w = self.grid_block(
                        points[start : start + self.config.esp_chunk_size],
                        weights[start : start + self.config.esp_chunk_size],
                        radii,
                        rg,
                    )
                    k = COULOMB / d
                    v = self.target_potential(hg, dr.to(hg), d.to(hg)).double()
                    a = a + k.T @ (w[:, None] * k)
                    rhs = rhs + k.T @ (w * v)
                    wsum = wsum + w.sum()
                # Trace normalization makes eta dimensionless. Differentiate
                # the normalization as well as the moving-grid geometry.
                scale = a.diagonal().mean()
                reg = self.config.projection_regularization
                q0 = torch.ones_like(rhs) * total / n
                q = constrained_solve(
                    a / scale + reg * prior, rhs / scale + reg * (prior @ q0), total
                )
                if diagnostics:
                    # Evaluate the residual directly: the normal-equation
                    # identity catastrophically cancels for accurate fits.
                    with torch.no_grad():
                        error = rg.new_zeros(())
                        for start in range(0, len(points), self.config.esp_chunk_size):
                            dr, d, w = self.grid_block(
                                points[start : start + self.config.esp_chunk_size],
                                weights[start : start + self.config.esp_chunk_size],
                                radii,
                                rg,
                            )
                            v = self.target_potential(hg, dr.to(hg), d.to(hg)).double()
                            error += (w * ((COULOMB / d) @ q - v).square()).sum()
                        residuals.append((error / wsum).sqrt())
                        p = (
                            torch.eye(n, device=rg.device, dtype=rg.dtype)
                            - rg.new_ones(n, n) / n
                        )
                        eigenvalues = torch.linalg.eigvalsh(p @ a.detach() @ p)
                        conditions.append(
                            (
                                eigenvalues[-1]
                                / eigenvalues[1].clamp_min(
                                    torch.finfo(rg.dtype).eps * eigenvalues[-1]
                                )
                            ).sqrt()
                        )
            outputs.append(q.to(h))
        result = {"charges": torch.cat(outputs)}
        if diagnostics and steps:
            result["charge_last_step"] = torch.stack(steps).to(h)
        if diagnostics and residuals:
            result["charge_esp_rmse"] = torch.stack(residuals).to(h)
            result["charge_condition"] = torch.stack(conditions).to(h)
        return result

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

    def project_batched(self, h, data, geometry=None):
        """Same fixed-metric projection; one NN call and device-side reduction."""
        if geometry is None:
            geometry = self.prepare(data)
        z = torch.cat([entry[1] for entry in geometry])
        raw = self.readout(h[:, : self.config.channels]).flatten().double()
        batch = data["batch"]
        total = data["total_charge"].reshape(-1).double()
        sums = raw.new_zeros(len(total)).index_add(0, batch, raw)
        return {"charges": (raw + z * (total - sums)[batch]).to(h)}


class ChargeReconstructionV4(ChargeReconstruction):
    """Active v4 heads retain normal nn.Module tracing and registration."""

    def forward(self, h, data, diagnostics=False, potential=None, geometry=None):
        if self.config.charge_mode == "charge_projection":
            return self.project_batched(h, data, geometry)
        return self.allocate(h, data, diagnostics)
