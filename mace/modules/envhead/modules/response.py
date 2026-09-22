"""Response-defined decomposition and hardness-regularized ESP reconstruction."""

import math
import torch
from .encoders import COULOMB, sum_nodes


def reconstruct(data, potential, config):
    positions, mm = data["positions"], data["mm_positions"]
    charges = torch.full_like(positions[:, 0], torch.nan)
    valid, errors, conditions = [], [], []
    start = 0
    for g, count in enumerate(data["mm_count"].reshape(-1).tolist()):
        lo, hi = int(data["ptr"][g]), int(data["ptr"][g + 1])
        r, sources = positions[lo:hi], mm[start : start + count]
        v = potential[start : start + count]
        start += count
        n = len(r)
        if count < max(1, n - 1):
            valid.append(False)
            errors.append(r.new_tensor(float("nan")))
            conditions.append(r.new_tensor(float("inf")))
            continue
        distance = torch.linalg.vector_norm(sources[:, None] - r[None, :], dim=-1)
        k = COULOMB / distance.clamp_min(1e-8)
        # Only the charge-conserving subspace needs to be identifiable.
        centered = k - k.mean(1, keepdim=True)
        singular = torch.linalg.svdvals(centered.detach())
        if n > 1:
            condition = singular[0] / singular[n - 2].clamp_min(
                torch.finfo(k.dtype).eps
            )
            good = bool(
                (singular[n - 2] > 1e-10)
                & (condition < config.projection_condition_limit)
            )
        else:
            condition, good = k.new_ones(()), True
        conditions.append(condition)
        if not good:
            valid.append(False)
            errors.append(r.new_tensor(float("nan")))
            continue
        # Gaussian self interaction has a finite analytic diagonal. Avoid sqrt
        # at a zero displacement in the differentiable hardness construction.
        d2 = (r[:, None] - r[None, :]).square().sum(-1)
        eye = torch.eye(n, dtype=r.dtype, device=r.device)
        d = torch.sqrt(d2 + eye)
        width = config.projection_width
        h = COULOMB * torch.erf(d / (2 * width)) / d
        h = h * (1 - eye) + eye * (
            COULOMB / (math.sqrt(math.pi) * width) + config.projection_hardness
        )
        h = h / (COULOMB / (math.sqrt(math.pi) * width) + config.projection_hardness)
        scale = k.detach().square().mean().sqrt().clamp_min(1e-8)
        kn, vn = k / scale, v / scale
        reg = config.projection_regularization
        a = kn.T @ kn / count + reg * h
        total = data["total_charge"].reshape(-1)[g]
        q0 = r.new_ones(n) * total / n
        rhs = kn.T @ vn / count + reg * (h @ q0)
        ones = r.new_ones(n, 1)
        bordered = torch.cat(
            (torch.cat((a, ones), 1), torch.cat((ones.T, r.new_zeros(1, 1)), 1)), 0
        )
        q = torch.linalg.solve(bordered, torch.cat((rhs, total.reshape(1))))[:n]
        charges[lo:hi] = q
        valid.append(True)
        errors.append((k @ q - v).square().mean().sqrt())
    return dict(
        charges=charges,
        charges_valid=torch.tensor(valid, device=positions.device),
        charge_esp_rmse=torch.stack(errors),
        charge_condition=torch.stack(conditions),
    )


def response_outputs(energy, data, config, training=False, compute_charges=False):
    b = data["mm_charges"]
    potential = torch.autograd.grad(
        energy.sum(), b, create_graph=training, retain_graph=True, allow_unused=True
    )[0]
    if potential is None:
        potential = b * 0
    potential = potential.flatten()
    ids = torch.repeat_interleave(
        torch.arange(len(energy), device=b.device), data["mm_count"].reshape(-1)
    )
    coupling = sum_nodes(potential * b.flatten(), ids, len(energy))
    result = dict(
        response_potential=potential,
        interaction_energy=coupling,
        internal_energy=energy - coupling,
    )
    if compute_charges:
        result.update(reconstruct(data, potential, config))
    return result
