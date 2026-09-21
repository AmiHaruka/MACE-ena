"""Analytic Gaussian q/mu/Theta kernels in e, Angstrom, eV.

rho = q*g - mu_a*d_a g + (1/2)*Theta_ab*d_a*d_b g.
Theta is the traceless second moment (NOT 3rr-r^2 I). Five coefficients
use an orthonormal e3nn 2e basis. Dipole components are Cartesian xyz.
Internal features exclude the entire same-atom block. No internal energy is
added to the foundation. All derivative formulas below are analytic Torch
operations, so coordinate and force-training derivatives remain available.
"""
import math
import torch
from e3nn import o3

COULOMB_EV_ANGSTROM = 14.3996454784255


def quadrupole_basis(like):
    return o3.wigner_3j(1, 1, 2, dtype=like.dtype, device=like.device) * math.sqrt(5)


def quadrupole_tensor(coefficients):
    return torch.einsum("...k,abk->...ab", coefficients, quadrupole_basis(coefficients))


def _radial_values(t, sigma, order):
    """Evaluate the same 32-term near-origin polynomial with Horner's rule."""
    b = t / (2 * sigma * sigma)
    safe = b.clamp_min(2.0)
    root = safe.sqrt()
    integral = 0.5 * math.sqrt(math.pi) * torch.erf(root) / root
    exponential = torch.exp(-safe)
    bs = b.clamp_max(2.0)
    values = []
    for n in range(order + 1):
        if n:
            integral = ((2 * n - 1) * integral - exponential) / (2 * safe)
        coefficient = lambda j: (-1.0) ** j / (math.factorial(j) * (2 * (n + j) + 1))
        series = torch.full_like(t, coefficient(31))
        for j in range(30, -1, -1):
            series = series * bs + coefficient(j)
        values.append(
            math.sqrt(2 / math.pi)
            / sigma
            * (-1 / (2 * sigma * sigma)) ** n
            * torch.where(b < 2.0, series, integral)
        )
    return torch.stack(values)


class _RadialDerivatives(torch.autograd.Function):
    """Save only t; df^(n)/dt=f^(n+1) also supports mixed derivatives.

    Sigma is a fixed configuration value, never a learned tensor. Recursive
    application in backward retains differentiability for force training/probes.
    The 32-term series approximates the same analytic kernel as the old path.
    """

    @staticmethod
    def forward(ctx, t, sigma, order):
        ctx.save_for_backward(t)
        ctx.sigma, ctx.order = sigma, order
        return _radial_values(t, sigma, order)

    @staticmethod
    def backward(ctx, grad_output):
        (t,) = ctx.saved_tensors
        higher = _RadialDerivatives.apply(t, ctx.sigma, ctx.order + 1)[1:]
        return (grad_output * higher).sum(0), None, None


def _radial_derivatives(t, sigma, order=4):
    return _RadialDerivatives.apply(t, sigma, order).unbind(0)


def derivatives(r, sigma, order=4):
    f = _radial_derivatives(r.square().sum(-1), sigma, order)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    d = [f[0], 2 * f[1][..., None] * r]
    if order < 2:
        return d
    d.append(
        4 * f[2][..., None, None] * torch.einsum("...a,...b->...ab", r, r)
        + 2 * f[1][..., None, None] * eye
    )
    if order < 3:
        return d
    s3 = sum(
        torch.einsum(eq, eye, r)
        for eq in ("ab,...c->...abc", "ac,...b->...abc", "bc,...a->...abc")
    )
    d.append(
        8
        * f[3][..., None, None, None]
        * torch.einsum("...a,...b,...c->...abc", r, r, r)
        + 4 * f[2][..., None, None, None] * s3
    )
    if order < 4:
        return d
    s4 = sum(
        torch.einsum(eq, eye, r, r)
        for eq in (
            "ab,...c,...d->...abcd",
            "ac,...b,...d->...abcd",
            "ad,...b,...c->...abcd",
            "bc,...a,...d->...abcd",
            "bd,...a,...c->...abcd",
            "cd,...a,...b->...abcd",
        )
    )
    ss = sum(
        torch.einsum(eq, eye, eye)
        for eq in ("ab,cd->abcd", "ac,bd->abcd", "ad,bc->abcd")
    )
    d.append(
        16
        * f[4][..., None, None, None, None]
        * torch.einsum("...a,...b,...c,...d->...abcd", r, r, r, r)
        + 8 * f[3][..., None, None, None, None] * s4
        + 4 * f[2][..., None, None, None, None] * ss
    )
    return d


def _operators(like, max_l):
    ops = [like.new_ones(1, 1), torch.eye(3, dtype=like.dtype, device=like.device)]
    if max_l == 2:
        ops.append(0.5 * quadrupole_basis(like).reshape(9, 5).T)
    return ops


def multipole_kernel(positions, width=1.0, max_l=2):
    r = positions[:, None] - positions[None, :]
    ds = derivatives(r, math.sqrt(2) * width, 2 * max_l)
    ops = _operators(r, max_l)
    rows = []
    for a, oa in enumerate(ops):
        blocks = []
        for b, ob in enumerate(ops):
            derivative = ds[a + b].reshape(len(r), len(r), 3**a, 3**b)
            blocks.append(
                (-1) ** b * torch.einsum("ux,ijxy,vy->ijuv", oa, derivative, ob)
            )
        rows.append(torch.cat(blocks, -1))
    result = torch.cat(rows, -2)
    mask = 1 - torch.eye(len(r), dtype=r.dtype, device=r.device)
    return COULOMB_EV_ANGSTROM * (result * mask[..., None, None]).permute(
        0, 2, 1, 3
    ).reshape(len(r) * (max_l + 1) ** 2, -1)


def source_features(
    positions, mm_positions, mm_charges, width=1.0, max_l=2, chunk_size=1024,
    cell=None, rf_cutoff=None, rf_dielectric=78.4,
):
    out = positions.new_zeros(len(positions), (max_l + 1) ** 2)
    ops = _operators(positions, max_l)
    if (cell is None) != (rf_cutoff is None):
        raise ValueError("Supply both mm cell and RF cutoff")
    if cell is not None:
        from .periodic import box_lengths, rf_derivatives, rf_coefficients
        rf_coefficients(rf_cutoff, rf_dielectric)
        lengths = box_lengths(cell, rf_cutoff)
        # The switched RF kernel is exactly zero at and beyond the cutoff, so
        # keeping only shorter pairs is exact. Filter first, then evaluate the
        # kernel on the kept pairs in one shot (AMP-BMS neighbor-list idea
        # without persistent state in the model).
        rows, cols = [], []
        with torch.no_grad():
            for start in range(0, len(mm_positions), chunk_size):
                r = positions[:, None] - mm_positions[None, start : start + chunk_size]
                r = r - torch.round(r / lengths) * lengths
                inside = torch.nonzero(r.square().sum(-1) < rf_cutoff * rf_cutoff)
                rows.append(inside[:, 0])
                cols.append(inside[:, 1] + start)
        if rows:
            rows, cols = torch.cat(rows), torch.cat(cols)
        if len(rows):
            r = positions[rows] - mm_positions[cols]
            r = r - torch.round(r / lengths) * lengths
            ds = rf_derivatives(r, width, rf_cutoff, rf_dielectric)
            terms = [
                torch.einsum("ux,ix->iu", op, ds[l].reshape(len(r), -1))
                for l, op in enumerate(ops)
            ]
            out = out.index_add(
                0, rows, torch.cat(terms, -1) * mm_charges[cols].reshape(-1, 1)
            )
        return COULOMB_EV_ANGSTROM * out
    for start in range(0, len(mm_positions), chunk_size):
        r = positions[:, None] - mm_positions[None, start : start + chunk_size]
        ds = derivatives(r, width, max_l)
        terms = [
            torch.einsum(
                "ux,ijx->iju", op, ds[l].reshape(len(positions), r.shape[1], -1)
            )
            for l, op in enumerate(ops)
        ]
        out = out + (
            torch.cat(terms, -1)
            * mm_charges[start : start + chunk_size].reshape(1, -1, 1)
        ).sum(1)
    return COULOMB_EV_ANGSTROM * out
