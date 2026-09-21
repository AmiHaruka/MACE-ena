"""Orthorhombic minimum images and smooth Gaussian reaction-field embedding.

RF coefficients follow AMP-BMS Modules.CoulombQMMM (m=4,n=6). A C4 switch
over the outer 20% of the radius keeps quadrupole forces/Hessians continuous.
All density couplings are derivatives of this single scalar kernel.
"""
import math
import torch


def rf_coefficients(cutoff, dielectric):
    if not math.isfinite(cutoff) or cutoff <= 0 or not math.isfinite(dielectric) or dielectric < 1:
        raise ValueError("Require positive finite RF cutoff and dielectric >= 1")
    k = (dielectric - 1) / ((1 + 2 * dielectric) * cutoff**3)
    a = 3 * (2 * dielectric + 5) / (8 * (1 + 2 * dielectric) * cutoff**5)
    b = -3 * (2 * dielectric + 3) / (12 * (1 + 2 * dielectric) * cutoff**7)
    c = 3 * dielectric / ((1 + 2 * dielectric) * cutoff) + a*cutoff**4 + b*cutoff**6
    return k, a, b, c


def box_lengths(cell, cutoff):
    if cell.shape != (3, 3) or not torch.isfinite(cell).all():
        raise ValueError("Expected a finite orthorhombic cell [3,3]")
    lengths = cell.diagonal()
    if not torch.allclose(cell, torch.diag(lengths), atol=1e-10, rtol=0):
        raise NotImplementedError("Only axis-aligned orthorhombic cells are supported")
    if (lengths <= 2*cutoff).any():
        raise ValueError("RF cutoff must be strictly below half every box length")
    return lengths


def rf_derivatives(r, sigma, cutoff, dielectric):
    from .electrostatics import _radial_derivatives

    t = r.square().sum(-1)
    f = list(_radial_derivatives(t, sigma, 2))
    k, a, b, c = rf_coefficients(cutoff, dielectric)
    f[0] = f[0] + k*t + a*t*t + b*t*t*t - c
    f[1] = f[1] + k + 2*a*t + 3*b*t*t
    f[2] = f[2] + 2*a + 6*b*t
    lower, upper = (.8*cutoff)**2, cutoff**2
    x = ((t-lower)/(upper-lower)).clamp(0, 1)
    # Factorization avoids cancellation near x=1 in float32.
    raw = [(1-x)**5*(1+5*x+15*x*x+35*x**3+70*x**4),
           -630*x**4*(1-x)**4/(upper-lower),
           -2520*x**3*(1-x)**3*(1-2*x)/(upper-lower)**2]
    s = [torch.where(t >= upper, torch.zeros_like(t), value)
         for value in raw]
    g0 = f[0]*s[0]
    g1 = f[1]*s[0] + f[0]*s[1]
    g2 = f[2]*s[0] + 2*f[1]*s[1] + f[0]*s[2]
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return [g0, 2*g1[..., None]*r,
            4*g2[..., None, None]*r[..., :, None]*r[..., None, :]+2*g1[..., None, None]*eye]
