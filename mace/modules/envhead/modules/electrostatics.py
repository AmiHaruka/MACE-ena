"""Fixed-source potentials: far descriptors and the complete cross interaction."""

import torch
from .encoders import COULOMB
from .graph import near_weight


def potentials(data, config):
    r = data["positions"]
    full, far = [], []
    start = 0
    for g, count in enumerate(data["mm_count"].reshape(-1).tolist()):
        lo, hi = int(data["ptr"][g]), int(data["ptr"][g + 1])
        phi = r.new_zeros(hi - lo)
        feat = r.new_zeros(hi - lo, 4)
        for j in range(start, start + count, config.mm_chunk_size):
            stop = min(j + config.mm_chunk_size, start + count)
            dr = r[lo:hi, None] - data["mm_positions"][None, j:stop]
            d = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-8)
            b = data["mm_charges"][j:stop].flatten()[None]
            phi = phi + (COULOMB * b / d).sum(1)
            far_b = b * (1 - near_weight(d, config))
            feat = feat + COULOMB * torch.cat(
                (
                    (far_b / d).sum(1, keepdim=True),
                    (far_b[..., None] * dr / d[..., None] ** 3).sum(1),
                ),
                -1,
            )
        full.append(phi)
        far.append(feat)
        start += count
    # Retain the well-defined empty-source derivative.
    return torch.cat(full) + data["mm_charges"].sum() * 0, torch.cat(far)
