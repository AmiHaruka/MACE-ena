"""Differentiable periodic geometry and point-charge PME (eV, Angstrom, e).

The mesh shape and Ewald splitting are fixed at construction. Each frame owns
its interpolation weights and coordinate graph, including rejected NPT moves.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
from scipy.fft import next_fast_len
from scipy.spatial import cKDTree
import torch

from ...modules.electrostatics import COULOMB
from ...modules.graph import EdgeView, OnceGraph, envelope, modulation, near_weight


def box_lengths(cell):
    box = cell.detach().cpu().numpy()
    lengths = np.diag(box).copy()
    if (
        not np.isfinite(box).all()
        or np.any(lengths <= 0)
        or not np.allclose(box, np.diag(lengths), atol=1e-7, rtol=0)
    ):
        raise ValueError("ML/MM requires a positive orthorhombic periodic box")
    return lengths


def minimum_image(vectors, cell):
    lengths = cell.diagonal()
    return vectors - torch.round((vectors / lengths).detach()) * lengths


def neighbor_pairs(query, sources, cell, cutoff, *, same=False):
    """CUDA cross searches stay on device; same-set searches use a sparse tree.

    Membership uses double precision, as in cKDTree. Edge vectors are rebuilt
    from the original tensors so all coordinate and cell derivatives survive.
    In particular, large ENV--ENV sets never allocate a dense distance matrix.
    """
    if query.is_cuda and not same:
        if cell is not None:
            box_lengths(cell)
        vectors = query.detach().double()[:, None] - sources.detach().double()[None]
        if cell is not None:
            vectors = minimum_image(vectors, cell.detach().double())
        i, j = (vectors.square().sum(-1) <= cutoff**2).nonzero(as_tuple=True)
        vectors = query[i] - sources[j]
        if cell is not None:
            vectors = minimum_image(vectors, cell)
        return i, j, vectors
    q = query.detach().cpu().numpy()
    b = sources.detach().cpu().numpy()
    lengths = None if cell is None else box_lengths(cell)
    if lengths is not None:
        # A tiny negative float32 coordinate can wrap to exactly L. The tree
        # requires [0,L), so choose discrete images in double precision.
        lengths = lengths.astype(np.float64)
        q, b = q.astype(np.float64) % lengths, b.astype(np.float64) % lengths
        q, b = np.where(q < lengths, q, 0.0), np.where(b < lengths, b, 0.0)
    if same:
        pairs = cKDTree(q, boxsize=lengths).query_pairs(cutoff, output_type="ndarray")
    elif not len(q) or not len(b):
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        neighbors = cKDTree(b, boxsize=lengths).query_ball_point(q, cutoff)
        pairs = np.asarray(
            [(i, j) for i, js in enumerate(neighbors) for j in js], dtype=np.int64
        ).reshape(-1, 2)
    indices = torch.as_tensor(pairs.T.copy(), dtype=torch.long, device=query.device)
    vectors = query[indices[0]] - sources[indices[1]]
    if cell is not None:
        vectors = minimum_image(vectors, cell)
    return indices[0], indices[1], vectors


def _cross_pairs(qm, mm, cell, cutoff, pairs):
    """Filter a same-frame superset without rounding membership to model dtype."""
    if pairs is None:
        return neighbor_pairs(qm, mm, cell, cutoff)
    i, j, vectors = pairs
    distance = minimum_image(
        qm.detach().double()[i] - mm.detach().double()[j], cell.detach().double()
    )
    keep = distance.square().sum(-1) <= cutoff**2
    return i[keep], j[keep], vectors[keep]


def periodic_graph(qm, mm, cell, cutoff, config, *, cross_pairs=None):
    """The existing once encoder consumes periodic EQ and EE edge geometry."""
    if min(box_lengths(cell)) <= 2 * max(cutoff, config.near_radius):
        raise ValueError("Box must exceed twice the model's local cutoffs")
    qi, mj, eq_vectors = _cross_pairs(qm, mm, cell, config.near_radius, cross_pairs)
    ids, inverse = torch.unique(mj, sorted=True, return_inverse=True)
    selected = mm[ids]
    dr = minimum_image(selected[:, None] - qm[None], cell)
    support = near_weight(torch.linalg.vector_norm(dr, dim=-1), config)
    coverage = 1 - (1 - support).prod(-1)
    a, b, ee_vectors = neighbor_pairs(selected, selected, cell, cutoff, same=True)
    ee = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
    ee_lengths = torch.cat((ee_vectors.norm(dim=-1), ee_vectors.norm(dim=-1)))
    eq_lengths = eq_vectors.norm(dim=-1)
    context = EdgeView(
        ee[0],
        ee[1],
        ee_lengths,
        envelope(ee_lengths / cutoff)
        * modulation(ee_lengths / cutoff, config)
        * coverage[ee[0]]
        * coverage[ee[1]],
    )
    density = EdgeView(
        inverse,
        qi,
        eq_lengths,
        envelope(eq_lengths / config.near_radius)
        * modulation(eq_lengths / config.near_radius, config)
        * coverage[inverse],
        eq_vectors,
    )
    # Molecular QQ is deliberately unwrapped and nonperiodic.
    a, b, _ = neighbor_pairs(qm, qm, None, cutoff, same=True)
    qq = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
    eq = torch.stack((inverse + len(qm), qi))
    edges = torch.cat((qq, ee + len(qm), eq, eq.flip(0)), dim=1)
    return OnceGraph(len(qm), ids, edges, context, density)


@torch.jit.script
def _p3m_weights(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    ns: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    zs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fifth-order cardinal B-splines, matching torch-pme's P3M stencil."""
    relative = ns * torch.matmul(positions, inverse_cell)
    indices = torch.round(relative).long()
    x = relative - indices
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    weights = torch.stack(
        [
            1 / 384 * (1 - 8 * x + 24 * x2 - 32 * x3 + 16 * x4),
            1 / 96 * (19 - 44 * x + 24 * x2 + 16 * x3 - 16 * x4),
            1 / 192 * (115 - 120 * x2 + 48 * x4),
            1 / 96 * (19 + 44 * x + 24 * x2 - 16 * x3 - 16 * x4),
            1 / 384 * (1 + 8 * x + 24 * x2 + 32 * x3 + 16 * x4),
        ]
    )
    grid = torch.stack(
        [
            (indices - 2) % ns,
            (indices - 1) % ns,
            indices % ns,
            (indices + 1) % ns,
            (indices + 2) % ns,
        ]
    )
    flat = (grid[xs, :, 0] * ns[1] + grid[ys, :, 1]) * ns[2] + grid[zs, :, 2]
    combined = weights[xs, :, 0] * weights[ys, :, 1] * weights[zs, :, 2]
    return flat, combined


@torch.jit.script
def _p3m_spread(
    charges: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    shape: Tuple[int, int, int],
) -> torch.Tensor:
    channels = charges.shape[1]
    values = (charges.T[:, None, :] * weights[None]).reshape(channels, -1)
    index = indices.reshape(1, -1).expand(channels, -1)
    mesh = torch.zeros(
        (channels, shape[0] * shape[1] * shape[2]),
        dtype=weights.dtype, device=weights.device,
    )
    return mesh.scatter_add(1, index, values).reshape(
        channels, shape[0], shape[1], shape[2]
    )


@torch.jit.script
def _p3m_gather(
    potential: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return (potential.flatten(1)[:, indices] * weights).sum(dim=1).T


class _P3MInterpolation:
    """One frame's differentiable stencil, shared by spread and gather."""

    def __init__(self, positions, inverse_cell, ns, shape, stencil):
        self.shape = shape
        self.indices, self.interpolation_weights = _p3m_weights(
            positions, inverse_cell, ns, *stencil
        )

    def points_to_mesh(self, charges):
        return _p3m_spread(
            charges, self.indices, self.interpolation_weights, self.shape
        )

    def mesh_to_points(self, potential):
        return _p3m_gather(potential, self.indices, self.interpolation_weights)


@dataclass
class PeriodicFrame:
    potential: torch.Tensor
    field: torch.Tensor
    qm: torch.Tensor
    cell: torch.Tensor
    query: object
    convolution: object
    kernel: object
    cutoff: float

    def image_energy(self, charges):
        """C_periodic(q) - C_isolated(q), retaining dq/dR throughout."""
        q = charges.reshape(-1, 1)
        mesh = self.query.points_to_mesh(q)
        phi = self.query.mesh_to_points(self.convolution(mesh)).flatten()
        volume = self.cell.det().abs()
        reciprocal = 0.5 * (q.flatten().double() * phi.double()).sum() / volume.double()
        self_energy = (
            -0.5 * self.kernel.self_contribution() * charges.double().square().sum()
        )
        plasma = (
            -self.kernel.background_correction()
            * charges.double().sum().square()
            / volume
        )
        i, j, vectors = neighbor_pairs(
            self.qm, self.qm, self.cell, self.cutoff, same=True
        )
        distance = vectors.norm(dim=-1)
        real = (
            charges[i].double()
            * charges[j].double()
            * self.kernel.sr_from_dist(distance).double()
        ).sum()
        i, j = torch.triu_indices(len(charges), len(charges), 1, device=charges.device)
        direct = (
            COULOMB
            * charges[i].double()
            * charges[j].double()
            / (self.qm[i] - self.qm[j]).double().norm(dim=-1)
        ).sum()
        return reciprocal + self_energy + plasma + real - direct


class PeriodicElectrostatics:
    """Source-to-query PME using torch-pme's public differentiable blocks.

    Returned potential is the full physical potential, without a factor 1/2.
    The conductor/zero-mean Green function includes the neutralizing plasma.
    """

    def __init__(self, cell, cutoff=10.0, tolerance=1e-4, mesh_spacing=None):
        if not 0 < tolerance < 0.1 or cutoff <= 0:
            raise ValueError("PME requires positive cutoff and tolerance in (0, 0.1)")
        lengths = box_lengths(cell)
        if min(lengths) <= 2 * cutoff:
            raise ValueError("PME cutoff must be below half the shortest box length")
        if mesh_spacing is not None and mesh_spacing <= 0:
            raise ValueError("mesh_spacing must be positive")
        self.cutoff = cutoff
        self.alpha = math.sqrt(-math.log(2 * tolerance)) / cutoff
        ns = (
            2 * self.alpha * lengths / (3 * tolerance**0.2)
            if mesh_spacing is None
            else lengths / mesh_spacing
        )
        self.mesh = tuple(next_fast_len(max(6, math.ceil(n))) for n in ns)
        self._fixed = None

    def _mesh_objects(self, qm, cell, lengths):
        """Cache only immutable fixed-cell data, never an interpolation graph.

        Replace rather than update objects when a volume move changes the box:
        a retained frame must still be able to use its original convolution.
        Differentiable cells always get a fresh graph, including on repeat calls.
        """
        from torchpme import CoulombPotential
        from torchpme.lib import P3MKSpaceFilter

        key = (tuple(lengths), qm.device, qm.dtype, self.mesh, self.alpha)
        if (
            not cell.requires_grad
            and self._fixed is not None
            and self._fixed[0] == key
        ):
            return self._fixed[1:]
        fixed_cell = cell if cell.requires_grad else cell.detach().clone()
        kernel = CoulombPotential(
            smearing=1 / (math.sqrt(2) * self.alpha), prefactor=COULOMB
        ).to(qm)
        ns = torch.tensor(self.mesh, dtype=torch.long, device=qm.device)
        convolution = P3MKSpaceFilter(
            cell=fixed_cell,
            ns_mesh=ns,
            kernel=kernel,
            interpolation_nodes=5,
            mode=0,
            fft_norm="backward",
            ifft_norm="forward",
        )
        shifts = torch.arange(5, device=qm.device)
        stencil = tuple(
            axis.flatten()
            for axis in torch.meshgrid(shifts, shifts, shifts, indexing="ij")
        )
        if not cell.requires_grad:
            self._fixed = key, kernel, ns, convolution, stencil
        return kernel, ns, convolution, stencil

    def prepare(self, qm, mm, charges, cell, *, cross_pairs=None):
        """Prepare one frame; optional cross_pairs must cover the real cutoff."""
        lengths = box_lengths(cell)
        if min(lengths) <= 2 * self.cutoff:
            raise ValueError("Current box is too small for the fixed PME cutoff")
        kernel, ns, convolution, stencil = self._mesh_objects(qm, cell, lengths)
        # Fifth-order cardinal B-splines (called P3M by torch-pme) are C3
        # across stencil changes. Lagrange stencils have small jumps there,
        # which are unsuitable for field-conditioned conservative MD forces.
        inverse = torch.linalg.inv_ex(cell)[0] if cell.is_cuda else torch.linalg.inv(cell)
        query = _P3MInterpolation(qm, inverse, ns, self.mesh, stencil)
        source = _P3MInterpolation(mm, inverse, ns, self.mesh, stencil)
        rho = source.points_to_mesh(charges.reshape(-1, 1))
        phi = query.mesh_to_points(convolution(rho)).flatten() / cell.det().abs()
        phi = (
            phi - 2 * kernel.background_correction() * charges.sum() / cell.det().abs()
        )
        i, j, vectors = _cross_pairs(qm, mm, cell, self.cutoff, cross_pairs)
        contribution = charges[j] * kernel.sr_from_dist(vectors.norm(dim=-1))
        phi = phi.index_add(0, i, contribution)
        field = -torch.autograd.grad(
            phi.sum(), qm, create_graph=True, retain_graph=True
        )[0]
        return PeriodicFrame(
            phi, field, qm, cell, query, convolution, kernel, self.cutoff
        )
