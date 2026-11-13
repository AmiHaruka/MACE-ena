"""Nonperiodic union graph, with differentiable geometry and smooth support."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary
import torch
import numpy as np
from scipy.spatial import cKDTree


def envelope(x):
    x = x.clamp(0, 1)
    return (1 - 10 * x**3 + 15 * x**4 - 6 * x**5).clamp_min(0)


def near_weight(distance, config):
    """C2 source support, used for node coverage and pairwise near/far partition."""
    return envelope(
        (distance / config.near_radius - config.switch_fraction)
        / (1 - config.switch_fraction)
    )


def modulation(x, config):
    if config.radial == "constant":
        f = torch.ones_like(x)
    elif config.radial == "cosine":
        f = (1 + torch.cos(torch.pi * x)) / 2
    elif config.radial == "polynomial":
        f = envelope(x)
    elif config.radial == "mollifier":
        f = torch.exp(1 - 1 / (1 - x.square()).clamp_min(1e-8))
    elif config.radial == "switch":
        f = envelope((x - 0.5) * 2)
    else:
        f = torch.exp(-config.radial_shape * x.square())
    return config.radial_baseline + config.radial_amplitude * f


@dataclass
class UnionGraph:
    positions: torch.Tensor
    batch: torch.Tensor
    env_indices: torch.Tensor
    edge_index: torch.Tensor
    vectors: torch.Tensor
    lengths: torch.Tensor
    edge_weight: torch.Tensor
    qm_count: int


@dataclass(frozen=True)
class EdgeView:
    """Role-local geometry, owned by the current forward and its autograd graph."""

    src: torch.Tensor
    dst: torch.Tensor
    lengths: torch.Tensor
    weight: torch.Tensor
    vectors: torch.Tensor | None = None


@dataclass(frozen=True)
class OnceGraph:
    """Topology and only the continuous geometry consumed by once encoding."""

    qm_count: int
    env_indices: torch.Tensor
    edge_index: torch.Tensor
    context: EdgeView
    density: EdgeView


def _once_graph(data, cutoff, config, ids, coverage, positions, edges):
    n = len(data["positions"])
    src, dst = edges
    # Neighbor membership has no coordinate derivative. Preserve the original
    # edge ordering and exact role-specific cutoffs before building geometry.
    p = positions.detach()
    lengths = torch.linalg.vector_norm(p[dst] - p[src], dim=-1)
    cross = (src < n) != (dst < n)
    rc = torch.where(
        cross, lengths.new_tensor(config.near_radius), lengths.new_tensor(cutoff)
    )
    edges = edges[:, lengths < rc]
    src, dst = edges
    ee = ((src >= n) & (dst >= n)).nonzero().flatten()
    eq = ((src >= n) & (dst < n)).nonzero().flatten()
    a, b = src[ee] - n, dst[ee] - n
    c, d = src[eq] - n, dst[eq]
    mm = data["mm_positions"].index_select(0, ids)
    ee_lengths = torch.linalg.vector_norm(mm[b] - mm[a], dim=-1)
    eq_vectors = data["positions"][d] - mm[c]
    eq_lengths = torch.linalg.vector_norm(eq_vectors, dim=-1)
    ee_x, eq_x = ee_lengths / cutoff, eq_lengths / config.near_radius
    context = EdgeView(
        a,
        b,
        ee_lengths,
        envelope(ee_x) * modulation(ee_x, config) * coverage[a] * coverage[b],
    )
    density = EdgeView(
        c,
        d,
        eq_lengths,
        envelope(eq_x) * modulation(eq_x, config) * coverage[c],
        eq_vectors,
    )
    return OnceGraph(n, ids, edges, context, density)


def _assemble_graph(
    data, cutoff, config, ids, coverage, positions, batch, edges, *, role_geometry
):
    """Apply the same differentiable geometry to either neighbor topology."""
    if role_geometry:
        return _once_graph(data, cutoff, config, ids, coverage, positions, edges)
    n = len(data["positions"])
    src, dst = edges
    vectors = positions[dst] - positions[src]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    cross = (src < n) != (dst < n)
    edge_cutoff = torch.where(
        cross, lengths.new_tensor(config.near_radius), lengths.new_tensor(cutoff)
    )
    keep = lengths.detach() < edge_cutoff
    edges, vectors, lengths = edges[:, keep], vectors[keep], lengths[keep]
    edge_cutoff = edge_cutoff[keep]
    src, dst = edges
    node_weight = torch.cat((data["positions"].new_ones(n), coverage))
    weight = (
        envelope(lengths / edge_cutoff)
        * modulation(lengths / edge_cutoff, config)
        * node_weight[src]
        * node_weight[dst]
    )
    return UnionGraph(positions, batch, ids, edges, vectors, lengths, weight, n)


def prepare_once_edges(graph, mm_positions):
    """Select each edge role once; EE radii depend only on the actual MM sites.

    Coverage weights still come from the union graph: their QM dependence is
    physical and must survive force training and environment-coordinate AD.
    """
    if isinstance(graph, OnceGraph):
        return graph.context, graph.density
    n = graph.qm_count
    src, dst = graph.edge_index
    ee = ((src >= n) & (dst >= n)).nonzero().flatten()
    eq = ((src >= n) & (dst < n)).nonzero().flatten()
    a, b = src.index_select(0, ee) - n, dst.index_select(0, ee) - n
    mm = mm_positions.index_select(0, graph.env_indices)
    context = EdgeView(
        a,
        b,
        torch.linalg.vector_norm(mm.index_select(0, b) - mm.index_select(0, a), dim=-1),
        graph.edge_weight.index_select(0, ee),
    )
    density = EdgeView(
        src.index_select(0, eq) - n,
        dst.index_select(0, eq),
        graph.lengths.index_select(0, eq),
        graph.edge_weight.index_select(0, eq),
        graph.vectors.index_select(0, eq),
    )
    return context, density


def prepare_reference_edges(graph):
    """Reference edge views built with boolean selection."""
    n = graph.qm_count
    src, dst = graph.edge_index
    ee = (src >= n) & (dst >= n)
    eq = (src >= n) & (dst < n)
    return (
        EdgeView(src[ee] - n, dst[ee] - n, graph.lengths[ee], graph.edge_weight[ee]),
        EdgeView(
            src[eq] - n,
            dst[eq],
            graph.lengths[eq],
            graph.edge_weight[eq],
            graph.vectors[eq],
        ),
    )


def _reference_edges(positions, batch, ngraphs, radius):
    coordinates = positions.detach().cpu().numpy()
    batch_ids = batch.detach().cpu().numpy()
    pairs = []
    for g in range(ngraphs):
        local = np.flatnonzero(batch_ids == g)
        pair = cKDTree(coordinates[local]).query_pairs(radius, output_type="ndarray")
        pairs.append(local[pair])
    pairs = np.concatenate(pairs, axis=0)
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0).copy()
    return torch.as_tensor(directed.T, dtype=torch.long, device=positions.device)


def qm_edges(data, cutoff, query_radius, *, neighbor_backend="reference", owner=None):
    """Vacuum-only topology, without building unused ENV geometry or weights.

    Keep the union query radius and pair ordering for old-checkpoint parity.
    Native MACE computes the differentiable QQ geometry itself.
    """
    positions = data["positions"]
    if resolve_neighbor_backend(neighbor_backend, positions.device) == "pyg":
        edges = _pyg_builder(owner).edges(
            positions,
            data["batch"],
            len(data["ptr"]) - 1,
            max(cutoff, query_radius),
        )
    else:
        edges = _reference_edges(
            positions, data["batch"], len(data["ptr"]) - 1, max(cutoff, query_radius)
        )
    vectors = positions.detach()[edges[1]] - positions.detach()[edges[0]]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    return edges[:, lengths < cutoff]


def build_graph(
    data,
    cutoff,
    config,
    *,
    neighbor_backend="reference",
    owner=None,
    role_geometry=False,
    unchunked=False,
):
    if resolve_neighbor_backend(neighbor_backend, data["positions"].device) == "pyg":
        return _pyg_builder(owner)(
            data, cutoff, config, role_geometry=role_geometry, unchunked=unchunked
        )
    return _reference_graph(
        data, cutoff, config, role_geometry=role_geometry, unchunked=unchunked
    )


def _reference_graph(data, cutoff, config, *, role_geometry=False, unchunked=False):
    qm, mm = data["positions"], data["mm_positions"]
    counts = data["mm_count"].reshape(-1)
    mb = torch.repeat_interleave(torch.arange(len(counts), device=qm.device), counts)
    # Whole-structure selection by default; explicit chunks bound workspace.
    selected, weights = [], []
    radius = config.near_radius
    ptr = data["ptr"].tolist()
    start = 0
    for g, count in enumerate(counts.tolist()):
        lo, hi = ptr[g], ptr[g + 1]
        centers = qm[lo:hi]
        step = (0 if unchunked else config.mm_chunk_size) or max(count, 1)
        for first in range(start, start + count, step):
            part = mm[first : min(first + step, start + count)]
            # Topology needs no derivative. Only selected nodes retain the
            # differentiable coverage graph, with no cross-sample distances.
            # Keep full-selection distances independent of cdist's >25-point
            # matrix-multiplication heuristic, which cancels at large origins.
            distance = torch.cdist(
                part.detach(),
                centers.detach(),
                compute_mode=(
                    "donot_use_mm_for_euclid_dist"
                    if unchunked
                    else "use_mm_for_euclid_dist_if_necessary"
                ),
            )
            local = (distance < radius).any(1).nonzero().flatten()
            selected.append(local + first)
            d = torch.linalg.vector_norm(part[local, None] - centers[None], dim=-1)
            w = near_weight(d, config)
            weights.append(1 - torch.prod(1 - w, dim=1))
        start += count
    ids = (
        torch.cat(selected)
        if selected
        else torch.empty(0, dtype=torch.long, device=qm.device)
    )
    ew = torch.cat(weights) if weights else qm.new_empty(0)
    pos = torch.cat((qm, mm[ids]))
    batch = torch.cat((data["batch"], mb[ids]))
    # One geometric neighbor query. The native QM view is a slice of this graph.
    edges = _reference_edges(pos, batch, len(counts), max(cutoff, radius))
    return _assemble_graph(
        data, cutoff, config, ids, ew, pos, batch, edges, role_geometry=role_geometry
    )


def resolve_neighbor_backend(backend, device):
    """Runtime choice, independent of weights and of the ENV execution mode."""
    if backend not in ("auto", "reference", "pyg"):
        raise ValueError("neighbor_backend must be auto, reference or pyg")
    if backend == "auto":
        return "pyg" if torch.device(device).type == "cuda" else "reference"
    if backend == "pyg" and torch.device(device).type != "cuda":
        raise ValueError(
            "neighbor_backend=pyg requires CUDA; use auto/reference on CPU"
        )
    return backend


# Values contain only an integer capacity and imported operators. They neither
# own the model nor survive its lifetime/save/load, and never retain geometry.
_pyg_builders = WeakKeyDictionary()


def _pyg_builder(owner):
    if owner is None:
        return _PyGNeighborBuilder()
    builder = _pyg_builders.get(owner)
    if builder is None:
        builder = _PyGNeighborBuilder()
        _pyg_builders[owner] = builder
    return builder


class _PyGNeighborBuilder:
    """Detached integer queries; all continuous geometry uses original tensors."""

    def __init__(self):
        try:
            from torch_cluster import radius, radius_graph
        except (ImportError, OSError) as error:
            raise ImportError(
                "The PyG neighbor backend needs a torch-cluster wheel matching "
                "your Torch/CUDA versions. See https://github.com/rusty1s/pytorch_cluster#installation "
                "or select neighbor_backend='reference'."
            ) from error
        self.radius, self.radius_graph = radius, radius_graph
        self.capacity = 32

    def select(self, qm, mm, qb, mb, ngraphs, config, *, unchunked=False):
        selected = []
        step = (0 if unchunked else config.mm_chunk_size) or max(len(mm), 1)
        for first in range(0, len(mm), step):
            # One witness is sufficient for membership, never for message edges.
            pair = self.radius(
                qm.detach(),
                mm[first : first + step].detach(),
                config.near_radius,
                batch_x=qb,
                batch_y=mb[first : first + step],
                batch_size=ngraphs,
                max_num_neighbors=1,
            )
            selected.append(pair[0].sort().values + first)
        return torch.cat(selected) if selected else qb.new_empty(0)

    @staticmethod
    def coverage(qm, mm, ids, mb, ptr, config, *, unchunked=False):
        if not len(ids):
            return qm.new_empty(0)
        sizes = ptr[1:] - ptr[:-1]
        offset = torch.arange(int(sizes.max()), device=qm.device)
        weights = []
        step = (0 if unchunked else config.mm_chunk_size) or len(ids)
        for selected in ids.split(step):
            batch = mb[selected]
            valid = offset[None] < sizes[batch, None]
            index = ptr[batch, None] + offset[None]
            index = torch.minimum(index, ptr[batch + 1, None] - 1)
            dr = mm[selected, None] - qm[index]
            # Mask before norm: padded zero vectors would poison double backward.
            dr = torch.where(valid[..., None], dr, torch.ones_like(dr))
            distance = torch.linalg.vector_norm(dr, dim=-1)
            weight = torch.where(valid, near_weight(distance, config), 0.0)
            weights.append(1 - torch.prod(1 - weight, dim=1))
        return torch.cat(weights)

    def edges(self, positions, batch, ngraphs, cutoff):
        if not len(positions):
            return batch.new_empty((2, 0))
        # Model order is [all QM | all ENV], so sort only the query view.
        order = torch.argsort(batch, stable=True)
        coordinates, groups = positions.detach()[order], batch[order]
        capacity = min(self.capacity, len(positions))
        while True:
            edges = self.radius_graph(
                coordinates,
                cutoff,
                batch=groups,
                batch_size=ngraphs,
                loop=False,
                max_num_neighbors=capacity,
            )
            degree = torch.bincount(edges[1], minlength=len(positions))
            maximum = int(degree.max())
            if maximum < capacity:
                # Adapt down too, so a previous dense structure does not dictate
                # allocations for all later small/sparse inputs.
                self.capacity = max(32, 1 << maximum.bit_length())
                edges = order[edges]
                key = edges[0] * len(positions) + edges[1]
                return edges[:, key.argsort(stable=True)]
            capacity *= 2

    def __call__(self, data, cutoff, config, *, role_geometry=False, unchunked=False):
        qm, mm = data["positions"], data["mm_positions"]
        counts = data["mm_count"].reshape(-1)
        mb = torch.repeat_interleave(
            torch.arange(len(counts), device=qm.device), counts, output_size=len(mm)
        )
        ids = self.select(
            qm, mm, data["batch"], mb, len(counts), config, unchunked=unchunked
        )
        coverage = self.coverage(
            qm, mm, ids, mb, data["ptr"], config, unchunked=unchunked
        )
        positions = torch.cat((qm, mm[ids]))
        batch = torch.cat((data["batch"], mb[ids]))
        edges = self.edges(
            positions, batch, len(counts), max(cutoff, config.near_radius)
        )
        return _assemble_graph(
            data, cutoff, config, ids, coverage, positions, batch, edges,
            role_geometry=role_geometry,
        )
