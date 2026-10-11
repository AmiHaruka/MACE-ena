"""Nonperiodic union graph, with differentiable geometry and smooth support."""

from dataclasses import dataclass
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


def qm_edges(data, cutoff, query_radius):
    """Vacuum-only topology, without building unused ENV geometry or weights.

    Keep the union query radius and pair ordering for old-checkpoint parity.
    Native MACE computes the differentiable QQ geometry itself.
    """
    positions = data["positions"]
    coordinates = positions.detach().cpu().numpy()
    batch = data["batch"].detach().cpu().numpy()
    pairs = []
    for g in range(len(data["ptr"]) - 1):
        local = np.flatnonzero(batch == g)
        pair = cKDTree(coordinates[local]).query_pairs(
            max(cutoff, query_radius), output_type="ndarray"
        )
        pair = local[pair]
        pairs.append(pair)
    pairs = np.concatenate(pairs, axis=0)
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0).copy()
    edges = torch.as_tensor(directed.T, dtype=torch.long, device=positions.device)
    vectors = positions.detach()[edges[1]] - positions.detach()[edges[0]]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    return edges[:, lengths < cutoff]


def build_graph(data, cutoff, config):
    qm, mm = data["positions"], data["mm_positions"]
    counts = data["mm_count"].reshape(-1)
    mb = torch.repeat_interleave(torch.arange(len(counts), device=qm.device), counts)
    # Chunking bounds the QM x all-MM selection workspace.
    selected, weights = [], []
    radius = config.near_radius
    start = 0
    for g, count in enumerate(counts.tolist()):
        lo, hi = int(data["ptr"][g]), int(data["ptr"][g + 1])
        centers = qm[lo:hi]
        for first in range(start, start + count, config.mm_chunk_size):
            part = mm[first : min(first + config.mm_chunk_size, start + count)]
            # Topology needs no derivative. Only selected nodes retain the
            # differentiable coverage graph, with no cross-sample distances.
            distance = torch.cdist(part.detach(), centers.detach())
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
    pairs = []
    coordinates = pos.detach().cpu().numpy()
    batch_ids = batch.detach().cpu().numpy()
    for g in range(len(counts)):
        local = np.flatnonzero(batch_ids == g)
        pair = cKDTree(coordinates[local]).query_pairs(
            max(cutoff, radius), output_type="ndarray"
        )
        pairs.append(local[pair])
    pairs = np.concatenate(pairs, axis=0)
    both = np.concatenate((pairs, pairs[:, ::-1]), axis=0).copy()
    edges = torch.as_tensor(both.T, dtype=torch.long, device=pos.device)
    src, dst = edges
    vectors = pos[dst] - pos[src]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    cross = (src < len(qm)) != (dst < len(qm))
    edge_cutoff = torch.where(
        cross, lengths.new_tensor(radius), lengths.new_tensor(cutoff)
    )
    keep = lengths.detach() < edge_cutoff
    edges, vectors, lengths = edges[:, keep], vectors[keep], lengths[keep]
    edge_cutoff = edge_cutoff[keep]
    src, dst = edges
    nw = torch.cat((qm.new_ones(len(qm)), ew))
    weight = (
        envelope(lengths / edge_cutoff)
        * modulation(lengths / edge_cutoff, config)
        * nw[src]
        * nw[dst]
    )
    return UnionGraph(pos, batch, ids, edges, vectors, lengths, weight, len(qm))
