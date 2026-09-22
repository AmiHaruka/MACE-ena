"""Nonperiodic union graph, with differentiable geometry and smooth support."""

from dataclasses import dataclass
import torch
import numpy as np
from scipy.spatial import cKDTree


def envelope(x):
    x = x.clamp(0, 1)
    return (1 - 10 * x**3 + 15 * x**4 - 6 * x**5).clamp_min(0)


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


def build_graph(data, cutoff, config):
    qm, mm = data["positions"], data["mm_positions"]
    counts = data["mm_count"].reshape(-1)
    mb = torch.repeat_interleave(torch.arange(len(counts), device=qm.device), counts)
    # Chunking bounds the QM x all-MM selection workspace.
    selected, weights = [], []
    radius = cutoff + config.r_env
    for start in range(0, len(mm), config.mm_chunk_size):
        part = mm[start : start + config.mm_chunk_size]
        delta = part[:, None, :] - qm[None, :, :]
        d = torch.linalg.vector_norm(delta, dim=-1)
        same = mb[start : start + len(part), None] == data["batch"][None, :]
        keep = ((d.detach() < radius) & same).any(1)
        ids = keep.nonzero().flatten()
        selected.append(ids + start)
        # Smooth union; only the final quarter of node coverage fades out.
        w = envelope((d[ids] / radius - 0.75) * 4) * same[ids]
        weights.append(1 - torch.prod(1 - w, dim=1))
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
        pair = cKDTree(coordinates[local]).query_pairs(cutoff, output_type="ndarray")
        pairs.append(local[pair])
    pairs = np.concatenate(pairs, axis=0)
    both = np.concatenate((pairs, pairs[:, ::-1]), axis=0).copy()
    edges = torch.as_tensor(both.T, dtype=torch.long, device=pos.device)
    src, dst = edges
    vectors = pos[dst] - pos[src]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    nw = torch.cat((qm.new_ones(len(qm)), ew))
    weight = (
        envelope(lengths / cutoff)
        * modulation(lengths / cutoff, config)
        * nw[src]
        * nw[dst]
    )
    return UnionGraph(pos, batch, ids, edges, vectors, lengths, weight, len(qm))
