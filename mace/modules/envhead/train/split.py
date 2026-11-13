"""Group-aware train/validation indices, independent of file parsing."""

from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Subset


class IndexedDataset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, list(indices))
        self.group_ids = [dataset.group_ids[i] for i in self.indices]
        self.sample_ids = [dataset.sample_ids[i] for i in self.indices]


def split_dataset(dataset, valid_fraction=0.1, seed=0):
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must lie between zero and one")
    if not any(dataset.group_ids):
        if len(dataset) < 2:
            raise ValueError(
                "At least two samples required; supply an explicit validation file"
            )
        order = np.arange(len(dataset))
        np.random.default_rng(seed).shuffle(order)
        ntrain = min(
            len(dataset) - int(len(dataset) * valid_fraction), len(dataset) - 1
        )
        return IndexedDataset(dataset, order[:ntrain].tolist()), IndexedDataset(
            dataset, order[ntrain:].tolist()
        )
    if not all(dataset.group_ids):
        raise ValueError("Partial group_id labels: supply all groups or none")
    names = sorted(set(dataset.group_ids))
    if len(names) < 2:
        raise ValueError(
            "At least two groups required; supply an explicit validation file"
        )
    order = torch.randperm(
        len(names), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    nvalid = max(1, min(len(names) - 1, round(len(names) * valid_fraction)))
    valid = {names[i] for i in order[:nvalid]}
    return (
        IndexedDataset(
            dataset, [i for i, g in enumerate(dataset.group_ids) if g not in valid]
        ),
        IndexedDataset(
            dataset, [i for i, g in enumerate(dataset.group_ids) if g in valid]
        ),
    )


def save_validation_indices(dataset, path):
    """Write original, zero-based frame indices even for a tiny validation set."""
    Path(path).write_text("".join(f"{index}\n" for index in dataset.indices))


def group_split(graphs, valid_fraction=0.1, seed=0):
    """Split whole geometry groups, retaining vacuum/PC pairs together."""
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must lie between zero and one")
    groups = {}
    for graph in graphs:
        key = getattr(graph, "group_id", None)
        if not key:
            raise ValueError(
                "Group splitting requires nonempty group_id on every graph"
            )
        groups.setdefault(key, []).append(graph)
    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    names = sorted(groups)
    order = torch.randperm(
        len(names), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    nvalid = max(1, min(len(names) - 1, round(len(names) * valid_fraction)))
    valid_names = {names[i] for i in order[:nvalid]}
    return (
        [g for name in names if name not in valid_names for g in groups[name]],
        [g for name in names if name in valid_names for g in groups[name]],
    )
