"""Native MACE file readers plus environment fields and lightweight split metadata."""
from pathlib import Path
import logging
from time import perf_counter

import h5py
from torch.utils.data import ConcatDataset

from mace.data.hdf5_dataset import HDF5Dataset, unpack_value
from mace.data.utils import config_from_atoms, load_from_xyz
from mace.tools import AtomicNumberTable

from .atomic_data import EnvAtomicData, key_specification


class GraphList(list):
    """A list of native graphs with small sample/group ID lists for splitting."""


def _graphs(configs, model):
    table = AtomicNumberTable(model.atomic_numbers.tolist())
    heads = getattr(model, "heads", ["Default"])
    graphs = GraphList()
    started = perf_counter()
    logging.info("Building QM graphs and attaching MM arrays for %d configurations", len(configs))
    for i, c in enumerate(configs, 1):
        graphs.append(EnvAtomicData.from_config(
            c,
            table,
            float(model.r_max),
            heads=heads,
            dtype=next(model.parameters()).dtype,
        ))
        if i % 1000 == 0:
            logging.info("Prepared graphs: %d/%d (%.1f s)", i, len(configs), perf_counter() - started)
    logging.info("Graph preparation complete: %d configurations in %.1f s", len(graphs), perf_counter() - started)
    graphs.group_ids = [g.group_id for g in graphs]
    graphs.sample_ids = [g.sample_id for g in graphs]
    return graphs


def graphs_from_atoms(frames, model, energy_key="REF_energy", forces_key="REF_forces"):
    keys = key_specification(energy_key, forces_key)
    head = getattr(model, "heads", ["Default"])[0]
    return _graphs([config_from_atoms(a, keys, head_name=head) for a in frames], model)


def load_dataset(
    path,
    model,
    energy_key="REF_energy",
    forces_key="REF_forces",
    total_charge_key="total_charge",
    dipole_key="REF_dipole",
):
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("Supply at least one dataset file")
        datasets = [
            load_dataset(
                item, model, energy_key, forces_key, total_charge_key, dipole_key
            )
            for item in path
        ]
        combined = ConcatDataset(datasets)
        combined.group_ids = [group for d in datasets for group in d.group_ids]
        combined.sample_ids = [sample for d in datasets for sample in d.sample_ids]
        return combined
    heads = getattr(model, "heads", ["Default"])
    if Path(path).suffix not in (".h5", ".hdf5"):
        started = perf_counter()
        logging.info("Parsing extxyz and all QM/MM arrays: %s", path)
        _, configs = load_from_xyz(
            str(path),
            key_specification(energy_key, forces_key, total_charge_key, dipole_key),
            head_name=heads[0],
            keep_isolated_atoms=True,
        )
        logging.info("Parsed %d configurations in %.1f s", len(configs), perf_counter() - started)
        return _graphs(configs, model)
    # Native HDF5 uses canonical property names; external key mappings apply to XYZ.
    # Read only IDs here, not all QM/MM arrays. Let the native reader do the work.
    group_ids, sample_ids = [], []
    with h5py.File(path, "r") as file:
        if "config_batch_0" not in file:
            raise ValueError(
                "Expected native MACE HDF5 with config_batch_0"
            )
        size = len(file["config_batch_0"])
        if not size or set(file) != {f"config_batch_{i}" for i in range(len(file))}:
            raise ValueError("Invalid native MACE HDF5 batch layout")
        for i in range(len(file)):
            group = file[f"config_batch_{i}"]
            if set(group) != {f"config_{j}" for j in range(size)}:
                raise ValueError(
                    "Native HDF5 requires equal-sized configuration batches"
                )
            for j in range(size):
                props = group[f"config_{j}/properties"]
                for name, values in (
                    ("group_id", group_ids),
                    ("sample_id", sample_ids),
                ):
                    values.append(
                        str(unpack_value(props[name][()]) or "")
                        if name in props
                        else ""
                    )
    dataset = HDF5Dataset(
        str(path),
        float(model.r_max),
        AtomicNumberTable(model.atomic_numbers.tolist()),
        atomic_dataclass=EnvAtomicData,
        heads=heads,
        dtype=next(model.parameters()).dtype,
    )
    # Close the constructor's handle so workers open their own native reader handle.
    dataset._file.close()
    dataset._file = None
    dataset.group_ids, dataset.sample_ids = group_ids, sample_ids
    return dataset
