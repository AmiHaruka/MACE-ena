"""One environment normalization point for native HDF5 and XYZ readers."""

from dataclasses import replace

import numpy as np
import torch

from mace.data import AtomicData
from mace.data.utils import KeySpecification

from .environment import attach_environment


def key_specification(
    energy_key="REF_energy",
    forces_key="REF_forces",
    total_charge_key="total_charge",
    dipole_key="REF_dipole",
):
    keys = KeySpecification.from_defaults()
    extra = (
        "mm_positions",
        "mm_charges",
        "mm_atomic_numbers",
        "mm_count",
        "mm_forces",
        "mm_forces_weight",
        "charge_group_totals",
        "multiplicity",
        "sample_id",
        "pair_id",
        "group_id",
    )
    keys.update(
        info_keys={
            **{k: k for k in extra},
            "energy": energy_key,
            "total_charge": total_charge_key,
            "dipole": dipole_key,
        },
        arrays_keys={"forces": forces_key, "charge_group": "charge_group"},
    )
    return keys


class EnvAtomicData(AtomicData):
    @classmethod
    def from_config(cls, config, z_table, cutoff, **kwargs):
        props = config.properties
        if props.get("charges") is None:
            config = replace(
                config, property_weights={**config.property_weights, "charges": 0.0}
            )
        if config.pbc is not None and np.asarray(config.pbc).any():
            raise NotImplementedError("Periodic EnvHead training is unsupported")
        if props.get("multiplicity") is not None and int(props["multiplicity"]) != 1:
            raise NotImplementedError(
                "Current EnvHead training supports singlet records only"
            )
        pc_r, pc_q = props.get("mm_positions"), props.get("mm_charges")
        if (pc_r is None) != (pc_q is None):
            raise ValueError("Both mm_positions and mm_charges must be supplied")
        if pc_r is None:
            pc_r, pc_q = np.empty((0, 3)), np.empty((0, 1))
        pc_r, pc_q = np.asarray(pc_r), np.asarray(pc_q)
        # JSON decoders can turn an empty matrix into a 1-D empty list.
        if not pc_r.size:
            pc_r = pc_r.reshape(0, 3)
        if not pc_q.size:
            pc_q = pc_q.reshape(0, 1)
        if pc_r.shape != (len(pc_r), 3) or pc_q.shape not in (
            (len(pc_r),),
            (len(pc_r), 1),
        ):
            raise ValueError(
                "MM coordinates/charges must have shapes [M,3] and [M] or [M,1]"
            )
        if not np.isfinite(pc_r).all() or not np.isfinite(pc_q).all():
            raise ValueError("Nonfinite MM sources")
        count = props.get("mm_count")
        if count is not None and (
            np.asarray(count).size != 1 or np.asarray(count).item() != len(pc_r)
        ):
            raise ValueError("mm_count does not match point sources")
        # A single-head foundation owns the head label, including legacy HDF5.
        heads = kwargs.get("heads", ["Default"])
        dtype = kwargs.pop("dtype", torch.get_default_dtype())
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            graph = super().from_config(
                replace(config, head=heads[0]), z_table, cutoff, **kwargs
            )
        finally:
            torch.set_default_dtype(previous)
        force_weight = props.get("mm_forces_weight")
        if force_weight is None:
            force_weight = config.property_weights.get("mm_forces", 1.0)
        attach_environment(
            graph,
            pc_r,
            pc_q,
            props.get("mm_forces"),
            force_weight,
            group_id=props.get("group_id"),
            charge_group=props.get("charge_group"),
            charge_group_totals=props.get("charge_group_totals"),
        )
        for key in ("sample_id", "pair_id"):
            setattr(graph, key, str(props.get(key) or ""))
        if props.get("mm_atomic_numbers") is not None:
            graph.mm_atomic_numbers = torch.as_tensor(
                props["mm_atomic_numbers"], dtype=torch.long
            )
        elif not len(pc_r):
            graph.mm_atomic_numbers = torch.empty(0, dtype=torch.long)
        return graph
