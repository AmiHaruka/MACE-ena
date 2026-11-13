"""ASE QM calculator with fixed external point-charge environment in atoms.info."""

from copy import deepcopy
from os import PathLike
import numpy as np
from ase.calculators.calculator import (
    Calculator,
    all_changes,
)
from .runtime import EnvRuntime


class EnvHeadCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "charges", "mm_forces"]
    input_keys = (
        "mm_positions",
        "mm_charges",
        "mm_atomic_numbers",
        "total_charge",
        "multiplicity",
    )

    def __init__(
        self,
        model,
        device="cpu",
        enable_cueq=False,
        neighbor_backend="reference",
        execution_mode=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(model, (str, PathLike)):
            from .train.checkpoint import load_model

            self.model = load_model(model, device)
        else:
            self.model = deepcopy(model).to(device).eval()
        if enable_cueq:
            from .backends import enable_cueq as convert

            convert(self.model)
        self.runtime = EnvRuntime(
            self.model, neighbor_backend=neighbor_backend, execution_mode=execution_mode
        )
        self._environment = None

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self._environment is None or any(
            not np.array_equal(atoms.info.get(k), self._environment[k])
            for k in self.input_keys
        ):
            changes.append("environment")
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.atoms.pbc.any():
            raise ValueError("EnvHead supports nonperiodic data")
        a = self.atoms
        result = self.runtime(
            a.numbers,
            a.positions,
            mm_positions=a.info.get("mm_positions", ()),
            mm_charges=a.info.get("mm_charges", ()),
            mm_atomic_numbers=a.info.get("mm_atomic_numbers", ()),
            total_charge=a.info.get("total_charge", 0),
            total_spin=a.info.get("multiplicity", 1),
            compute_charges="charges" in (properties or []),
            compute_mm_force="mm_forces" in (properties or []),
        )
        self.results = {k: v.detach().cpu().numpy() for k, v in result.items()}
        self.results["energy"] = float(self.results["energy"][0])
        self.results["free_energy"] = self.results["energy"]
        self._environment = {k: deepcopy(a.info.get(k)) for k in self.input_keys}

    def get_mm_forces(self, atoms=None):
        """Return QM-induced forces on background sites, in their input order."""
        return self.get_property("mm_forces", atoms)

    def compare_environment(self, atoms):
        """Return environment-induced energy/charge changes without updating the ASE cache."""
        if atoms.pbc.any():
            raise ValueError("EnvHead supports nonperiodic data")
        result = self.runtime.compare_environment(
            atoms.numbers,
            atoms.positions,
            mm_positions=atoms.info.get("mm_positions", ()),
            mm_charges=atoms.info.get("mm_charges", ()),
            mm_atomic_numbers=atoms.info.get("mm_atomic_numbers", ()),
            total_charge=atoms.info.get("total_charge", 0),
            total_spin=atoms.info.get("multiplicity", 1),
        )
        comparison = {
            key: value.detach().cpu().numpy().copy() for key, value in result.items()
        }
        for key in ("energy", "energy_vac", "energy_response", "energy_ele"):
            comparison[key] = float(comparison[key][0])
        return comparison
