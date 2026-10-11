"""ASE QM calculator with fixed external point-charge environment in atoms.info."""

from copy import deepcopy
import numpy as np
from ase.calculators.calculator import (
    Calculator,
    all_changes,
)
from .runtime import EnvRuntime


class EnvHeadCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "charges"]
    input_keys = (
        "mm_positions",
        "mm_charges",
        "mm_atomic_numbers",
        "total_charge",
        "multiplicity",
    )

    def __init__(self, model, device="cpu", enable_cueq=False, **kwargs):
        super().__init__(**kwargs)
        self.model = deepcopy(model).to(device).eval()
        if enable_cueq:
            from .backends import enable_cueq as convert

            convert(self.model)
        self.runtime = EnvRuntime(self.model)
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
        )
        self.results = {k: v.detach().cpu().numpy() for k, v in result.items()}
        self.results["energy"] = float(self.results["energy"][0])
        self.results["free_energy"] = self.results["energy"]
        self._environment = {k: deepcopy(a.info.get(k)) for k in self.input_keys}

    def compare_environment_charges(self, atoms):
        """Compare embedded and zero-source QM charges without changing ASE cache."""
        if atoms.pbc.any():
            raise ValueError("EnvHead supports nonperiodic data")
        mm_positions = atoms.info.get("mm_positions", ())
        charges = np.asarray(atoms.info.get("mm_charges", ()))
        comparison = {}
        for level, source in ((1, charges), (0, np.zeros_like(charges))):
            result = self.runtime(
                atoms.numbers,
                atoms.positions,
                mm_positions=mm_positions,
                mm_charges=source,
                mm_atomic_numbers=atoms.info.get("mm_atomic_numbers", ()),
                total_charge=atoms.info.get("total_charge", 0),
                total_spin=atoms.info.get("multiplicity", 1),
                compute_force=False,
                compute_charges=True,
            )
            comparison[f"energy_lambda{level}"] = float(result["energy"][0].detach())
            for key in ("charges",):
                comparison[f"{key}_lambda{level}"] = (
                    result[key].detach().cpu().numpy().copy()
                )
        comparison["delta_charges"] = (
            comparison["charges_lambda1"] - comparison["charges_lambda0"]
        )
        return comparison

    def compare_environment(self, atoms):
        """Return v4 energy/charge changes without updating the normal ASE cache."""
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
