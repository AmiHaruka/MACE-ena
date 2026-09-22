"""ASE QM calculator with fixed external point-charge environment in atoms.info."""

from copy import deepcopy
import numpy as np
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
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
            raise ValueError("EnvHead v2 supports nonperiodic data")
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
        if "charges" in (properties or []) and not self.results["charges_valid"].all():
            self.results.pop("charges", None)
            raise PropertyNotImplementedError(
                "Insufficient independent environment samples for charge reconstruction"
            )
