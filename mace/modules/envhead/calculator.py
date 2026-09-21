"""ASE interface for nonperiodic EnvMACE with fixed external point charges."""
from copy import deepcopy

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .runtime import EnvRuntime
from .train.checkpoint import load_head


class EnvHeadCalculator(Calculator):
    """QM atoms belong in Atoms; external sources belong in atoms.info.

    Units: eV, Angstrom, elementary charge. MM forces contain only the model's
    embedding contribution, not MM internal forces or cross Lennard-Jones terms.
    Supply an assembled model OR a foundation model file and head-only file.
    """

    implemented_properties = [
        "energy", "free_energy", "forces", "charges", "mm_forces",
        "atomic_dipoles", "atomic_quadrupoles", "density_dipole",
        "base_energy", "delta_energy", "learned_energy", "external_energy",
    ]
    input_keys = (
        "mm_positions", "mm_charges", "total_charge", "multiplicity",
        "charge_group", "charge_group_totals", "mm_cell", "rf_cutoff", "rf_dielectric",
    )

    def __init__(
        self, model=None, *, foundation=None, head=None, device="cpu",
        enable_cueq=False, **kwargs,
    ):
        super().__init__(**kwargs)
        if model is None:
            if foundation is None or head is None:
                raise ValueError("Supply model, or both foundation and head paths")
            base = torch.load(foundation, map_location="cpu", weights_only=False)
            model = load_head(head, base)
        elif foundation is not None or head is not None:
            raise ValueError("model and foundation/head are mutually exclusive")
        # Calculator ownership prevents changing the caller's training model.
        self.model = deepcopy(model).to(device).eval()
        self.model._envhead_enable_cueq = False
        if enable_cueq:
            from .backends import enable_cueq as configure_cueq

            configure_cueq(self.model)
        self.runtime = EnvRuntime(self.model)
        self._environment = None

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        current = {key: atoms.info.get(key) for key in self.input_keys}
        if self._environment is None or any(
            not np.array_equal(current[key], self._environment[key])
            for key in self.input_keys
        ):
            changes.append("environment")
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        atoms = self.atoms
        if atoms.pbc.any():
            raise NotImplementedError("EnvHeadCalculator supports nonperiodic systems only")
        info = atoms.info
        pc_r, pc_q = info.get("mm_positions"), info.get("mm_charges")
        if (pc_r is None) != (pc_q is None):
            raise ValueError("Both mm_positions and mm_charges must be supplied")
        if pc_r is None:
            pc_r, pc_q = np.empty((0, 3)), np.empty((0, 1))
        pc_r, pc_q = np.asarray(pc_r, dtype=float), np.asarray(pc_q, dtype=float)
        if pc_r.size == 0:
            pc_r = pc_r.reshape(0, 3)
        if pc_r.ndim != 2 or pc_r.shape[1] != 3:
            raise ValueError("mm_positions must have shape [M,3]")
        if pc_q.shape not in ((len(pc_r),), (len(pc_r), 1)):
            raise ValueError("mm_charges must have shape [M] or [M,1]")
        output = self.runtime(
            atoms.numbers, atoms.positions, pc_r, pc_q,
            total_charge=info.get("total_charge", 0.0),
            total_spin=info.get("multiplicity", 1.0),
            charge_group=info.get("charge_group"),
            charge_group_totals=info.get("charge_group_totals"),
            mm_cell=info.get("mm_cell"), rf_cutoff=info.get("rf_cutoff"),
            rf_dielectric=info.get("rf_dielectric", 78.4),
        )
        scalars = {"energy", "base_energy", "delta_energy", "learned_energy", "external_energy"}
        for key in self.implemented_properties:
            value = output.get(key)
            if value is not None:
                value = value.detach().cpu().numpy()
                self.results[key] = float(value.reshape(-1)[0]) if key in scalars else value.copy()
        self.results["free_energy"] = self.results["energy"]
        self._environment = deepcopy({key: info.get(key) for key in self.input_keys})

    def get_mm_forces(self, atoms=None):
        """Return forces in the same source order as mm_positions (eV/Angstrom)."""
        return self.get_property("mm_forces", atoms)

    def get_hessian(self, atoms=None, *, method="analytic", step=None):
        """QM energy Hessian with fixed MM sources, in eV/Angstrom**2.

        Returns [3*N, N, 3], matching the single-model MACE calculator.
        Central differences of unconstrained forces require 6*N evaluations.
        Analytic mode differentiates the complete energy twice.
        The model dtype is preserved in both modes.
        """
        if method not in ("analytic", "numerical"):
            raise ValueError("method must be analytic or numerical")
        if method == "analytic" and step is not None:
            raise ValueError("step only applies to numerical Hessians")
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms not set")
        if method == "analytic":
            if atoms.pbc.any():
                raise NotImplementedError("EnvHeadCalculator supports nonperiodic systems only")
            info = atoms.info
            pc_r, pc_q = info.get("mm_positions"), info.get("mm_charges")
            if (pc_r is None) != (pc_q is None):
                raise ValueError("Both mm_positions and mm_charges must be supplied")
            if pc_r is None:
                pc_r, pc_q = np.empty((0, 3)), np.empty((0, 1))
            output = self.runtime(
                atoms.numbers, atoms.positions, pc_r, pc_q,
                total_charge=info.get("total_charge", 0.0),
                total_spin=info.get("multiplicity", 1.0),
                charge_group=info.get("charge_group"),
                charge_group_totals=info.get("charge_group_totals"),
                mm_cell=info.get("mm_cell"), rf_cutoff=info.get("rf_cutoff"),
                rf_dielectric=info.get("rf_dielectric", 78.4),
                compute_hessian=True,
            )
            return output["hessian"].detach().cpu().numpy().copy()
        if step is None:
            step = 1e-4 if next(self.model.parameters()).dtype == torch.float64 else 1e-3
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be a positive finite displacement in Angstrom")
        probe = atoms.copy()
        probe.info = deepcopy(atoms.info)
        positions = probe.positions.copy()
        # Preserve the original ASE cache, including on evaluation failure.
        saved = self.atoms, self.results, self._environment
        rows = []
        try:
            for atom in range(len(probe)):
                for axis in range(3):
                    probe.positions[:] = positions
                    probe.positions[atom, axis] += step
                    plus = self.get_forces(probe)
                    probe.positions[atom, axis] -= 2 * step
                    minus = self.get_forces(probe)
                    rows.append(-(plus - minus) / (2 * step))
        finally:
            self.atoms, self.results, self._environment = saved
        return np.asarray(rows).reshape(3 * len(probe), len(probe), 3)
