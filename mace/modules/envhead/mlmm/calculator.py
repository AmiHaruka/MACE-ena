"""Subtractive EE-ML/MM, following ref/frag-method-on-ASE's calculator layout.

E = MM_full(q_QM=0) - MM_QM(q_QM=0) + EnvMACE.
Both classical systems retain the original particle indexing and parameters.
The subtraction system has all terms involving MM particles disabled.
"""
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import kJ, mol, nm


def classical_systems(system, qm_indices, rf_cutoff=None, rf_dielectric=78.4,
                      keep_qm_charges=False):
    """Create independent full/subtraction systems without reparameterization.

    keep_qm_charges=False (EE embedding): QM charges are zeroed everywhere and
    the ML potential supplies all QM-MM electrostatics.
    keep_qm_charges=True (fixed-charge baseline): QM particles keep their
    force-field charges toward MM particles (standard fixed-charge QM/MM
    electrostatics); all QM-QM pairs additionally get zero-charge exceptions
    so the classical intramolecular Coulomb cannot double-count the ML energy.
    """
    import openmm as mm
    from openmm import unit

    qm = set(qm_indices)
    if system.usesPeriodicBoundaryConditions() and rf_cutoff is None:
        raise NotImplementedError("Periodic ML/MM is unsupported")
    if system.getNumConstraints() or any(
        system.isVirtualSite(i) for i in range(system.getNumParticles())
    ):
        raise NotImplementedError("Use an unconstrained system without virtual sites")
    full = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    internal = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    charges = None
    bonded = {
        mm.HarmonicBondForce: ('Bond', 2),
        mm.HarmonicAngleForce: ('Angle', 3),
        mm.PeriodicTorsionForce: ('Torsion', 4),
        mm.RBTorsionForce: ('Torsion', 4),
    }
    for original, outer, inner in zip(system.getForces(), full.getForces(), internal.getForces()):
        kind = type(original)
        if kind is mm.CMMotionRemover:
            continue  # Contexts are evaluated only; ASE performs integration.
        if kind is mm.NonbondedForce:
            if charges is not None:
                raise NotImplementedError("Exactly one NonbondedForce is supported")
            if rf_cutoff is None and original.getNonbondedMethod() != mm.NonbondedForce.NoCutoff:
                raise NotImplementedError("First ML/MM implementation requires NoCutoff")
            if rf_cutoff is not None:
                # ReactionFieldDielectric is inert here: add_rf later zeroes
                # every particle charge, leaving only the switched LJ below.
                for target in (outer, inner):
                    target.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
                    target.setCutoffDistance(rf_cutoff / nm)
                    target.setReactionFieldDielectric(rf_dielectric)
                    target.setUseDispersionCorrection(False)
                    target.setUseSwitchingFunction(True)
                    target.setSwitchingDistance(.8*rf_cutoff / nm)
            if original.getNumParticleParameterOffsets() or original.getNumExceptionParameterOffsets():
                raise NotImplementedError("Nonbonded parameter offsets are unsupported")
            charges = []
            qm_list = sorted(qm)
            existing = set()
            for i in range(original.getNumExceptions()):
                a, b, *_ = original.getExceptionParameters(i)
                existing.add((min(a, b), max(a, b)))
            for i in range(original.getNumParticles()):
                q, sigma, epsilon = original.getParticleParameters(i)
                charges.append(q.value_in_unit(unit.elementary_charge))
                if keep_qm_charges:
                    outer.setParticleParameters(i, q, sigma, epsilon)
                else:
                    outer.setParticleParameters(i, q * 0 if i in qm else q, sigma, epsilon)
                inner.setParticleParameters(i, q * 0, sigma, epsilon if i in qm else epsilon * 0)
            if keep_qm_charges:
                params = {i: outer.getParticleParameters(i) for i in qm_list}
                for a in range(len(qm_list)):
                    for b in range(a + 1, len(qm_list)):
                        i, j = qm_list[a], qm_list[b]
                        if (i, j) in existing:
                            continue  # already Coulomb-zeroed in the exception loop
                        sigma = 0.5 * (params[i][1] + params[j][1])
                        epsilon = (params[i][2] * params[j][2]) ** 0.5
                        outer.addException(i, j, 0.0, sigma, epsilon)
            for i in range(original.getNumExceptions()):
                a, b, qprod, sigma, epsilon = original.getExceptionParameters(i)
                if (a in qm) != (b in qm):
                    raise NotImplementedError("Cross-region nonbonded exceptions require a boundary scheme")
                outer.setExceptionParameters(i, a, b, qprod * 0 if a in qm else qprod, sigma, epsilon)
                inner.setExceptionParameters(i, a, b, qprod * 0, sigma,
                                             epsilon if a in qm else epsilon * 0)
        elif kind in bonded:
            term, n_atoms = bonded[kind]
            for i in range(getattr(original, 'getNum' + term + 's')()):
                values = list(getattr(original, 'get' + term + 'Parameters')(i))
                membership = [a in qm for a in values[:n_atoms]]
                if any(membership) and not all(membership):
                    raise NotImplementedError("Covalent ML/MM boundaries are unsupported")
                if not any(membership):
                    if kind is mm.RBTorsionForce:
                        values[4:] = [v * 0 for v in values[4:]]
                    else:
                        values[-1] *= 0
                    getattr(inner, 'set' + term + 'Parameters')(i, *values)
        else:
            raise NotImplementedError(f"Unsupported classical force: {kind.__name__}")
    if charges is None:
        raise ValueError("System must contain a NonbondedForce")
    if rf_cutoff is not None:
        from .periodic import add_rf
        add_rf(full, rf_cutoff / nm, rf_dielectric,
               [0 if i in qm else q for i, q in enumerate(charges)])
    return full, internal, np.asarray(charges)


class EnvQMMMCalculator(Calculator):
    """ASE integration, OpenMM MM energies, and an EnvHeadCalculator for QM.

    Inputs: ASE atoms and corresponding OpenMM System in identical atom order,
    explicit zero-based QM indices, and an existing EnvHeadCalculator. Charges
    come from the original System. No external MM list or reparameterized QM
    topology is needed. MM coordinates move, while their charges stay fixed.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms, system, qm_indices, env_calculator, total_charge=0.0,
                 platform='CPU', platform_properties=None, rf_cutoff=None,
                 rf_dielectric=78.4, keep_qm_charges=False, **kwargs):
        import openmm as mm

        super().__init__(**kwargs)
        indices = np.asarray(qm_indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or not len(indices) or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(atoms))):
            raise ValueError("qm_indices must contain unique valid integer indices")
        self.rf_cutoff, self.rf_dielectric = rf_cutoff, rf_dielectric
        self._cell = None
        if atoms.pbc.any():
            from .periodic import validate_cell, molecules, unwrap
            from ..modules.periodic import rf_coefficients
            if not atoms.pbc.all() or rf_cutoff is None:
                raise ValueError('Periodic ML/MM requires 3D PBC and explicit rf_cutoff in Angstrom')
            rf_coefficients(rf_cutoff, rf_dielectric)
            self._cell = np.asarray(atoms.cell).copy()
            lengths = validate_cell(self._cell, rf_cutoff)
            self._adjacency = molecules(system)
            _, components = unwrap(atoms.positions, lengths, self._adjacency)
            if set(indices) not in components:
                raise NotImplementedError('Periodic ML region must be one complete bonded molecule')
        elif rf_cutoff is not None:
            raise ValueError('rf_cutoff requires periodic atoms')
        if system.getNumParticles() != len(atoms):
            raise ValueError("System particles and ASE atoms must have identical ordering and count")
        if not np.isfinite(total_charge):
            raise ValueError("total_charge must be finite")
        self.qm_indices = indices.copy()
        self.mm_indices = np.setdiff1d(np.arange(len(atoms)), indices)
        self.numbers = atoms.numbers.copy()
        self.env_calculator = env_calculator
        self.total_charge = float(total_charge)
        full, internal, charges = classical_systems(system, indices.tolist(),
                                                    rf_cutoff, rf_dielectric,
                                                    keep_qm_charges=keep_qm_charges)
        if self._cell is not None:
            for target in (full, internal):
                target.setDefaultPeriodicBoxVectors(*(self._cell / nm))
        self.mm_charges = charges[self.mm_indices].copy()
        engine = mm.Platform.getPlatformByName(platform)
        self._integrators = [mm.VerletIntegrator(0.001) for _ in range(2)]
        self._contexts = [mm.Context(s, integrator, engine, platform_properties or {})
                          for s, integrator in zip((full, internal), self._integrators)]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        from openmm import unit

        super().calculate(atoms, properties, system_changes)
        self.results = {}
        positions = self.atoms.positions
        if self._cell is not None:
            from .periodic import unwrap
            if not self.atoms.pbc.all() or not np.allclose(self.atoms.cell, self._cell, atol=1e-10, rtol=0):
                raise NotImplementedError('Periodic ML/MM currently requires a fixed cell')
            positions, _ = unwrap(positions, np.diag(self._cell), self._adjacency)
            if np.any(np.ptp(positions[self.qm_indices], axis=0) >= np.diag(self._cell)/2):
                raise ValueError('ML molecule must fit within half the box along each axis')
        elif self.atoms.pbc.any():
            raise NotImplementedError('Rebuild calculator to enable periodic ML/MM')
        if not np.array_equal(self.atoms.numbers, self.numbers):
            raise ValueError("Atom identity/order changed; rebuild the calculator")
        energy, forces = [], []
        for context in self._contexts:
            context.setPositions(positions / nm)
            state = context.getState(getEnergy=True, getForces=True)
            energy.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * kJ / mol)
            forces.append(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer) * kJ / mol / nm)
        qm_atoms = Atoms(numbers=self.numbers[self.qm_indices],
                         positions=positions[self.qm_indices])
        qm_atoms.info.update(total_charge=self.total_charge,
                             mm_positions=positions[self.mm_indices].copy(),
                             mm_charges=self.mm_charges.copy())
        if self._cell is not None:
            qm_atoms.info.update(mm_cell=self._cell.copy(), rf_cutoff=self.rf_cutoff,
                                 rf_dielectric=self.rf_dielectric)
        # No centering or removal of the QM net force: embedding transfers momentum.
        self.env_calculator.calculate(qm_atoms, ['energy', 'forces'], all_changes)
        env = self.env_calculator.results
        combined = forces[0] - forces[1]
        combined[self.qm_indices] += env['forces']
        combined[self.mm_indices] += env['mm_forces']
        total = energy[0] - energy[1] + env['energy']
        if not np.isfinite(total) or not np.isfinite(combined).all():
            raise FloatingPointError("Nonfinite ML/MM energy or forces")
        self.results = dict(energy=float(total), forces=combined)

    def close(self):
        """Release OpenMM contexts when the calculator is no longer needed."""
        self._contexts.clear()
        self._integrators.clear()
