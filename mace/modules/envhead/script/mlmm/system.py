"""Prepare the classical terms for periodic electronic/mechanical ML/MM.

The native PME force retains MM electrostatics and MM Lennard-Jones terms.
A separate interaction group supplies ML--MM Lennard-Jones interactions from
the original parameters. EM places ML--MM electrostatics in the model force;
MM retains all fixed charges in native PME and subtracts isolated ML Coulomb
in the molecule-only callback attached by the caller.
"""

from numbers import Integral

import numpy as np


def _openmm():
    try:
        import openmm
    except ImportError as exc:
        raise ImportError("Periodic ML/MM system preparation requires OpenMM.") from exc
    return openmm


def _selection(topology, ml_indices):
    atoms = list(topology.atoms())
    indices = tuple(ml_indices)
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in indices
    ):
        raise TypeError("ml_indices must contain integer atom indices.")
    indices = tuple(int(i) for i in indices)
    selected = set(indices)
    if not indices or len(selected) != len(indices):
        raise ValueError(
            "ML selection must be nonempty and contain no duplicate indices."
        )
    if min(indices) < 0 or max(indices) >= len(atoms):
        raise ValueError("ML indices are outside the topology atom range.")
    if len(selected) == len(atoms):
        raise ValueError("ML selection must leave at least one MM atom.")
    if any(atom.index != index for index, atom in enumerate(atoms)):
        raise ValueError("Topology atom indices must match OpenMM particle order.")
    neighbors = [set() for _ in atoms]
    bonds = []
    for atom1, atom2 in topology.bonds():
        i, j = atom1.index, atom2.index
        neighbors[i].add(j)
        neighbors[j].add(i)
        bonds.append((i, j))
    component, pending = set(), [indices[0]]
    while pending:
        index = pending.pop()
        if index not in component:
            component.add(index)
            pending.extend(neighbors[index] - component)
    if component != selected:
        raise ValueError(
            "ML indices must select exactly one whole connected molecule in the topology."
        )
    return indices, selected, bonds


def molecule_bonds(topology, ml_indices):
    """Return topology bonds in the local order given by ``ml_indices``.

    These bonds provide the connectivity needed to unwrap the single ML
    molecule.  The selection follows the same whole-molecule restriction as
    :func:`prepare_system`.
    """
    indices, selected, bonds = _selection(topology, ml_indices)
    local = {index: position for position, index in enumerate(indices)}
    return tuple(
        (local[i], local[j]) for i, j in bonds if i in selected and j in selected
    )


def _region(particles, selected, label):
    count = sum(int(particle) in selected for particle in particles)
    if 0 < count < len(particles):
        raise ValueError(
            f"{label} crosses the ML/MM boundary; select one complete uncoupled ML molecule."
        )
    return bool(count)


def _prepare_bonded(force, selected, openmm):
    """Remove wholly ML bonded energies while retaining BB force parameters."""
    if type(force) is openmm.HarmonicBondForce:
        for index in range(force.getNumBonds()):
            i, j, length, k = force.getBondParameters(index)
            if _region((i, j), selected, "Harmonic bond"):
                force.setBondParameters(index, i, j, length, 0 * k)
    elif type(force) is openmm.HarmonicAngleForce:
        for index in range(force.getNumAngles()):
            i, j, k, angle, stiffness = force.getAngleParameters(index)
            if _region((i, j, k), selected, "Harmonic angle"):
                force.setAngleParameters(index, i, j, k, angle, 0 * stiffness)
    elif type(force) is openmm.PeriodicTorsionForce:
        for index in range(force.getNumTorsions()):
            *particles, periodicity, phase, k = force.getTorsionParameters(index)
            if _region(particles, selected, "Periodic torsion"):
                force.setTorsionParameters(index, *particles, periodicity, phase, 0 * k)
    elif type(force) is openmm.RBTorsionForce:
        for index in range(force.getNumTorsions()):
            parameters = force.getTorsionParameters(index)
            if _region(parameters[:4], selected, "Ryckaert-Bellemans torsion"):
                force.setTorsionParameters(
                    index, *parameters[:4], *(0 * k for k in parameters[4:])
                )


def _prepare_cmap(force, selected, openmm):
    # CMAP has no removeTorsion method.  Rebuild it with unchanged maps and BB
    # torsions; using an all-zero shared map would also alter BB torsions.
    replacement = openmm.CMAPTorsionForce()
    replacement.setName(force.getName())
    replacement.setForceGroup(force.getForceGroup())
    replacement.setUsesPeriodicBoundaryConditions(
        force.usesPeriodicBoundaryConditions()
    )
    for index in range(force.getNumMaps()):
        replacement.addMap(*force.getMapParameters(index))
    for index in range(force.getNumTorsions()):
        parameters = force.getTorsionParameters(index)
        if not _region(parameters[1:], selected, "CMAP torsion"):
            replacement.addTorsion(*parameters)
    return replacement


def prepare_system(
    system, topology, ml_indices, *, mode="mlmm-em", keep_ml_constraints=False
):
    """Clone an orthorhombic PME system and retain its classical ML/MM terms.

    ``ml_indices`` must select one complete connected topology molecule.
    Virtual sites, parameter offsets, covalent ML/MM boundaries, intermolecular
    exceptions and forces outside the supported native force classes are
    rejected. All particles and MM constraints are retained. Source ML
    constraints are retained when ``keep_ml_constraints`` is true; otherwise
    they are removed. Zero-energy bonds preserve molecular connectivity for
    OpenMM's molecular volume scaling.

    The returned system contains no model/Python force.  Its classical energy
    is MM internal energy plus the original ML--MM Lennard-Jones interaction.
    Native dispersion correction and custom interaction-group long-range
    correction follow OpenMM's respective finite-size conventions.
    In MM mode native PME retains all fixed particle charges. ML exception
    products are restored to full Coulomb products; the ML callback subtracts
    isolated ML Coulomb, leaving periodic images intact.
    The input system is never modified.
    """
    openmm = _openmm()
    from openmm import unit

    if mode not in ("mlmm-em", "mlmm-mm"):
        raise ValueError("mode must be mlmm-em or mlmm-mm")
    _, selected, topology_bonds = _selection(topology, ml_indices)
    count = system.getNumParticles()
    if topology.getNumAtoms() != count:
        raise ValueError(
            "Topology atom count must equal the OpenMM System particle count."
        )
    if any(system.isVirtualSite(index) for index in range(count)):
        raise ValueError(
            "Virtual sites are unsupported in the initial periodic ML/MM interface."
        )
    allowed = {
        openmm.NonbondedForce,
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
        openmm.RBTorsionForce,
        openmm.CMAPTorsionForce,
        openmm.CMMotionRemover,
    }
    for force in system.getForces():
        if type(force) not in allowed:
            raise ValueError(
                f"Unsupported force {type(force).__name__}; attach barostats after preparing the classical system."
            )
    nonbonded = [
        force for force in system.getForces() if type(force) is openmm.NonbondedForce
    ]
    if len(nonbonded) != 1:
        raise ValueError("Periodic ML/MM requires exactly one native NonbondedForce.")
    original_nb = nonbonded[0]
    if original_nb.getNonbondedMethod() != openmm.NonbondedForce.PME:
        raise ValueError("Periodic ML/MM requires a PME NonbondedForce.")
    if original_nb.getNumParticles() != count:
        raise ValueError(
            "NonbondedForce particle count must equal the OpenMM System particle count."
        )
    if (
        original_nb.getNumParticleParameterOffsets()
        or original_nb.getNumExceptionParameterOffsets()
    ):
        raise ValueError(
            "NonbondedForce particle and exception parameter offsets are unsupported."
        )
    if not original_nb.getIncludeDirectSpace():
        raise ValueError(
            "The PME NonbondedForce must include direct-space interactions."
        )
    box = np.asarray(
        [
            vector.value_in_unit(unit.nanometer)
            for vector in system.getDefaultPeriodicBoxVectors()
        ]
    )
    if (
        not np.isfinite(box).all()
        or np.any(np.diag(box) <= 0)
        or not np.allclose(box, np.diag(np.diag(box)), atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Periodic ML/MM requires a positive axis-aligned orthorhombic box."
        )
    cutoff = original_nb.getCutoffDistance().value_in_unit(unit.nanometer)
    if not 0 < cutoff <= np.min(np.diag(box)) / 2:
        raise ValueError(
            "PME cutoff must be positive and at most half the shortest box length."
        )
    for index in range(original_nb.getNumExceptions()):
        _region(
            original_nb.getExceptionParameters(index)[:2],
            selected,
            "Nonbonded exception",
        )

    prepared = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    constraint_lengths = {}
    for index in reversed(range(prepared.getNumConstraints())):
        i, j, length = prepared.getConstraintParameters(index)
        if _region((i, j), selected, "Constraint"):
            constraint_lengths[frozenset((i, j))] = length
            if not keep_ml_constraints:
                prepared.removeConstraint(index)
    replacements = []
    for index, force in enumerate(prepared.getForces()):
        if type(force) is openmm.CMAPTorsionForce:
            replacements.append((index, _prepare_cmap(force, selected, openmm)))
        else:
            _prepare_bonded(force, selected, openmm)
    # Replacing in reverse keeps every removal index valid.
    for index, replacement in reversed(replacements):
        prepared.removeForce(index)
        prepared.addForce(replacement)

    bonded = [
        force
        for force in prepared.getForces()
        if type(force) is openmm.HarmonicBondForce
    ]
    present = {
        frozenset(force.getBondParameters(index)[:2])
        for force in bonded
        for index in range(force.getNumBonds())
    }
    missing = [
        (i, j)
        for i, j in topology_bonds
        if i in selected and j in selected and frozenset((i, j)) not in present
    ]
    if missing:
        connectivity = bonded[0] if bonded else openmm.HarmonicBondForce()
        if not bonded:
            connectivity.setName("EnvHead ML molecule connectivity")
            prepared.addForce(connectivity)
        for i, j in missing:
            connectivity.addBond(
                i, j, constraint_lengths.get(frozenset((i, j)), 0 * unit.nanometer), 0.0
            )

    nb = next(
        force for force in prepared.getForces() if type(force) is openmm.NonbondedForce
    )
    cross_lj = openmm.CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)"
    )
    cross_lj.setName("EnvHead ML-MM Lennard-Jones")
    cross_lj.setForceGroup(nb.getForceGroup())
    cross_lj.addPerParticleParameter("sigma")
    cross_lj.addPerParticleParameter("epsilon")
    cross_lj.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cross_lj.setCutoffDistance(nb.getCutoffDistance())
    cross_lj.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    cross_lj.setSwitchingDistance(nb.getSwitchingDistance())
    cross_lj.setUseLongRangeCorrection(nb.getUseDispersionCorrection())
    cross_lj.addInteractionGroup(selected, set(range(count)) - selected)
    for index in range(count):
        charge, sigma, epsilon = original_nb.getParticleParameters(index)
        cross_lj.addParticle([sigma, epsilon])
        if index in selected:
            nb.setParticleParameters(
                index, charge if mode == "mlmm-mm" else 0 * charge, sigma, 0 * epsilon
            )
    # Identical exclusions across nonbonded forces permit shared GPU neighbor
    # lists.  BB exclusions remain necessary even with an AB interaction group.
    for index in range(nb.getNumExceptions()):
        i, j, charge_product, sigma, epsilon = nb.getExceptionParameters(index)
        cross_lj.addExclusion(i, j)
        if i in selected:
            product = (
                original_nb.getParticleParameters(i)[0]
                * original_nb.getParticleParameters(j)[0]
                if mode == "mlmm-mm" else 0 * charge_product
            )
            nb.setExceptionParameters(
                index, i, j, product, sigma, 0 * epsilon
            )
    prepared.addForce(cross_lj)
    return prepared
