"""Periodic classical RF and molecular unwrapping for ASE-driven ML/MM."""
import numpy as np
from ..modules.periodic import rf_coefficients


def add_rf(system, cutoff, dielectric, charges):
    """Replace ordinary MM electrostatics by switched AMP-style RF.

    LJ and explicit exceptions remain in NonbondedForce. Exception pairs keep
    their scaled Coulomb charges there (only particle charges are zeroed) and
    are excluded from this force, so scaled 1-2/1-3/1-4 electrostatics survive
    the switch from NoCutoff. Periodic LJ tail corrections are disabled in
    both subtractive systems.
    """
    import openmm as mm

    k, a, b, c = rf_coefficients(cutoff, dielectric)
    expression = (f'138.93545764438198*q1*q2*(1/r+({k})*r^2+({a})*r^4+({b})*r^6-({c}))*s;'
                  's=(1-x)^5*(1+5*x+15*x^2+35*x^3+70*x^4);'
                  f'x=min(1,max(0,(r^2-{(.8*cutoff)**2})/{cutoff**2-(.8*cutoff)**2}))')
    force = mm.CustomNonbondedForce(expression)
    force.addPerParticleParameter('q')
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)
    for q in charges:
        force.addParticle([q])
    for nb in system.getForces():
        if isinstance(nb, mm.NonbondedForce):
            for i in range(nb.getNumParticles()):
                q, sigma, epsilon = nb.getParticleParameters(i)
                nb.setParticleParameters(i, q*0, sigma, epsilon)
            for i in range(nb.getNumExceptions()):
                left, right, *_ = nb.getExceptionParameters(i)
                force.addExclusion(left, right)
    system.addForce(force)


def molecules(system):
    import openmm as mm

    adjacency = [[] for _ in range(system.getNumParticles())]
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a, b, *_ = force.getBondParameters(i)
                adjacency[a].append(b)
                adjacency[b].append(a)
    return adjacency


def unwrap(positions, lengths, adjacency):
    result = positions.copy()
    seen = set()
    components = []
    for root in range(len(positions)):
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        for a in queue:
            for b in adjacency[a]:
                delta = positions[b]-positions[a]
                candidate = result[a]+delta-lengths*np.round(delta/lengths)
                if b not in seen:
                    result[b] = candidate
                    queue.append(b)
                    seen.add(b)
                elif not np.allclose(candidate, result[b], atol=1e-6, rtol=0):
                    raise ValueError('Molecule winds around the periodic box')
        components.append(set(queue))
    return result, components


def validate_cell(cell, cutoff):
    cell = np.asarray(cell)
    if (cell.shape != (3, 3) or not np.isfinite(cell).all()
            or not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-10, rtol=0)):
        raise ValueError('Require an axis-aligned orthorhombic cell')
    if np.any(np.diag(cell) <= 2*cutoff):
        raise ValueError('RF cutoff must be strictly below half every box length')
    return np.diag(cell).copy()
