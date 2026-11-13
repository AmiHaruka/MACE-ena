"""OpenMM callback for a complete ML molecule in a fixed-charge periodic MM bath."""

from contextlib import contextmanager
import ctypes
from functools import lru_cache

import numpy as np
import torch

from ...runtime import EnvRuntime
from ...modules.electrostatics import COULOMB
from .periodic import (
    PeriodicElectrostatics,
    box_lengths,
    neighbor_pairs,
    periodic_graph,
)
from .system import molecule_bonds, prepare_system

EV_TO_KJ_MOL = 96.48533212331002
EV_ANGSTROM_TO_KJ_MOL_NM = 10 * EV_TO_KJ_MOL


@lru_cache(maxsize=1)
def _cuda_driver():
    """Bind the installed driver; no CUDA handles are stored on the callback."""
    driver = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "cuCtxPushCurrent_v2": [ctypes.c_void_p],
        "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
        "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
    }
    for name, signature in signatures.items():
        function = getattr(driver, name)
        function.argtypes = signature
        function.restype = ctypes.c_int
    return driver


@contextmanager
def _torch_context(device):
    """Isolate Torch work from OpenMM's private CUDA context, including FFTs."""
    if device.type != "cuda":
        yield
        return
    driver = _cuda_driver()

    def checked(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            raise RuntimeError(f"ML/MM CUDA context operation {name} failed ({status})")

    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    cuda_device, primary = ctypes.c_int(), ctypes.c_void_p()
    checked("cuDeviceGet", ctypes.byref(cuda_device), ordinal)
    checked("cuDevicePrimaryCtxRetain", ctypes.byref(primary), cuda_device)
    try:
        checked("cuCtxPushCurrent_v2", primary)
        try:
            yield
        finally:
            popped = ctypes.c_void_p()
            checked("cuCtxPopCurrent_v2", ctypes.byref(popped))
    finally:
        checked("cuDevicePrimaryCtxRelease_v2", cuda_device)


def _unwrap(qm, cell, tree):
    """Integer image choices are discrete; the unfolded coordinates retain AD."""
    raw = qm.detach().cpu().numpy()
    lengths = box_lengths(cell)
    images = np.zeros_like(raw)
    images[0] = -np.floor(raw[0] / lengths)
    for parent, child in tree:
        images[child] = images[parent] - np.rint((raw[child] - raw[parent]) / lengths)
    return qm + qm.new_tensor(images) * cell.diagonal()


class PeriodicEmbedding:
    """One NN forward and one complete force derivative per callback.

    This object owns an inference model; use a separate model for concurrent
    simulations. evaluate() uses nm inputs and returns eV and eV/Angstrom.
    The callable OpenMM interface converts to kJ/mol and kJ/mol/nm.
    """

    mode = "mlmm-em"

    def __init__(self, model, system, topology, ml_indices, total_charge):
        from openmm import NonbondedForce, unit

        if getattr(model, "format_version", None) != 4:
            raise ValueError("Periodic ML/MM requires an EnvHead model")
        if self.mode == "mlmm-em" and (
            model.config.environment_mode != "once"
            or model.config.field_source != "all"
        ):
            raise ValueError(
                "Periodic ML/MM currently requires once/all environment encoding"
            )
        self.model = model.eval()
        self.runtime = EnvRuntime(model)
        self.ml_indices = np.asarray(ml_indices, dtype=np.int64)
        bonds = molecule_bonds(topology, ml_indices)
        adjacency = [[] for _ in self.ml_indices]
        for i, j in bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        self.tree, visited = [], {0}
        queue = [0]
        for parent in queue:
            for child in adjacency[parent]:
                if child not in visited:
                    visited.add(child)
                    self.tree.append((parent, child))
                    queue.append(child)
        atoms = list(topology.atoms())
        if any(atom.element is None for atom in atoms):
            raise ValueError("All ML and MM sites require chemical elements")
        all_indices = np.arange(system.getNumParticles())
        self.mm_indices = all_indices[~np.isin(all_indices, self.ml_indices)]
        self.particles = len(atoms)
        parameter = next(model.parameters())
        self.ml = torch.as_tensor(self.ml_indices, device=parameter.device)
        self.mm = torch.as_tensor(self.mm_indices, device=parameter.device)
        nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        if self.mode == "mlmm-mm":
            fixed = np.array([
                nb.getParticleParameters(int(i))[0].value_in_unit(unit.elementary_charge)
                for i in self.ml_indices
            ])
            if not np.isclose(fixed.sum(), total_charge, rtol=0, atol=1e-5):
                raise ValueError(
                    f"Force-field ML charges sum to {fixed.sum():.8g}, "
                    f"but total_charge is {total_charge}"
                )
            self.fixed_charges = torch.as_tensor(
                fixed, dtype=parameter.dtype, device=parameter.device
            )
        charges = [
            nb.getParticleParameters(int(i))[0].value_in_unit(unit.elementary_charge)
            for i in self.mm_indices
        ]
        numbers = np.array([a.element.atomic_number for a in atoms])
        self.template = self.runtime._data(
            numbers[self.ml_indices],
            np.zeros((len(self.ml), 3)),
            np.zeros((len(self.mm), 3)),
            charges,
            numbers[self.mm_indices],
            total_charge,
        )
        box = np.asarray(
            [
                v.value_in_unit(unit.nanometer)
                for v in system.getDefaultPeriodicBoxVectors()
            ]
        )
        if self.mode == "mlmm-em":
            self.pme = PeriodicElectrostatics(
                torch.as_tensor(box * 10, dtype=parameter.dtype, device=parameter.device),
                cutoff=10 * nb.getCutoffDistance().value_in_unit(unit.nanometer),
                tolerance=nb.getEwaldErrorTolerance(),
            )
        self.force_group = None

    def _vacuum(self, data):
        """Trained QM energy, retaining Q conditioning without ENV or charge heads."""
        vacuum = self.model._vacuum_data(data)
        native = {key: value for key, value in vacuum.items() if not key.startswith("mm_")}
        qm = data["positions"]
        a, b, _ = neighbor_pairs(qm, qm, None, float(self.model.r_max), same=True)
        native["edge_index"] = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
        native["atomic_numbers"] = self.model.atomic_numbers[native["node_attrs"].argmax(-1)]
        native["shifts"] = qm.new_zeros(native["edge_index"].shape[1], 3)
        native["unit_shifts"] = torch.zeros_like(native["shifts"], dtype=torch.long)
        base, _ = self.model._pass(
            native, vacuum, None, None, False, False, read_features=False
        )
        return base

    def evaluate(self, positions_nm, box_nm):
        with _torch_context(next(self.model.parameters()).device):
            return self._evaluate(positions_nm, box_nm)

    def _evaluate(self, positions_nm, box_nm):
        parameter = next(self.model.parameters())
        with torch.enable_grad():
            positions = (
                torch.as_tensor(
                    positions_nm, dtype=parameter.dtype, device=parameter.device
                ).reshape(-1, 3)
                * 10
            )
            if len(positions) != self.particles:
                raise ValueError("Position count differs from the OpenMM topology")
            positions.requires_grad_(True)
            cell = (
                torch.as_tensor(box_nm, dtype=parameter.dtype, device=parameter.device)
                * 10
            )
            qm = _unwrap(positions[self.ml], cell, self.tree)
            mm = positions[self.mm]
            # Image electrostatics are included, short-range learned chemical
            # interactions with replicas of the same ML molecule are not.
            a, b, v = neighbor_pairs(qm, qm, cell, float(self.model.r_max), same=True)
            if len(a) and torch.any((qm[a] - qm[b] - v).abs().amax(-1) > 1e-4):
                raise ValueError(
                    "ML molecule has short-range contact with its own periodic image"
                )
            # One current-frame QM--ENV search covers both real electrostatics
            # and the local encoder. No topology is cached across MD steps.
            cross_pairs = neighbor_pairs(
                qm, mm, cell, max(self.pme.cutoff, self.model.config.near_radius)
            )
            frame = self.pme.prepare(
                qm, mm, self.template["mm_charges"].flatten(), cell,
                cross_pairs=cross_pairs,
            )
            data = dict(self.template, positions=qm, mm_positions=mm)
            graph = periodic_graph(
                qm, mm, cell, float(self.model.r_max), self.model.config,
                cross_pairs=cross_pairs,
            )
            result, native = self.model._energy_forward(
                data,
                prepared_environment=(
                    graph,
                    frame.potential,
                    torch.cat((frame.potential[:, None], frame.field), dim=-1),
                ),
            )
            charges = result["charges"]
            ele = (charges.double() * frame.potential.double()).sum()
            image = frame.image_energy(charges)
            # Drop only the common, fixed reference energy before float32 E0
            # assembly. Converting an already rounded total to double is too late.
            energy = native["interaction_energy"].double().sum() + ele + image
            forces = -torch.autograd.grad(energy, positions)[0]
            return dict(
                energy=float(energy.detach()),
                energy_ele=float(ele.detach()),
                energy_image=float(image.detach()),
                forces=forces.detach().cpu().double().numpy(),
                charges=charges.detach().cpu().double().numpy(),
            )

    def __call__(self, state):
        from openmm import unit

        result = self.evaluate(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
        )
        return (
            result["energy"] * EV_TO_KJ_MOL,
            result["forces"] * EV_ANGSTROM_TO_KJ_MOL_NM,
        )


class VacuumEmbedding(PeriodicEmbedding):
    """Molecule-only MM callback; native OpenMM owns all periodic Coulomb.

    Restore full ML particle products in native exceptions and subtract the
    isolated ML Coulomb here. This retains the existing periodic image ledger
    without a second PME calculation or a learned charge evaluation.
    """

    mode = "mlmm-mm"

    def _evaluate(self, positions_nm, box_nm):
        parameter = next(self.model.parameters())
        with torch.enable_grad():
            positions = torch.as_tensor(
                positions_nm, dtype=parameter.dtype, device=parameter.device
            ).reshape(-1, 3) * 10
            if len(positions) != len(self.ml_indices):
                raise ValueError("Vacuum callback requires only the selected ML atoms")
            positions.requires_grad_(True)
            cell = torch.as_tensor(
                box_nm, dtype=parameter.dtype, device=parameter.device
            ) * 10
            qm = _unwrap(positions, cell, self.tree)
            native = self._vacuum(dict(self.template, positions=qm))
            i, j = torch.triu_indices(len(qm), len(qm), 1, device=qm.device)
            distance = (qm[i].double() - qm[j].double()).norm(dim=-1)
            isolated = (
                COULOMB * self.fixed_charges[i].double()
                * self.fixed_charges[j].double() / distance
            ).sum()
            energy = native["interaction_energy"].double().sum() - isolated
            forces = -torch.autograd.grad(energy, positions)[0]
            return dict(
                energy=float(energy.detach()),
                energy_coulomb_subtraction=-float(isolated.detach()),
                forces=forces.detach().cpu().double().numpy(),
                charges=self.fixed_charges.detach().cpu().double().numpy(),
            )


def create_mlmm_system(
    system, topology, ml_indices, model, total_charge, *, mode="mlmm-em",
    keep_ml_constraints=False,
):
    """Clone classical terms and attach one periodic full-system model callback."""
    import openmm

    if not hasattr(openmm, "PythonForce"):
        raise ImportError(
            "Periodic ML/MM needs OpenMM 8.6.1 with PythonForce; install the ML/MM optional dependencies"
        )
    prepared = prepare_system(
        system, topology, ml_indices, mode=mode,
        keep_ml_constraints=keep_ml_constraints,
    )
    callback_type = VacuumEmbedding if mode == "mlmm-mm" else PeriodicEmbedding
    callback = callback_type(model, system, topology, ml_indices, total_charge)
    force = openmm.PythonForce(callback)
    # Use the ordered particle-copy kernel. OpenMM 8.6.1's implicit whole-
    # system coordinate path can stall when CUDA PME and PythonForce coexist.
    # Every particle is still included, in the original topology order.
    force.setParticles(
        list(ml_indices) if mode == "mlmm-mm" else list(range(system.getNumParticles()))
    )
    force.setName(
        "EnvHead periodic embedding" if mode == "mlmm-em" else "EnvHead mechanical embedding native PME"
    )
    force.setUsesPeriodicBoundaryConditions(True)
    used = {f.getForceGroup() for f in prepared.getForces()}
    available = sorted(set(range(32)) - used)
    if not available:
        raise ValueError("No unused OpenMM force group for the ML/MM callback")
    callback.force_group = available[-1]
    force.setForceGroup(callback.force_group)
    prepared.addForce(force)
    return prepared, callback
