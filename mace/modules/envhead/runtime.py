"""Tensor inference for nonperiodic EnvHead models (eV, Angstrom, charge)."""

import torch


class EnvRuntime:
    def __init__(self, model, neighbor_backend="reference", execution_mode=None):
        self.model = model
        self.neighbor_backend = neighbor_backend
        allowed = ("reference", "optimized")
        if execution_mode is not None and execution_mode not in allowed:
            raise ValueError(f"execution_mode must be one of {allowed}")
        self.execution_mode = execution_mode or "optimized"
        if getattr(model, "_cueq_enabled", False):
            from .backends import enable_cueq, prepare_cueq_device

            if next(model.parameters()).is_cuda:
                enable_cueq(model)
            else:
                prepare_cueq_device(model)

    def __call__(
        self,
        atomic_numbers,
        positions,
        mm_positions=(),
        mm_charges=(),
        mm_atomic_numbers=(),
        total_charge=0.0,
        total_spin=1.0,
        compute_force=True,
        compute_response=False,
        compute_charges=False,
        compute_vacuum_charges=False,
        *,
        compute_mm_force=False,
    ):
        data = self._data(
            atomic_numbers,
            positions,
            mm_positions,
            mm_charges,
            mm_atomic_numbers,
            total_charge,
            total_spin,
        )
        return self.model(
            data,
            compute_force=compute_force,
            compute_response=compute_response,
            compute_charges=compute_charges,
            compute_vacuum_charges=compute_vacuum_charges,
            compute_mm_force=compute_mm_force,
            neighbor_backend=self.neighbor_backend,
            execution_mode=self.execution_mode,
        )

    def _data(
        self,
        atomic_numbers,
        positions,
        mm_positions=(),
        mm_charges=(),
        mm_atomic_numbers=(),
        total_charge=0.0,
        total_spin=1.0,
    ):
        p = next(self.model.parameters())
        r = torch.as_tensor(positions, dtype=p.dtype, device=p.device).reshape(-1, 3)
        z = torch.as_tensor(atomic_numbers, device=p.device)
        attrs = z[:, None] == self.model.atomic_numbers[None, :]
        if not attrs.any(1).all():
            raise ValueError("QM element outside foundation element table")
        mm = torch.as_tensor(mm_positions, dtype=p.dtype, device=p.device).reshape(
            -1, 3
        )
        return dict(
            positions=r,
            node_attrs=attrs.to(p.dtype),
            batch=torch.zeros(len(r), dtype=torch.long, device=p.device),
            ptr=torch.tensor([0, len(r)], device=p.device),
            cell=r.new_zeros(3, 3),
            pbc=torch.zeros(1, 3, dtype=torch.bool, device=p.device),
            head=torch.zeros(1, dtype=torch.long, device=p.device),
            total_charge=r.new_tensor([total_charge]),
            total_spin=r.new_tensor([total_spin]),
            mm_positions=mm,
            mm_charges=torch.as_tensor(
                mm_charges, dtype=p.dtype, device=p.device
            ).reshape(-1, 1),
            mm_atomic_numbers=torch.as_tensor(
                mm_atomic_numbers, dtype=torch.long, device=p.device
            ),
            mm_count=torch.tensor([[len(mm)]], device=p.device),
        )

    def compare_environment(
        self,
        atomic_numbers,
        positions,
        mm_positions=(),
        mm_charges=(),
        mm_atomic_numbers=(),
        total_charge=0.0,
        total_spin=1.0,
        compute_charges=True,
    ):
        data = self._data(
            atomic_numbers,
            positions,
            mm_positions,
            mm_charges,
            mm_atomic_numbers,
            total_charge,
            total_spin,
        )
        return self.model.compare_environment(
            data, compute_charges=compute_charges,
            neighbor_backend=self.neighbor_backend,
            execution_mode=self.execution_mode,
        )
