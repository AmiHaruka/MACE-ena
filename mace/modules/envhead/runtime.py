"""Tensor runtime, independent of ASE Atoms and MD-engine APIs."""
import torch

from .data import charge_groups


class EnvRuntime:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(
        self,
        atomic_numbers,
        positions,
        mm_positions,
        mm_charges,
        total_charge=0.0,
        total_spin=1.0,
        charge_group=None,
        charge_group_totals=None,
        compute_polarizability=False,
        compute_hessian=False,
        potential_shift=0.0,
        reference_origin=None,
        mm_cell=None, rf_cutoff=None, rf_dielectric=78.4,
        **unsupported,
    ):
        if unsupported:
            raise NotImplementedError(
                f"Unsupported runtime options: {list(unsupported)}"
            )
        model = self.model
        device = model.r_max.device
        r = torch.as_tensor(
            positions, dtype=next(model.parameters()).dtype, device=device
        ).clone()
        z = torch.as_tensor(atomic_numbers, device=device)
        matches = z[:, None] == model.atomic_numbers[None, :]
        if not matches.any(1).all() or r.shape != (len(z), 3):
            raise ValueError("Invalid ML species or coordinate shape")
        # A dense nonperiodic neighbor search is adequate for this reference path.
        distances = torch.cdist(r.detach(), r.detach())
        mask = (distances < model.r_max) & ~torch.eye(
            len(r), dtype=torch.bool, device=device
        )
        edges = mask.nonzero().T
        mm = (
            torch.as_tensor(mm_positions, dtype=r.dtype, device=device)
            .reshape(-1, 3)
            .clone()
        )
        data = dict(
            positions=r,
            node_attrs=matches.to(r.dtype),
            edge_index=edges,
            shifts=r.new_zeros(edges.shape[1], 3),
            unit_shifts=r.new_zeros(edges.shape[1], 3),
            cell=r.new_zeros(3, 3),
            pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
            batch=torch.zeros(len(r), dtype=torch.long, device=device),
            ptr=torch.tensor([0, len(r)], device=device),
            head=torch.zeros(1, dtype=torch.long, device=device),
            total_charge=r.new_tensor([total_charge]),
            total_spin=r.new_tensor([total_spin]),
            mm_positions=mm,
            mm_charges=torch.as_tensor(
                mm_charges, dtype=r.dtype, device=device
            ).reshape(-1, 1),
            mm_count=torch.tensor([[len(mm)]], device=device),
        )
        ids, totals = charge_groups(r, total_charge, charge_group, charge_group_totals)
        data.update(
            charge_group=ids,
            charge_group_totals=totals,
            charge_group_count=torch.tensor([[len(totals)]], device=device),
        )
        data["potential_shift"] = r.new_tensor([potential_shift])
        if reference_origin is not None:
            data["reference_origin"] = torch.as_tensor(
                reference_origin, dtype=r.dtype, device=device
            ).reshape(1, 3)
        if (mm_cell is None) != (rf_cutoff is None):
            raise ValueError("Supply both mm_cell and rf_cutoff")
        if mm_cell is not None:
            data["mm_cell"] = torch.as_tensor(mm_cell, dtype=r.dtype, device=device).reshape(1, 3, 3)
            data["rf_cutoff"] = rf_cutoff
            data["rf_dielectric"] = rf_dielectric
        return model(data, compute_polarizability=compute_polarizability,
                     compute_hessian=compute_hessian)
