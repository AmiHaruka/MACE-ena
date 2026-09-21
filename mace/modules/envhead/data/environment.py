"""Add environment tensors to existing MACE graphs; never add MM graph nodes."""
import torch


def attach_environment(
    graph,
    positions,
    charges,
    forces=None,
    force_weight=1.0,
    group_id=None,
    charge_group=None,
    charge_group_totals=None,
):
    kwargs = dict(dtype=graph.positions.dtype, device=graph.positions.device)
    graph.mm_positions = torch.as_tensor(positions, **kwargs).reshape(-1, 3)
    graph.mm_charges = torch.as_tensor(charges, **kwargs).reshape(-1, 1)
    if len(graph.mm_positions) != len(graph.mm_charges):
        raise ValueError("MM position/charge counts differ")
    count = len(graph.mm_positions)
    graph.mm_count = torch.tensor(
        [[count]], dtype=torch.long, device=graph.positions.device
    )
    graph.mm_forces = (
        torch.zeros_like(graph.mm_positions)
        if forces is None
        else torch.as_tensor(forces, **kwargs).reshape(count, 3)
    )
    graph.mm_forces_weight = torch.full(
        (count, 1), float(force_weight) if forces is not None else 0.0, **kwargs
    )
    graph.group_id = str(group_id) if group_id is not None else ""
    ids, totals = charge_groups(
        graph.positions, graph.total_charge, charge_group, charge_group_totals
    )
    graph.charge_group = ids
    graph.charge_group_totals = totals
    graph.charge_group_count = torch.tensor(
        [[len(totals)]], dtype=torch.long, device=graph.positions.device
    )
    return graph


def charge_groups(positions, total_charge, ids=None, totals=None):
    """Validate fixed local group IDs; no geometry-dependent group assignment.

    All atoms must belong to exactly one nonempty group. Group charges sum to Q.
    Omitting both fields reproduces the original single global constraint.
    """
    if (ids is None) != (totals is None):
        raise ValueError(
            "charge_group and charge_group_totals must be supplied together"
        )
    if ids is None:
        ids = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
        totals = torch.as_tensor(
            total_charge, device=positions.device, dtype=positions.dtype
        ).reshape(1)
    else:
        ids = torch.as_tensor(ids, device=positions.device)
        totals = torch.as_tensor(totals, device=positions.device, dtype=positions.dtype)
    if ids.shape != (len(positions),) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "charge_group must be an integer vector with one ID per ML atom"
        )
    if totals.ndim != 1 or not len(totals) or not torch.isfinite(totals).all():
        raise ValueError("charge_group_totals must be a finite nonempty vector")
    if not torch.equal(torch.unique(ids), torch.arange(len(totals), device=ids.device)):
        raise ValueError(
            "charge_group IDs must cover contiguous nonempty groups starting at zero"
        )
    q = torch.as_tensor(total_charge, device=positions.device, dtype=positions.dtype)
    if (
        q.numel() != 1
        or not torch.isfinite(q).all()
        or abs((totals.sum() - q.reshape(())).item()) > 1e-8
    ):
        raise ValueError("charge_group_totals must sum to total_charge")
    return ids.long(), totals
