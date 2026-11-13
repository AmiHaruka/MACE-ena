"""Full cross interaction and configurable far/all-source field descriptors."""

from functools import lru_cache
from importlib.util import find_spec
import torch
from .graph import near_weight

COULOMB = 14.3996454784255  # eV Angstrom / e^2


def potentials(data, config):
    """Per-structure electrostatic descriptors with optional source chunking."""
    r = data["positions"]
    full, far = [], []
    all_sources = getattr(config, "field_source", "far") == "all"
    ptr = data["ptr"].tolist()
    start = 0
    for g, count in enumerate(data["mm_count"].reshape(-1).tolist()):
        lo, hi = ptr[g], ptr[g + 1]
        phi = r.new_zeros(hi - lo)
        descriptor_phi = r.new_zeros(hi - lo)
        field = r.new_zeros(hi - lo, 3)
        step = config.mm_chunk_size or max(count, 1)
        for j in range(start, start + count, step):
            stop = min(j + step, start + count)
            dr = r[lo:hi, None] - data["mm_positions"][None, j:stop]
            d = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-8)
            b = data["mm_charges"][j:stop].flatten()[None]
            phi = phi + (
                COULOMB * (b / d).sum(1) if all_sources else (COULOMB * b / d).sum(1)
            )
            descriptor_b = b if all_sources else b * (1 - near_weight(d, config))
            if not all_sources:
                descriptor_phi = descriptor_phi + COULOMB * (descriptor_b / d).sum(1)
            field = field + COULOMB * (
                descriptor_b[..., None] * dr / d[..., None] ** 3
            ).sum(1)
        full.append(phi)
        far.append(
            torch.cat(((phi if all_sources else descriptor_phi)[:, None], field), -1)
        )
        start += count
    # Retain the well-defined empty-source derivative.
    return torch.cat(full) + data["mm_charges"].sum() * 0, torch.cat(far)


@lru_cache(maxsize=1)
def _has_triton():
    return find_spec("triton") is not None


def triton_available(device, dtype, config):
    """Check capability without importing/compiling kernels or probing memory."""
    return (
        torch.device(device).type == "cuda"
        and torch.version.hip is None
        and dtype in (torch.float32, torch.float64)
        and getattr(config, "field_source", "far") == "all"
        and _has_triton()
    )


def field_backend(device, dtype, config):
    """Use fused full-source CUDA fields when supported, otherwise Torch.

    Capability selection changes no model fields or parameters. Errors after
    kernel selection propagate; there is no execution-time fallback.
    """
    return "triton" if triton_available(device, dtype, config) else "torch"


def _triton_rectangle(qm, mm, charges, config, sizes=None, counts=None):
    if not triton_available(qm.device, qm.dtype, config):
        raise ValueError("Triton fields require CUDA float32/float64 and field_source=all")
    if not mm.shape[1]:
        return _torch_rectangle(qm, mm, charges, config, sizes, counts)
    from .coulomb_triton import coulomb_fields

    phi, field = coulomb_fields(qm, mm, charges, sizes, counts)
    return phi, torch.cat((phi[..., None], field), -1)


def _rectangle(qm, mm, charges, config, sizes=None, counts=None):
    operation = (
        _triton_rectangle
        if field_backend(qm.device, qm.dtype, config) == "triton"
        else _torch_rectangle
    )
    return operation(qm, mm, charges, config, sizes, counts)


def _torch_rectangle(qm, mm, charges, config, sizes=None, counts=None):
    """Full-source fields for independent structures in one rectangular group."""
    if not mm.shape[1]:
        phi = qm.new_zeros(qm.shape[:2])
        return phi, qm.new_zeros(*qm.shape[:2], 4)
    dr = qm[:, :, None] - mm[:, None]
    source = charges[:, None]
    if sizes is not None:
        valid = (
            torch.arange(qm.shape[1], device=qm.device)[None, :, None]
            < sizes[:, None, None]
        ) & (
            torch.arange(mm.shape[1], device=qm.device)[None, None, :]
            < counts[:, None, None]
        )
        # Padding must have a safe norm even for force-loss double backward.
        dr = torch.where(valid[..., None], dr, torch.ones_like(dr))
        source = source * valid
    distance = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-8)
    all_sources = getattr(config, "field_source", "far") == "all"
    phi = (
        COULOMB * (source / distance).sum(-1)
        if all_sources
        else (COULOMB * source / distance).sum(-1)
    )
    descriptor_b = (
        source if all_sources else source * (1 - near_weight(distance, config))
    )
    descriptor_phi = phi if all_sources else COULOMB * (descriptor_b / distance).sum(-1)
    field = COULOMB * (descriptor_b[..., None] * dr / distance[..., None] ** 3).sum(-2)
    return phi, torch.cat((descriptor_phi[..., None], field), -1)


def grouped_potentials(data, config):
    """Batch all sources, with views for equal sizes and padding within buckets.

    Only integer layout metadata crosses to the host. All coordinates, charges
    and derivatives stay on device. The mm_chunk_size option applies only to
    the reference operator; this execution always evaluates each full group.
    """
    qm, mm = data["positions"], data["mm_positions"]
    charges = data["mm_charges"].reshape(-1)
    counts = data["mm_count"].reshape(-1)
    if not len(mm):
        return qm.new_zeros(len(qm)) + charges.sum() * 0, qm.new_zeros(len(qm), 4)
    if len(counts) == 1:
        phi, fields = _rectangle(qm[None], mm[None], charges[None], config)
        return phi.flatten() + charges.sum() * 0, fields.reshape(-1, 4)

    ptr, batch = data["ptr"], data["batch"]
    layout = torch.stack((ptr[1:] - ptr[:-1], counts), dim=1).cpu().tolist()
    if all(pair == layout[0] for pair in layout):
        nq, nm = layout[0]
        phi, fields = _rectangle(
            qm.view(-1, nq, 3), mm.view(-1, nm, 3), charges.view(-1, nm), config
        )
        return phi.flatten() + charges.sum() * 0, fields.reshape(-1, 4)

    buckets = {}
    for graph, (nq, nm) in enumerate(layout):
        key = ((nq - 1).bit_length(), (nm - 1).bit_length() if nm else -1)
        buckets.setdefault(key, []).append(graph)
    qbase, mbase = [0] * len(layout), [0] * len(layout)
    specs, qcapacity, mcapacity = [], 0, 0
    for graphs in buckets.values():
        nq = max(layout[g][0] for g in graphs)
        nm = max(layout[g][1] for g in graphs)
        for slot, graph in enumerate(graphs):
            qbase[graph] = qcapacity + slot * nq
            mbase[graph] = mcapacity + slot * nm
        specs.append((graphs, nq, nm, qcapacity, mcapacity))
        qcapacity += len(graphs) * nq
        mcapacity += len(graphs) * nm
    qbase, mbase = ptr.new_tensor(qbase), ptr.new_tensor(mbase)
    qi = qbase[batch] + torch.arange(len(qm), device=qm.device) - ptr[batch]
    mb = torch.repeat_interleave(
        torch.arange(len(counts), device=qm.device), counts, output_size=len(mm)
    )
    mptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    mi = mbase[mb] + torch.arange(len(mm), device=qm.device) - mptr[mb]
    rq = qm.new_zeros(qcapacity, 3).index_copy(0, qi, qm)
    re = mm.new_zeros(mcapacity, 3).index_copy(0, mi, mm)
    sources = charges.new_zeros(mcapacity).index_copy(0, mi, charges)
    phis, fields = [], []
    for graphs, nq, nm, qstart, mstart in specs:
        n = len(graphs)
        uniform = all(layout[g] == layout[graphs[0]] for g in graphs)
        shape = None if uniform else ptr.new_tensor([layout[g] for g in graphs])
        phi, descriptor = _rectangle(
            rq[qstart : qstart + n * nq].view(n, nq, 3),
            re[mstart : mstart + n * nm].view(n, nm, 3),
            sources[mstart : mstart + n * nm].view(n, nm),
            config,
            None if uniform else shape[:, 0],
            None if uniform else shape[:, 1],
        )
        phis.append(phi.flatten())
        fields.append(descriptor.reshape(-1, 4))
    return (
        torch.cat(phis).index_select(0, qi) + charges.sum() * 0,
        torch.cat(fields).index_select(0, qi),
    )
