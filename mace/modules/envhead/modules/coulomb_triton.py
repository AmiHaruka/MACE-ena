"""Exact full-source Coulomb fields and analytic double backward on CUDA.

Only inputs and upstream gradients survive a forward. Pairwise intermediates
stay in a tile; row reductions write small partial sums, never a pair tensor.
This module is imported lazily by electrostatics, independently of cuEq.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .electrostatics import COULOMB

# Internal execution tiles, not model hyperparameters or workspace budgets.
_QUERY_TILE, _SOURCE_TILE, _SOURCE_QUERY_TILE = 4, 256, 32


@triton.jit
def _geometry(
    Q,
    M,
    B,
    SIZES,
    COUNTS,
    group,
    qi,
    mj,
    NQ: tl.constexpr,
    NM: tl.constexpr,
    RAGGED: tl.constexpr,
):
    qvalid, mvalid = qi < NQ, mj < NM
    if RAGGED:
        qvalid = qvalid & (qi < tl.load(SIZES + group))
        mvalid = mvalid & (mj < tl.load(COUNTS + group))
    valid = qvalid & mvalid
    x = tl.load(Q + (group * NQ + qi) * 3, qvalid, other=0) - tl.load(
        M + (group * NM + mj) * 3, mvalid, other=0
    )
    y = tl.load(Q + (group * NQ + qi) * 3 + 1, qvalid, other=0) - tl.load(
        M + (group * NM + mj) * 3 + 1, mvalid, other=0
    )
    z = tl.load(Q + (group * NQ + qi) * 3 + 2, qvalid, other=0) - tl.load(
        M + (group * NM + mj) * 3 + 2, mvalid, other=0
    )
    x, y, z = tl.where(valid, x, 0.0), tl.where(valid, y, 0.0), tl.where(valid, z, 0.0)
    r2 = x * x + y * y + z * z
    # The eager norm uses precise sqrt; fast approximate sqrt perturbs mixed
    # derivatives of close pairs noticeably in float32.
    radius = tl.sqrt_rn(r2) if r2.dtype == tl.float32 else tl.sqrt(r2)
    epsilon = tl.full((), 1.0e-8, radius.dtype)
    safe_radius = tl.maximum(radius, epsilon)
    inv = tl.div_rn(1.0, safe_radius) if r2.dtype == tl.float32 else 1.0 / safe_radius
    active = radius >= epsilon
    charge = tl.load(B + group * NM + mj, mvalid, other=0)
    return x, y, z, inv, active, charge, valid


@triton.jit
def _contributions(
    Q,
    M,
    B,
    A,
    V,
    U,
    W,
    T,
    SIZES,
    COUNTS,
    group,
    qi,
    mj,
    NQ: tl.constexpr,
    NM: tl.constexpr,
    RAGGED: tl.constexpr,
    PHASE: tl.constexpr,
    SOURCE: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_V: tl.constexpr,
    HAS_U: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_T: tl.constexpr,
    K: tl.constexpr,
):
    x, y, z, inv, active, charge, valid = _geometry(
        Q, M, B, SIZES, COUNTS, group, qi, mj, NQ, NM, RAGGED
    )
    i3 = inv * inv * inv
    if PHASE == 0:
        c = K * charge
        return (
            tl.where(valid, c * inv, 0.0),
            tl.where(valid, c * x * i3, 0.0),
            tl.where(valid, c * y * i3, 0.0),
            tl.where(valid, c * z * i3, 0.0),
            x * 0.0,
            x * 0.0,
            x * 0.0,
        )
    a, vx, vy, vz = x * 0.0, x * 0.0, x * 0.0, x * 0.0
    if HAS_A:
        a = tl.load(A + group * NQ + qi, qi < NQ, other=0)
    if HAS_V:
        vx = tl.load(V + (group * NQ + qi) * 3, qi < NQ, other=0)
        vy = tl.load(V + (group * NQ + qi) * 3 + 1, qi < NQ, other=0)
        vz = tl.load(V + (group * NQ + qi) * 3 + 2, qi < NQ, other=0)
    vd = vx * x + vy * y + vz * z
    i5 = i3 * inv * inv
    ax = vx * i3 - tl.where(active, a * x * i3 + 3.0 * vd * x * i5, 0.0)
    ay = vy * i3 - tl.where(active, a * y * i3 + 3.0 * vd * y * i5, 0.0)
    az = vz * i3 - tl.where(active, a * z * i3 + 3.0 * vd * z * i5, 0.0)
    sign: tl.constexpr = -1.0 if SOURCE else 1.0
    if PHASE == 1:
        return (
            tl.where(valid, sign * K * charge * ax, 0.0),
            tl.where(valid, sign * K * charge * ay, 0.0),
            tl.where(valid, sign * K * charge * az, 0.0),
            tl.where(valid, K * (a * inv + vd * i3), 0.0),
            x * 0.0,
            x * 0.0,
            x * 0.0,
        )
    ux, uy, uz = x * 0.0, x * 0.0, x * 0.0
    if HAS_U:
        ux = tl.load(U + (group * NQ + qi) * 3, qi < NQ, other=0)
        uy = tl.load(U + (group * NQ + qi) * 3 + 1, qi < NQ, other=0)
        uz = tl.load(U + (group * NQ + qi) * 3 + 2, qi < NQ, other=0)
    if HAS_W:
        ux = ux - tl.load(W + (group * NM + mj) * 3, mj < NM, other=0)
        uy = uy - tl.load(W + (group * NM + mj) * 3 + 1, mj < NM, other=0)
        uz = uz - tl.load(W + (group * NM + mj) * 3 + 2, mj < NM, other=0)
    t = x * 0.0
    if HAS_T:
        t = tl.load(T + group * NM + mj, mj < NM, other=0)
    dd = ux * x + uy * y + uz * z
    vu = ux * vx + uy * vy + uz * vz
    i7 = i5 * inv * inv
    bx = tl.where(
        active,
        -a * ux * i3
        + 3.0 * a * x * dd * i5
        - 3.0 * (vx * dd + x * vu + vd * ux) * i5
        + 15.0 * vd * x * dd * i7,
        0.0,
    )
    by = tl.where(
        active,
        -a * uy * i3
        + 3.0 * a * y * dd * i5
        - 3.0 * (vy * dd + y * vu + vd * uy) * i5
        + 15.0 * vd * y * dd * i7,
        0.0,
    )
    bz = tl.where(
        active,
        -a * uz * i3
        + 3.0 * a * z * dd * i5
        - 3.0 * (vz * dd + z * vu + vd * uz) * i5
        + 15.0 * vd * z * dd * i7,
        0.0,
    )
    c0 = tl.where(valid, sign * K * (charge * bx + t * ax), 0.0)
    c1 = tl.where(valid, sign * K * (charge * by + t * ay), 0.0)
    c2 = tl.where(valid, sign * K * (charge * bz + t * az), 0.0)
    if SOURCE:
        return (
            c0,
            c1,
            c2,
            tl.where(valid, K * (ax * ux + ay * uy + az * uz), 0.0),
            x * 0.0,
            x * 0.0,
            x * 0.0,
        )
    c3 = tl.where(valid, K * (-tl.where(active, charge * dd * i3, 0.0) + t * inv), 0.0)
    c4 = tl.where(
        valid,
        K
        * (charge * (ux * i3 - tl.where(active, 3.0 * x * dd * i5, 0.0)) + t * x * i3),
        0.0,
    )
    c5 = tl.where(
        valid,
        K
        * (charge * (uy * i3 - tl.where(active, 3.0 * y * dd * i5, 0.0)) + t * y * i3),
        0.0,
    )
    c6 = tl.where(
        valid,
        K
        * (charge * (uz * i3 - tl.where(active, 3.0 * z * dd * i5, 0.0)) + t * z * i3),
        0.0,
    )
    return c0, c1, c2, c3, c4, c5, c6


@triton.jit
def _rows(
    Q,
    M,
    B,
    A,
    V,
    U,
    W,
    T,
    SIZES,
    COUNTS,
    OUT,
    NQ: tl.constexpr,
    NM: tl.constexpr,
    SPLITS: tl.constexpr,
    RAGGED: tl.constexpr,
    PHASE: tl.constexpr,
    C: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_V: tl.constexpr,
    HAS_U: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_T: tl.constexpr,
    BQ: tl.constexpr,
    BM: tl.constexpr,
    K: tl.constexpr,
):
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)[:, None]
    mj = tl.program_id(1) * BM + tl.arange(0, BM)[None, :]
    group = tl.program_id(2)
    values = _contributions(
        Q,
        M,
        B,
        A,
        V,
        U,
        W,
        T,
        SIZES,
        COUNTS,
        group,
        qi,
        mj,
        NQ,
        NM,
        RAGGED,
        PHASE,
        False,
        HAS_A,
        HAS_V,
        HAS_U,
        HAS_W,
        HAS_T,
        K,
    )
    offset = ((group * NQ + tl.reshape(qi, (BQ,))) * SPLITS + tl.program_id(1)) * C
    for channel in tl.static_range(C):
        value = tl.sum(values[channel], axis=1)
        tl.store(OUT + offset + channel, value, tl.reshape(qi < NQ, (BQ,)))


@triton.jit
def _finish(
    PARTIAL,
    OUT,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    C: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    split = tl.arange(0, BS)
    offset = ((row[:, None] // C) * SPLITS + split[None, :]) * C + row[:, None] % C
    value = tl.load(
        PARTIAL + offset, (row[:, None] < N) & (split[None, :] < SPLITS), other=0
    )
    tl.store(OUT + row, tl.sum(value, 1), row < N)


@triton.jit
def _sources(
    Q,
    M,
    B,
    A,
    V,
    U,
    W,
    T,
    SIZES,
    COUNTS,
    GM,
    GB,
    NQ: tl.constexpr,
    NM: tl.constexpr,
    RAGGED: tl.constexpr,
    PHASE: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_V: tl.constexpr,
    HAS_U: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_T: tl.constexpr,
    NEED_M: tl.constexpr,
    NEED_B: tl.constexpr,
    TILE: tl.constexpr,
    K: tl.constexpr,
):
    mj = tl.program_id(0) * TILE + tl.arange(0, TILE)[:, None]
    group = tl.program_id(1)
    # Use the input dtype for accumulation, including the float64 reference.
    seed = tl.load(
        B + group * NM + tl.reshape(mj, (TILE,)), tl.reshape(mj < NM, (TILE,)), other=0
    )
    g0, g1, g2, g3 = seed * 0.0, seed * 0.0, seed * 0.0, seed * 0.0
    for start in range(tl.cdiv(NQ, TILE)):
        qi = start * TILE + tl.arange(0, TILE)[None, :]
        values = _contributions(
            Q,
            M,
            B,
            A,
            V,
            U,
            W,
            T,
            SIZES,
            COUNTS,
            group,
            qi,
            mj,
            NQ,
            NM,
            RAGGED,
            PHASE,
            True,
            HAS_A,
            HAS_V,
            HAS_U,
            HAS_W,
            HAS_T,
            K,
        )
        if NEED_M:
            g0 += tl.sum(values[0], 1)
            g1 += tl.sum(values[1], 1)
            g2 += tl.sum(values[2], 1)
        if NEED_B:
            g3 += tl.sum(values[3], 1)
    offset = group * NM + tl.reshape(mj, (TILE,))
    mask = tl.reshape(mj < NM, (TILE,))
    if NEED_M:
        tl.store(GM + offset * 3, g0, mask)
        tl.store(GM + offset * 3 + 1, g1, mask)
        tl.store(GM + offset * 3 + 2, g2, mask)
    if NEED_B:
        tl.store(GB + offset, g3, mask)


def _row_reduction(
    q, m, b, sizes, counts, phase, a=None, v=None, u=None, w=None, t=None
):
    groups, nq = q.shape[:2]
    nm = m.shape[1]
    channels = (4, 3, 7)[phase]
    splits = triton.cdiv(nm, _SOURCE_TILE)
    partial = q.new_empty(groups, nq, splits, channels)
    _rows[(triton.cdiv(nq, _QUERY_TILE), splits, groups)](
        q,
        m,
        b,
        a,
        v,
        u,
        w,
        t,
        sizes,
        counts,
        partial,
        nq,
        nm,
        splits,
        sizes is not None,
        phase,
        channels,
        a is not None,
        v is not None,
        u is not None,
        w is not None,
        t is not None,
        _QUERY_TILE,
        _SOURCE_TILE,
        COULOMB,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if splits == 1:
        return partial.view(groups, nq, channels)
    out = q.new_empty(groups, nq, channels)
    _finish[(triton.cdiv(out.numel(), 16),)](
        partial,
        out,
        out.numel(),
        splits,
        channels,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return out


def _source_reduction(
    q,
    m,
    b,
    sizes,
    counts,
    phase,
    need_m,
    need_b,
    a=None,
    v=None,
    u=None,
    w=None,
    t=None,
):
    gm = torch.empty_like(m) if need_m else None
    gb = torch.empty_like(b) if need_b else None
    if need_m or need_b:
        _sources[(triton.cdiv(m.shape[1], _SOURCE_QUERY_TILE), q.shape[0])](
            q,
            m,
            b,
            a,
            v,
            u,
            w,
            t,
            sizes,
            counts,
            gm,
            gb,
            q.shape[1],
            m.shape[1],
            sizes is not None,
            phase,
            a is not None,
            v is not None,
            u is not None,
            w is not None,
            t is not None,
            need_m,
            need_b,
            _SOURCE_QUERY_TILE,
            COULOMB,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return gm, gb


def _contiguous(value):
    return None if value is None else value.contiguous()


class _CoulombVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, m, b, a, v, sizes, counts, need_q, need_m, need_b):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, m, b, a, v, sizes, counts)
        gq = _row_reduction(q, m, b, sizes, counts, 1, a, v) if need_q else None
        gm, gb = _source_reduction(q, m, b, sizes, counts, 1, need_m, need_b, a, v)
        return gq, gm, gb

    @staticmethod
    @once_differentiable
    def backward(ctx, u, w, t):
        q, m, b, a, v, sizes, counts = ctx.saved_tensors
        u, w, t = _contiguous(u), _contiguous(w), _contiguous(t)
        need_q, need_m, need_b, need_a, need_v = ctx.needs_input_grad[:5]
        rows = (
            _row_reduction(q, m, b, sizes, counts, 2, a, v, u, w, t)
            if need_q or need_a or need_v
            else None
        )
        gm, gb = _source_reduction(
            q, m, b, sizes, counts, 2, need_m, need_b, a, v, u, w, t
        )
        return (
            rows[..., :3] if need_q else None,
            gm,
            gb,
            rows[..., 3] if need_a else None,
            rows[..., 4:] if need_v else None,
            None,
            None,
            None,
            None,
            None,
        )


class _CoulombFields(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, m, b, sizes, counts):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, m, b, sizes, counts)
        values = _row_reduction(q, m, b, sizes, counts, 0)
        return values[..., 0], values[..., 1:]

    @staticmethod
    def backward(ctx, a, v):
        q, m, b, sizes, counts = ctx.saved_tensors
        gradients = _CoulombVJP.apply(
            q,
            m,
            b,
            _contiguous(a),
            _contiguous(v),
            sizes,
            counts,
            *ctx.needs_input_grad[:3],
        )
        return *gradients, None, None


def coulomb_fields(qm, mm, charges, sizes=None, counts=None):
    """Return phi [B,Q] and E [B,Q,3]; differentiable twice in all inputs.

    Uniform groups omit sizes/counts. Ragged groups supply both, and invalid
    padding contributes neither values nor gradients. Third backward is outside
    the E/F/q contract and raises through once_differentiable.
    """
    if not qm.is_cuda or qm.dtype not in (torch.float32, torch.float64):
        raise ValueError("Fused Coulomb fields require CUDA float32 or float64")
    if not mm.shape[1]:
        phi = qm.new_zeros(qm.shape[:2]) + charges.sum() * 0
        return phi, qm.new_zeros(*qm.shape[:2], 3)
    with torch.cuda.device(qm.device):
        return _CoulombFields.apply(
            qm.contiguous(),
            mm.contiguous(),
            charges.contiguous(),
            _contiguous(sizes),
            _contiguous(counts),
        )
