"""PaiNN scalar/vector update and E2Former first-order node factorization.

E2 reference implements T(r_i, sum_j a_ij h_j)-sum_j a_ij T(r_j,h_j),
using Cartesian/e3nn tensor products. This is the first-order node-factorized
identity of the UBio-MolFM E2Former SO2 path, with explicit O(3) parity.
It is not the upstream trained architecture or its SO2 acceleration kernel.
"""

import math
import torch
from e3nn import o3


def scatter(x, index, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add(0, index, x)


class PaiNNOperator(torch.nn.Module):
    def __init__(self, channels, irreps, radial_basis):
        super().__init__()
        self.c, self.irreps = channels, irreps
        self.context = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.SiLU(),
            torch.nn.Linear(channels, 3 * channels),
        )
        self.filter = torch.nn.Linear(radial_basis, 3 * channels)
        self.u = torch.nn.Linear(channels, channels, bias=False)
        self.v = torch.nn.Linear(channels, channels, bias=False)
        self.mix = torch.nn.Sequential(
            torch.nn.Linear(2 * channels, channels),
            torch.nn.SiLU(),
            torch.nn.Linear(channels, 3 * channels),
        )

    def pack(self, s, v):
        return torch.cat(
            (s, v.flatten(1), s.new_zeros(len(s), self.irreps.dim - 4 * self.c)), -1
        )

    def forward(
        self, x, src, dst, vector, distance, radial, weight, positions, triton=False
    ):
        c = self.c
        s, v = x[:, :c], x[:, c : 4 * c].reshape(-1, c, 3)
        ds, dv, dr = (
            self.context(s[src]) * self.filter(radial) * weight[:, None]
        ).chunk(3, -1)
        vm = dv[:, :, None] * v[src] + dr[:, :, None] * vector[:, None, :] / distance[
            :, None, None
        ].clamp_min(1e-8)
        message = self.pack(scatter(ds, dst, len(x)), scatter(vm, dst, len(x))) / 4
        sm = s + message[:, :c]
        vv = v + message[:, c : 4 * c].reshape(-1, c, 3)
        uv = self.u(vv.transpose(1, 2)).transpose(1, 2)
        vv2 = self.v(vv.transpose(1, 2)).transpose(1, 2)
        norm = torch.sqrt(vv2.square().sum(-1) + 1e-12)
        av, ass, asv = self.mix(torch.cat((sm, norm), -1)).chunk(3, -1)
        updated = self.pack(
            sm + ass + asv * (uv * vv2).sum(-1), vv + av[:, :, None] * uv
        )
        return message, updated


class E2FormerOperator(torch.nn.Module):
    def __init__(self, channels, irreps, radial_basis):
        super().__init__()
        self.c, self.irreps = channels, irreps
        self.query = torch.nn.Linear(channels, channels, bias=False)
        self.key = torch.nn.Linear(channels, channels, bias=False)
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(radial_basis, channels),
            torch.nn.SiLU(),
            torch.nn.Linear(channels, 1),
        )
        self.value = o3.Linear(irreps, irreps)
        self.tp = o3.FullyConnectedTensorProduct(o3.Irreps("1o"), irreps, irreps)
        self.output = o3.Linear(irreps, irreps)
        self.gate = torch.nn.Linear(channels, channels * (len(irreps) - 1))

    def aggregate(self, values, alpha, src, dst, n, accelerated):
        if not accelerated:
            return scatter(values[src] * alpha[:, None], dst, n)
        from .triton_ops.sparse_v_agg_lastdim_autograd import (
            sparse_v_agg_edge_index_triton,
        )

        return sparse_v_agg_edge_index_triton(
            values[:, None, :], alpha[:, None], dst, src, n
        ).squeeze(1)

    def forward(
        self, x, src, dst, vector, distance, radial, weight, positions, triton=False
    ):
        logits = (self.query(x[:, : self.c])[dst] * self.key(x[:, : self.c])[src]).sum(
            -1
        ) / math.sqrt(self.c)
        a = torch.sigmoid(logits + self.radial(radial).squeeze(-1))
        # Fixed self mass and squared smooth weights avoid abs(q) cusps at zero.
        denominator = 1 + scatter(a * weight.square(), dst, len(x))
        alpha = a * weight / denominator[dst]
        value = self.value(x)
        agg = self.aggregate(value, alpha, src, dst, len(x), triton)
        positional = self.aggregate(
            self.tp(positions, value), alpha, src, dst, len(x), triton
        )
        message = self.output(agg + self.tp(positions, agg) - positional)
        updated = x + message
        scalars = torch.nn.functional.silu(updated[:, : self.c])
        gates = torch.sigmoid(self.gate(scalars)).reshape(
            len(x), len(self.irreps) - 1, self.c
        )
        blocks = [scalars]
        for k, ((mul, ir), sl) in enumerate(
            zip(list(self.irreps)[1:], self.irreps.slices()[1:])
        ):
            blocks.append(
                (
                    updated[:, sl].reshape(-1, mul, ir.dim) * gates[:, k, :, None]
                ).flatten(1)
            )
        return message, torch.cat(blocks, -1)
