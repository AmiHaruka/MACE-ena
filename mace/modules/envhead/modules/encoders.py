"""Equivariant environment operators and all-source descriptors."""

import torch
from e3nn import o3
from mace.modules.blocks import EquivariantProductBasisBlock
from mace.modules.irreps_tools import reshape_irreps

COULOMB = 14.3996454784255  # eV Angstrom / e^2


def flatten_message(x, irreps):
    chunks, start = [], 0
    for mul, ir in irreps:
        chunks.append(x[:, :, start : start + ir.dim].reshape(len(x), mul * ir.dim))
        start += ir.dim
    return torch.cat(chunks, -1)


def sum_nodes(x, ids, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add(0, ids, x)


class EnvironmentInteraction(torch.nn.Module):
    def __init__(self, qm_irreps, out_irreps, hidden, cutoff, config):
        super().__init__()
        self.hidden, self.config, self.cutoff = hidden, config, cutoff
        self.qm_map = o3.Linear(qm_irreps, hidden)
        self.output = o3.Linear(hidden, out_irreps)
        self.output_layout = reshape_irreps(out_irreps)
        self.sh = o3.Irreps.spherical_harmonics(config.max_l)
        self.tp = o3.FullyConnectedTensorProduct(
            hidden, self.sh, hidden, shared_weights=False
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(config.radial_basis, config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(config.channels, self.tp.weight_numel),
        )
        self.register_buffer("centers", torch.linspace(0, 1, config.radial_basis))
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=hidden,
            target_irreps=hidden,
            correlation=config.correlation,
            num_elements=1,
            use_sc=True,
        )
        self.layout = reshape_irreps(hidden)
        self.operator = None
        if config.backend != "ace":
            from .operators import PaiNNOperator, E2FormerOperator

            cls = PaiNNOperator if config.backend == "painn" else E2FormerOperator
            self.operator = cls(config.channels, hidden, config.radial_basis)
            del self.tp, self.radial, self.product

    def forward(self, qm_features, env_features, graph, charges):
        qn = graph.qm_count
        x = torch.cat((self.qm_map(qm_features), env_features))
        src, dst = graph.edge_index
        keep = (src >= qn) | (dst >= qn)
        if not self.config.bidirectional:
            keep &= src >= qn
        src, dst = src[keep], dst[keep]
        d, v = graph.lengths[keep], graph.vectors[keep]
        w = graph.edge_weight[keep]
        # ENV chemical context is retained even for zero-charge nodes. Gate
        # transfer into QM so the all-zero-charge environment recovers vacuum.
        source_q = torch.cat(
            (charges.new_ones(qn), charges[graph.env_indices].flatten())
        )
        w = w * torch.where(dst < qn, source_q[src], torch.ones_like(w))
        radial = torch.exp(
            -(
                ((d[:, None] / self.cutoff - self.centers) * self.config.radial_basis)
                ** 2
            )
        )
        if self.operator is not None:
            if self.config.triton and not x.is_cuda:
                raise ValueError("Triton aggregation requires CUDA")
            if self.config.triton and x.dtype != torch.float32:
                raise ValueError(
                    "Vendored Triton aggregation currently supports float32"
                )
            # Center positions to improve cancellation in node factorization.
            sizes = torch.bincount(graph.batch, minlength=int(graph.batch.max()) + 1)
            center = (
                sum_nodes(graph.positions, graph.batch, len(sizes)) / sizes[:, None]
            )
            aggregated, updated = self.operator(
                x,
                src,
                dst,
                v,
                d,
                radial,
                w,
                graph.positions - center[graph.batch],
                self.config.triton,
            )
            return self.output_layout(self.output(aggregated[:qn])), updated[qn:]
        sh = o3.spherical_harmonics(
            self.sh, v, normalize=True, normalization="component"
        )
        messages = self.tp(x[src], sh, self.radial(radial)) * w[:, None]
        aggregated = sum_nodes(messages, dst, len(x))
        # Fixed normalization keeps vanishing nodes from changing neighbor counts.
        aggregated = aggregated / 4.0
        updated = self.product(
            self.layout(aggregated), sc=x, node_attrs=x.new_ones(len(x), 1)
        )
        return self.output_layout(self.output(aggregated[:qn])), updated[qn:]


class GlobalEncoder(torch.nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.config, self.hidden = config, hidden
        self.sh = o3.Irreps.spherical_harmonics(config.max_l)
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(config.channels, config.radial_basis),
        )
        self.learned_irreps = o3.Irreps(
            [(config.radial_basis, ir) for _, ir in self.sh]
        )
        self.project = o3.Linear(
            self.learned_irreps
            if config.global_mode == "learned"
            else o3.Irreps("0e+1o"),
            hidden,
        )

    def forward(self, data):
        r, b = data["positions"], data["mm_charges"].flatten()
        mb = torch.repeat_interleave(
            torch.arange(len(data["ptr"]) - 1, device=r.device),
            data["mm_count"].reshape(-1),
        )
        dim = self.learned_irreps.dim if self.config.global_mode == "learned" else 4
        feat = r.new_zeros(len(r), dim)
        for start in range(0, len(b), self.config.mm_chunk_size):
            dr = (
                r[:, None, :]
                - data["mm_positions"][
                    None, start : start + self.config.mm_chunk_size, :
                ]
            )
            distance = torch.linalg.vector_norm(dr, dim=-1).clamp_min(1e-8)
            q = b[start : start + self.config.mm_chunk_size][None, :] * (
                data["batch"][:, None]
                == mb[None, start : start + self.config.mm_chunk_size]
            )
            if self.config.global_mode == "physical":
                part = (
                    torch.cat(
                        (
                            (q / distance).sum(1, keepdim=True),
                            (q[:, :, None] * dr / distance[:, :, None] ** 3).sum(1),
                        ),
                        -1,
                    )
                    * COULOMB
                )
            else:
                radial = (
                    self.radial((1 / (1 + distance)).unsqueeze(-1))
                    * q[:, :, None]
                    / distance[:, :, None]
                )
                sh = o3.spherical_harmonics(
                    self.sh, dr, normalize=True, normalization="component"
                )
                chunks, offset = [], 0
                for _, ir in self.sh:
                    chunks.append(
                        (
                            radial[:, :, :, None]
                            * sh[:, :, None, offset : offset + ir.dim]
                        )
                        .sum(1)
                        .flatten(1)
                    )
                    offset += ir.dim
                part = torch.cat(chunks, -1)
            feat = feat + part
        # Preserve an empty-source derivative path for gas batches.
        return self.project(feat) + b.sum() * 0
