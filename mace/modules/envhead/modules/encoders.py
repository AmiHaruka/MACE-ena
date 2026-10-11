"""ACE environment messages in the selected near graph."""

from dataclasses import dataclass

import torch
from e3nn import o3
from mace.modules.blocks import EquivariantProductBasisBlock
from mace.modules.irreps_tools import reshape_irreps, tp_out_irreps_with_instructions

COULOMB = 14.3996454784255  # eV Angstrom / e^2


def flatten_message(x, irreps):
    chunks, start = [], 0
    for mul, ir in irreps:
        chunks.append(x[:, :, start : start + ir.dim].reshape(len(x), mul * ir.dim))
        start += ir.dim
    return torch.cat(chunks, -1)


def sum_nodes(x, ids, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add(0, ids, x)


@dataclass
class EnvironmentGeometry:
    """Differentiable geometric inputs belonging to one forward only."""

    src: torch.Tensor
    dst: torch.Tensor
    weight: torch.Tensor
    radial: torch.Tensor
    spherical: torch.Tensor


class EnvironmentInteraction(torch.nn.Module):
    def __init__(self, qm_irreps, out_irreps, hidden, cutoff, config):
        super().__init__()
        self.hidden, self.config, self.cutoff = hidden, config, cutoff
        self.qm_map = o3.Linear(qm_irreps, hidden)
        self.output = o3.Linear(hidden, out_irreps)
        self.output_layout = reshape_irreps(out_irreps)
        self.sh = o3.Irreps.spherical_harmonics(config.max_l)
        self.message_mode = config.message_mode
        if self.message_mode == "channelwise":
            mid, instructions = tp_out_irreps_with_instructions(hidden, self.sh, hidden)
            self.linear_up = o3.Linear(hidden, hidden)
            self.tp = o3.TensorProduct(
                hidden,
                self.sh,
                mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )
            self.linear_down = o3.Linear(mid, hidden)
        else:
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

    def prepare_geometry(self, graph):
        src, dst = graph.edge_index
        keep = (src >= graph.qm_count) | (dst >= graph.qm_count)
        d, v = graph.lengths[keep], graph.vectors[keep]
        radial = torch.exp(
            -(
                ((d[:, None] / self.cutoff - self.centers) * self.config.radial_basis)
                ** 2
            )
        )
        sh = o3.spherical_harmonics(
            self.sh, v, normalize=True, normalization="component"
        )
        return EnvironmentGeometry(
            src[keep], dst[keep], graph.edge_weight[keep], radial, sh
        )

    def forward(
        self, qm_features, env_features, graph, charges, update_env=True, geometry=None
    ):
        qn = graph.qm_count
        x = torch.cat((self.qm_map(qm_features), env_features))
        if getattr(self, "message_mode", "full") == "channelwise":
            send_features = self.linear_up(x)
        else:
            send_features = x
        src, dst = (
            graph.edge_index if geometry is None else (geometry.src, geometry.dst)
        )
        keep = (src >= qn) | (dst >= qn)
        if not update_env:
            keep &= dst < qn
        if not self.config.bidirectional:
            keep &= src >= qn
        src, dst = src[keep], dst[keep]
        if geometry is None:
            d, v = graph.lengths[keep], graph.vectors[keep]
            w = graph.edge_weight[keep]
        else:
            w = geometry.weight[keep]
        # ENV chemical context is retained even for zero-charge nodes. Gate
        # transfer into QM so the all-zero-charge environment recovers vacuum.
        source_q = torch.cat(
            (charges.new_ones(qn), charges[graph.env_indices].flatten())
        )
        w = w * torch.where(dst < qn, source_q[src], torch.ones_like(w))
        if geometry is None:
            radial = torch.exp(
                -(
                    (
                        (d[:, None] / self.cutoff - self.centers)
                        * self.config.radial_basis
                    )
                    ** 2
                )
            )
            sh = o3.spherical_harmonics(
                self.sh, v, normalize=True, normalization="component"
            )
        else:
            radial, sh = geometry.radial[keep], geometry.spherical[keep]
        messages = self.tp(send_features[src], sh, self.radial(radial)) * w[:, None]
        aggregated = sum_nodes(messages, dst, len(x) if update_env else qn)
        if getattr(self, "message_mode", "full") == "channelwise":
            aggregated = self.linear_down(aggregated)
        # Fixed normalization keeps vanishing nodes from changing neighbor counts.
        aggregated = aggregated / 4.0
        updated = None
        if update_env and len(env_features):
            updated = self.product(
                self.layout(aggregated[qn:]),
                sc=x[qn:],
                node_attrs=x.new_ones(len(env_features), 1),
            )
        elif update_env:
            updated = env_features
        return self.output_layout(self.output(aggregated[:qn])), updated
