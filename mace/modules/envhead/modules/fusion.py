"""Interchangeable O(3) fusion blocks; common density/energy contracts."""
import torch
from mace.modules.blocks import (
    RealAgnosticResidualInteractionBlock,
    EquivariantProductBasisBlock,
)
from mace.modules.wrapper_ops import Linear, FullyConnectedTensorProduct


def unflatten(x, channels):
    return torch.cat(
        [
            x[:, s:e].reshape(len(x), channels, l)
            for s, e, l in (
                (0, channels, 1),
                (channels, 4 * channels, 3),
                (4 * channels, 9 * channels, 5),
            )
        ],
        -1,
    )


class FusionBlock(torch.nn.Module):
    def __init__(self, hidden, density, interaction, config, cueq_config=None):
        super().__init__()
        self.mode = config.fusion
        self.channels = config.channels
        self.avg = max(float(interaction.avg_num_neighbors), 1.0)
        self.maps = torch.nn.ModuleList(
            [
                Linear(ir, hidden, cueq_config=cueq_config)
                for ir in (hidden, hidden, density, density, density)
            ]
        )
        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=hidden,
            target_irreps=hidden,
            correlation=config.correlation,
            num_elements=interaction.node_attrs_irreps.dim,
            use_sc=True,
            cueq_config=cueq_config,
        )
        if self.mode in ("additive", "product"):
            self.interaction = RealAgnosticResidualInteractionBlock(
                node_attrs_irreps=interaction.node_attrs_irreps,
                node_feats_irreps=hidden,
                edge_attrs_irreps=interaction.edge_attrs_irreps,
                edge_feats_irreps=interaction.edge_feats_irreps,
                target_irreps=hidden,
                hidden_irreps=hidden,
                avg_num_neighbors=self.avg,
                radial_MLP=[config.channels, config.channels],
                cueq_config=cueq_config,
            )
        if self.mode == "product":
            fields = (density + density).simplify()
            self.geometry_field = FullyConnectedTensorProduct(
                hidden, fields, hidden, cueq_config=cueq_config
            )
            self.density_field = FullyConnectedTensorProduct(
                density, fields, hidden, cueq_config=cueq_config
            )
        if self.mode == "message":
            self.sender = Linear(hidden, hidden, cueq_config=cueq_config)
            self.receiver = Linear(hidden, hidden, cueq_config=cueq_config)
            self.message_tp = FullyConnectedTensorProduct(
                hidden, interaction.edge_attrs_irreps, hidden, cueq_config=cueq_config
            )
            self.weights = torch.nn.Sequential(
                torch.nn.Linear(
                    2 * config.channels + interaction.edge_feats_irreps.dim,
                    config.channels,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(config.channels, 3 * config.channels),
            )

    def forward(self, h, z, p, internal, external, attrs, edges, sh, radial, cutoff):
        mixed = sum(
            layer(value)
            for layer, value in zip(self.maps, (h, z, p, internal, external))
        )
        if self.mode == "product":
            fields = torch.cat((internal, external), -1)
            mixed = (
                mixed + self.geometry_field(h, fields) + self.density_field(p, fields)
            )
        if self.mode == "message":
            s, r = edges
            edge = self.sender(mixed[s]) + self.receiver(mixed[r])
            weights = self.weights(
                torch.cat(
                    (mixed[s, : self.channels], mixed[r, : self.channels], radial), -1
                )
            ).tanh()
            # Bounded edge coefficients prevent state-dependent message weights
            # from adding uncontrolled polynomial growth before the product basis.
            # The multipole outputs themselves are not clipped.
            weights = torch.cat(
                [
                    part.repeat_interleave(2 * l + 1, -1)
                    for l, part in enumerate(weights.split(self.channels, -1))
                ],
                -1,
            )
            messages = self.message_tp(edge, sh) * weights
            if cutoff is not None:
                messages = messages * cutoff.reshape(-1, 1)
            # Smooth local mean, including a unit self weight. Receiver-conditioned
            # messages contain a repeated receiver contribution; a fixed backbone
            # average does not bound that contribution on denser neighborhoods.
            edge_weight = (
                torch.ones_like(r, dtype=mixed.dtype)
                if cutoff is None
                else cutoff.reshape(-1)
            )
            degree = mixed.new_ones(len(mixed)).index_add(0, r, edge_weight)
            aggregate = (
                mixed.new_zeros(mixed.shape).index_add(0, r, messages) / degree[:, None]
            )
            values, sc = unflatten(aggregate, self.channels), mixed
        else:
            values, sc = self.interaction(
                node_attrs=attrs,
                node_feats=mixed,
                edge_attrs=sh,
                edge_feats=radial,
                edge_index=edges,
                cutoff=cutoff,
            )
        return self.product(values, sc, attrs)
