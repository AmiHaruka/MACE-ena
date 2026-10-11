"""POL-EF-style invariant response basis, without explicit multipole outputs."""

import torch
from e3nn import o3


class ResponseReadout(torch.nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.hidden, self.config = hidden, config
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                config.channels * (3 + (config.max_l == 2)) + 2, config.channels
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(config.channels, 1),
        )
        # Small, nonzero output allows a useful first-step gradient to the head.
        torch.nn.init.normal_(self.mlp[-1].weight, std=0.01)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, q, far):
        c = self.config.channels
        e = far[:, 1:]
        scalar = h[:, :c]
        vector = h[:, c : 4 * c].reshape(-1, c, 3)
        invariants = [
            scalar,
            (vector * e[:, None]).sum(-1),
            scalar * e.square().sum(-1, keepdim=True),
        ]
        if self.config.max_l == 2:
            rank2 = h[:, 4 * c :].reshape(-1, c, 5)
            y2 = o3.spherical_harmonics(
                2, e, normalize=False, normalization="component"
            )
            invariants.append((rank2 * y2[:, None]).sum(-1))
        invariants.extend(
            [
                q[:, None]
                if self.config.residual_charges
                else torch.zeros_like(q[:, None]),
                far[:, :1],
            ]
        )
        return self.mlp(torch.cat(invariants, -1)).squeeze(-1)
