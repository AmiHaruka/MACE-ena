"""EnvHead v2: single foundation pass with environment-conditioned messages."""

from copy import deepcopy
import torch
from e3nn import o3
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.extensions import MACELES
from .config import EnvHeadConfig
from .graph import build_graph
from .encoders import EnvironmentInteraction, GlobalEncoder, flatten_message, sum_nodes
from .response import response_outputs


class EnvMACE(torch.nn.Module):
    format_version = 2

    def __init__(self, foundation, config=None, dtype=None):
        super().__init__()
        self.config = config or EnvHeadConfig()
        if type(foundation) not in (MACE, ScaleShiftMACE, MACELES):
            raise TypeError("v2 supports native MACE, ScaleShiftMACE and MACELES")
        if len(getattr(foundation, "heads", ["Default"])) != 1:
            raise ValueError("v2 requires a single foundation energy head")
        self.foundation = deepcopy(foundation)
        dtype = dtype or next(foundation.parameters()).dtype
        self.foundation.to(dtype=dtype)
        self.foundation.requires_grad_(not self.config.freeze_foundation)
        self.hidden = o3.Irreps(
            [
                (self.config.channels, (l, (-1) ** l))
                for l in range(self.config.max_l + 1)
            ]
        )
        self.env_embedding = torch.nn.Embedding(119, self.config.channels)
        self.env_charge_embedding = torch.nn.Linear(1, self.config.channels, bias=False)
        self.total_charge_embedding = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, i.target_irreps[0].mul, bias=False)
                for i in self.foundation.interactions
            ]
        )
        self.env_layers = torch.nn.ModuleList(
            [
                EnvironmentInteraction(
                    i.node_feats_irreps,
                    i.target_irreps,
                    self.hidden,
                    float(self.r_max),
                    self.config,
                )
                for i in self.foundation.interactions
            ]
        )
        self.mixers = torch.nn.ModuleList(
            [
                o3.Linear(p.linear.irreps_out, self.hidden)
                for p in self.foundation.products
            ]
        )
        self.last_mixer = o3.Linear(
            self.foundation.interactions[-1].target_irreps, self.hidden
        )
        self.global_encoder = GlobalEncoder(self.hidden, self.config)
        # Scalar contractions of each angular block preserve parity and rotation.
        nscalar = self.config.channels * (2 + 2 * (self.config.max_l + 1))
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(nscalar, self.config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.channels, 1),
        )
        self.to(dtype=dtype, device=next(foundation.parameters()).device)
        if self.config.cueq:
            from ..backends import enable_cueq

            enable_cueq(self)

    @property
    def r_max(self):
        return self.foundation.r_max

    @property
    def atomic_numbers(self):
        return self.foundation.atomic_numbers

    @property
    def heads(self):
        return getattr(self.foundation, "heads", ["Default"])

    def _invariants(self, h, f):
        c = self.config.channels
        result = [h[:, :c], f[:, :c]]
        for (mul, ir), sl in zip(self.hidden, self.hidden.slices()):
            a, b = h[:, sl].reshape(-1, mul, ir.dim), f[:, sl].reshape(-1, mul, ir.dim)
            result.extend(((a * b).sum(-1), b.square().sum(-1)))
        return torch.cat(result, -1)

    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_response=False,
        compute_charges=False,
    ):
        with torch.enable_grad():
            data = dict(data)
            if data.get("pbc") is not None and data["pbc"].any():
                raise ValueError("EnvHead v2 requires nonperiodic data")
            if data.get("total_spin") is not None and (data["total_spin"] != 1).any():
                raise ValueError("EnvHead v2 currently supports singlet data")
            data["positions"] = data["positions"].clone().requires_grad_(True)
            data["mm_positions"] = data["mm_positions"].clone()
            data["mm_charges"] = (
                data["mm_charges"]
                .clone()
                .requires_grad_(compute_response or compute_charges)
            )
            if "mm_atomic_numbers" not in data:
                if len(data["mm_positions"]):
                    raise ValueError(
                        "v2 requires mm_atomic_numbers for environment nodes"
                    )
                data["mm_atomic_numbers"] = torch.empty(
                    0, dtype=torch.long, device=data["positions"].device
                )
            if len(data["mm_atomic_numbers"]) != len(data["mm_positions"]):
                raise ValueError("Environment species and position counts differ")
            graph = build_graph(data, float(self.r_max), self.config)
            n = graph.qm_count
            keep = (graph.edge_index < n).all(0)
            native = {k: v for k, v in data.items() if not k.startswith("mm_")}
            native["atomic_numbers"] = self.atomic_numbers[
                data["node_attrs"].argmax(-1)
            ]
            native["edge_index"] = graph.edge_index[:, keep]
            native["shifts"] = data["positions"].new_zeros(int(keep.sum()), 3)
            native["unit_shifts"] = torch.zeros_like(native["shifts"])
            ids = graph.env_indices
            scalar = self.env_embedding(
                data["mm_atomic_numbers"][ids]
            ) + self.env_charge_embedding(data["mm_charges"][ids])
            env = torch.cat(
                (scalar, scalar.new_zeros(len(ids), self.hidden.dim - scalar.shape[1])),
                -1,
            )
            products, last, handles = [], [], []
            sizes = data["ptr"][1:] - data["ptr"][:-1]
            local_q = (data["total_charge"].reshape(-1) / sizes)[data["batch"], None]

            def interaction_hook(index):
                def hook(module, args, kwargs, output):
                    nonlocal env
                    incoming = kwargs["node_feats"]
                    message, env = self.env_layers[index](
                        incoming, env, graph, data["mm_charges"]
                    )
                    qmessage = self.total_charge_embedding[index](local_q)
                    padding = torch.nn.functional.pad(
                        qmessage[:, :, None], (0, output[0].shape[-1] - 1)
                    )
                    updated = output[0] + self.config.epsilon * message + padding
                    last[:] = [flatten_message(updated, module.target_irreps)]
                    return updated, output[1]

                return hook

            try:
                for i, interaction in enumerate(self.foundation.interactions):
                    handles.append(
                        interaction.register_forward_hook(
                            interaction_hook(i), with_kwargs=True
                        )
                    )
                for product in self.foundation.products:
                    handles.append(
                        product.register_forward_hook(
                            lambda module, args, output: products.append(output)
                        )
                    )
                base = self.foundation(native, training=training, compute_force=False)
            finally:
                for handle in handles:
                    handle.remove()
            h = (
                self.last_mixer(last[0])
                if self.config.feature_source == "last_message"
                else sum(m(x) for m, x in zip(self.mixers, products))
            )
            f = self.global_encoder(data)
            # Conditional residual vanishes at zero global descriptor.
            correction = (
                self.readout(self._invariants(h, f))
                - self.readout(self._invariants(h, torch.zeros_like(f)))
            ).squeeze(-1)
            energy = base["energy"] + sum_nodes(correction, data["batch"], len(sizes))
            result = {"energy": energy}
            if compute_force:
                force = torch.autograd.grad(
                    energy.sum(),
                    data["positions"],
                    create_graph=training,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                result["forces"] = (
                    -force if force is not None else data["positions"] * 0
                )
            if compute_response or compute_charges:
                result.update(
                    response_outputs(
                        energy, data, self.config, training, compute_charges
                    )
                )
            return result
