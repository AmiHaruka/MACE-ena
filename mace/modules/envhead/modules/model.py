"""v3: shared vacuum/embedded features, reconstructed QM charges, explicit qV."""

from copy import deepcopy
import torch
from e3nn import o3
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.extensions import MACELES
from .config import EnvHeadConfig
from .graph import build_graph, qm_edges
from .encoders import EnvironmentInteraction, flatten_message, sum_nodes
from .electrostatics import potentials
from .charges import ChargeReconstruction
from .readout import ResponseReadout
from .execution import VacuumTrace, run_shared_passes


class _FeaturesReady(Exception):
    """End a native feature-only pass before final readouts/LES electrostatics."""


class EnvMACE(torch.nn.Module):
    format_version = 3

    def __init__(
        self,
        foundation,
        config=None,
        dtype=None,
        _build_response_head=True,
        _charge_class=ChargeReconstruction,
    ):
        super().__init__()
        self.config = config or EnvHeadConfig()
        if type(foundation) not in (MACE, ScaleShiftMACE, MACELES):
            raise TypeError("EnvHead supports native MACE, ScaleShiftMACE and MACELES")
        if len(getattr(foundation, "heads", ["Default"])) != 1:
            raise ValueError("EnvHead requires one foundation energy head")
        self.foundation = deepcopy(foundation)
        dtype = dtype or next(foundation.parameters()).dtype
        self.foundation.to(dtype=dtype)
        self.foundation.requires_grad_(self.config.finetune == "full")
        if self.config.finetune == "partial":
            self.foundation.interactions[-1].requires_grad_(True)
            self.foundation.products[-1].requires_grad_(True)
            self.foundation.readouts.requires_grad_(True)
        self.foundation.atomic_energies_fn.requires_grad_(False)
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
        self.far_layers = torch.nn.ModuleList(
            [o3.Linear("0e+1o", i.target_irreps) for i in self.foundation.interactions]
        )
        # New adapters should perturb, rather than overwhelm, pretrained ACE
        # messages. Nonzero small initialization preserves first-step gradients.
        with torch.no_grad():
            for layer in self.env_layers:
                layer.output.weight.mul_(0.01)
            for layer in self.far_layers:
                layer.weight.mul_(0.01)
            for layer in self.total_charge_embedding:
                layer.weight.mul_(0.01)
        # Atomic-unit features for learned paths; explicit qV stays in eV.
        self.register_buffer(
            "field_units",
            torch.tensor(
                [27.211386245988, 51.4220674763, 51.4220674763, 51.4220674763]
            ),
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
        self.charge_reconstruction = _charge_class(self.hidden, self.config)
        if _build_response_head:
            self.response_readout = ResponseReadout(self.hidden, self.config)
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

    def _pass(
        self,
        native,
        data,
        graph,
        far,
        embedded,
        training,
        capture_common=False,
        complete_embedded_energy=False,
        environment_geometry=None,
    ):
        """Scoped native hooks preserve each supported foundation's preprocessing.

        The v3 feature-only pass stops at the final product. V4 explicitly
        continues to the native energy assembly. Hook handles are removed on
        both success and exceptions.
        Concurrent forwards require separate model instances.
        """
        if embedded:
            ids = graph.env_indices
            scalar = self.env_embedding(
                data["mm_atomic_numbers"][ids]
            ) + self.env_charge_embedding(data["mm_charges"][ids].reshape(-1, 1))
            env = torch.nn.functional.pad(
                scalar, (0, self.hidden.dim - scalar.shape[1])
            )
        else:
            env = None
        sizes = data["ptr"][1:] - data["ptr"][:-1]
        local_q = (data["total_charge"].reshape(-1) / sizes)[data["batch"], None]
        products, last, handles = [], [], []
        trace = VacuumTrace() if capture_common else None

        def interaction_hook(index):
            def hook(module, args, kwargs, output):
                nonlocal env
                padding = torch.nn.functional.pad(
                    self.total_charge_embedding[index](local_q)[:, :, None],
                    (0, output[0].shape[-1] - 1),
                )
                updated = output[0] + padding
                if trace is not None:
                    trace.charge_terms.append(padding)
                    if index == 0:
                        trace.interaction_kwargs = dict(kwargs)
                        trace.first_message, trace.first_sc = output
                        trace.first_conditioned_message = updated
                if embedded:
                    environment_kwargs = {}
                    if environment_geometry is not None:
                        environment_kwargs["geometry"] = environment_geometry
                    message, env = self.env_layers[index](
                        kwargs["node_feats"],
                        env,
                        graph,
                        data["mm_charges"],
                        update_env=index < len(self.env_layers) - 1,
                        **environment_kwargs,
                    )
                    field = self.env_layers[index].output_layout(
                        self.far_layers[index](far)
                    )
                    updated = updated + self.config.epsilon * message + field
                last[:] = [flatten_message(updated, module.target_irreps)]
                return updated, output[1]

            return hook

        def product_hook(index):
            def hook(module, args, output):
                products.append(output)
                if (
                    embedded
                    and not complete_embedded_energy
                    and index == len(self.foundation.products) - 1
                ):
                    raise _FeaturesReady()

            return hook

        base = None
        try:
            for i, interaction in enumerate(self.foundation.interactions):
                handles.append(
                    interaction.register_forward_hook(
                        interaction_hook(i), with_kwargs=True
                    )
                )
            for i, product in enumerate(self.foundation.products):
                handles.append(product.register_forward_hook(product_hook(i)))
            try:
                base = self.foundation(
                    dict(native), training=training, compute_force=False
                )
            except _FeaturesReady:
                if not embedded:
                    raise
        finally:
            for handle in handles:
                handle.remove()
        h = (
            self.last_mixer(last[0])
            if self.config.feature_source == "last_message"
            else sum(m(x) for m, x in zip(self.mixers, products))
        )
        return (base, h, trace) if capture_common else (base, h)

    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_response=False,
        compute_charges=False,
        compute_vacuum_charges=False,
        execution_mode="shared",
    ):
        if execution_mode not in ("legacy", "shared"):
            raise ValueError("execution_mode must be legacy or shared")
        with torch.enable_grad():
            data = dict(data)
            if data.get("pbc") is not None and data["pbc"].any():
                raise ValueError("EnvHead v3 requires nonperiodic data")
            if data.get("total_spin") is not None and (data["total_spin"] != 1).any():
                raise ValueError("EnvHead v3 currently supports singlet data")
            data["positions"] = data["positions"].clone().requires_grad_(True)
            data["mm_positions"] = data["mm_positions"].clone()
            data["mm_charges"] = data["mm_charges"].clone().reshape(-1, 1)
            if len(data["mm_charges"]) != len(data["mm_positions"]):
                raise ValueError("Background charge and position counts differ")
            if compute_response:
                data["mm_charges"].requires_grad_(True)
            if "mm_atomic_numbers" not in data:
                if len(data["mm_positions"]):
                    raise ValueError("ENV elements are required")
                data["mm_atomic_numbers"] = torch.empty(
                    0, dtype=torch.long, device=data["positions"].device
                )
            if len(data["mm_atomic_numbers"]) != len(data["mm_positions"]):
                raise ValueError("ENV species and position counts differ")
            data["atomic_numbers"] = self.atomic_numbers[data["node_attrs"].argmax(-1)]
            native = {k: v for k, v in data.items() if not k.startswith("mm_")}
            # Old charge routes keep their zero/None gradient behavior during
            # training, including optimizer state loaded from old checkpoints.
            vacuum_only = (
                execution_mode == "shared"
                and len(data["mm_positions"]) == 0
                and (not training or self.config.charge_mode == "fukui_projection")
            )
            if vacuum_only:
                native["edge_index"] = qm_edges(
                    data, float(self.r_max), self.config.near_radius
                )
                native["shifts"] = data["positions"].new_zeros(
                    native["edge_index"].shape[1], 3
                )
                native["unit_shifts"] = torch.zeros_like(native["shifts"])
                base, h0 = self._pass(native, data, None, None, False, training)
                reconstructed = self.charge_reconstruction(
                    h0, data, diagnostics=compute_charges
                )
                q0 = reconstructed["charges"]
                electrostatic = torch.zeros_like(base["energy"])
                residual = torch.zeros_like(base["energy"])
            else:
                graph = build_graph(data, float(self.r_max), self.config)
                keep = (graph.edge_index < graph.qm_count).all(0)
                native["edge_index"] = graph.edge_index[:, keep]
                native["shifts"] = data["positions"].new_zeros(int(keep.sum()), 3)
                native["unit_shifts"] = torch.zeros_like(native["shifts"])
                phi, far = potentials(data, self.config)
                far = far / self.field_units
                if execution_mode == "shared":
                    base, h0, he = run_shared_passes(
                        self, native, data, graph, far, training
                    )
                else:
                    base, h0 = self._pass(native, data, graph, far, False, training)
                    _, he = self._pass(native, data, graph, far, True, training)
                charge_geometry = (
                    self.charge_reconstruction.prepare(data)
                    if self.config.charge_mode
                    in ("charge_projection", "iterative_response")
                    else None
                )
                q0 = self.charge_reconstruction(h0, data, geometry=charge_geometry)[
                    "charges"
                ]
                reconstructed = self.charge_reconstruction(
                    he,
                    data,
                    diagnostics=compute_charges,
                    potential=phi,
                    geometry=charge_geometry,
                )
                q = reconstructed["charges"]
                n = len(data["ptr"]) - 1
                electrostatic = sum_nodes(q * phi, data["batch"], n)
                local_residual = self.response_readout(
                    he, q, far
                ) - self.response_readout(h0, q0, torch.zeros_like(far))
                residual = sum_nodes(local_residual, data["batch"], n)
            energy = base["energy"] + electrostatic + residual
            result = dict(
                energy=energy,
                energy_vacuum=base["energy"],
                energy_electrostatic=electrostatic,
                energy_residual=residual,
                **reconstructed,
            )
            if compute_vacuum_charges:
                result["charges_vacuum"] = q0
            if compute_force:
                result["forces"] = -torch.autograd.grad(
                    energy.sum(),
                    data["positions"],
                    create_graph=training,
                    retain_graph=True,
                )[0]
            if compute_response:
                v = (
                    data["mm_charges"].new_empty(0)
                    if vacuum_only
                    else torch.autograd.grad(
                        energy.sum(),
                        data["mm_charges"],
                        create_graph=training,
                        retain_graph=True,
                    )[0]
                )
                result["response_potential"] = v.flatten()
            return result
