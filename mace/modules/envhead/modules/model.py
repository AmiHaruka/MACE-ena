"""One conditioned foundation pass plus full-source QM--ENV monopole energy."""

from copy import deepcopy
import torch
from e3nn import o3
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.extensions import MACELES

from .config import EnvHeadConfig
from .charges import ChargeReconstruction
from .encoders import (
    EnvironmentGate,
    EnvironmentInteraction,
    OnceEnvironmentEncoder,
    sum_nodes,
    flatten_message,
)
from .electrostatics import grouped_potentials, potentials
from .graph import build_graph, prepare_once_edges, qm_edges


class EnvMACE(torch.nn.Module):
    format_version = 4

    def __init__(self, foundation, config=None, dtype=None):
        super().__init__()
        self.config = config or EnvHeadConfig()
        if type(foundation) not in (MACE, ScaleShiftMACE, MACELES):
            raise TypeError("EnvHead supports native MACE, ScaleShiftMACE and MACELES")
        if len(getattr(foundation, "heads", ["Default"])) != 1:
            raise ValueError("EnvHead requires one foundation energy head")
        self.foundation = deepcopy(foundation)
        dtype = dtype or next(foundation.parameters()).dtype
        self.foundation.to(dtype=dtype)
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
        environment_class = (
            EnvironmentGate
            if self.config.environment_mode == "once"
            else EnvironmentInteraction
        )
        self.env_layers = torch.nn.ModuleList(
            [
                environment_class(
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
        # Small nonzero adapter messages preserve first-step gradients.
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
        self.charge_reconstruction = ChargeReconstruction(self.config)
        self.to(dtype=dtype, device=next(foundation.parameters()).device)
        if self.config.cueq:
            from ..backends import enable_cueq

            enable_cueq(self)
        config = self.config
        if config.environment_mode == "once":
            self.env_encoder = OnceEnvironmentEncoder(
                self.hidden, float(self.r_max), config
            ).to(next(self.parameters()))
            # Initialize known ENV elements from the foundation's chemical
            # embedding; independent rows can then adapt to MM environments.
            with torch.no_grad():
                features = self.foundation.node_embedding(
                    torch.eye(
                        len(self.atomic_numbers),
                        device=self.atomic_numbers.device,
                        dtype=next(self.parameters()).dtype,
                    )
                )
                projection = torch.nn.Linear(
                    features.shape[1], config.channels, bias=False
                ).to(features)
                self.env_embedding.weight[self.atomic_numbers] = projection(features)
        # Conversion creates new contraction parameters. Select trainability
        # after conversion, before the training entry point creates Adam.
        self.foundation.requires_grad_(self.config.finetune == "full")
        if self.config.finetune == "partial":
            self.foundation.interactions[-1].requires_grad_(True)
            self.foundation.products[-1].requires_grad_(True)
            self.foundation.readouts.requires_grad_(True)
        self.foundation.atomic_energies_fn.requires_grad_(False)

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
        self, native, data, graph, far, embedded, training, environment_geometry=None,
        *, read_features=True,
    ):
        """Condition each native interaction, then retain native energy assembly.

        Hooks are scoped to this call, including exceptional exits. Concurrent
        forwards require separate model instances.
        """
        if embedded and not hasattr(self, "env_encoder"):
            ids = graph.env_indices
            scalar = self.env_embedding(data["mm_atomic_numbers"][ids])
            scalar = scalar + self.env_charge_embedding(
                data["mm_charges"][ids].reshape(-1, 1)
            )
            env = torch.nn.functional.pad(
                scalar, (0, self.hidden.dim - scalar.shape[1])
            )
        else:
            env = None
        sizes = data["ptr"][1:] - data["ptr"][:-1]
        local_q = (data["total_charge"].reshape(-1) / sizes)[data["batch"], None]
        products, handles = [], []
        last = None

        def interaction_hook(index):
            def hook(module, args, kwargs, output):
                nonlocal env, last
                padding = torch.nn.functional.pad(
                    self.total_charge_embedding[index](local_q)[:, :, None],
                    (0, output[0].shape[-1] - 1),
                )
                updated = output[0] + padding
                if embedded:
                    message, env = self.env_layers[index](
                        kwargs["node_feats"],
                        env,
                        graph,
                        data["mm_charges"],
                        update_env=index < len(self.env_layers) - 1,
                        geometry=environment_geometry,
                    )
                    field = self.env_layers[index].output_layout(
                        self.far_layers[index](far)
                    )
                    updated = updated + self.config.epsilon * message + field
                if (
                    read_features
                    and self.config.feature_source == "last_message"
                    and index == len(self.env_layers) - 1
                ):
                    last = flatten_message(updated, module.target_irreps)
                return updated, output[1]

            return hook

        try:
            for i, interaction in enumerate(self.foundation.interactions):
                handles.append(
                    interaction.register_forward_hook(
                        interaction_hook(i), with_kwargs=True
                    )
                )
            if read_features and self.config.feature_source == "mixed_products":
                for product in self.foundation.products:
                    handles.append(
                        product.register_forward_hook(
                            lambda module, args, output: products.append(output)
                        )
                    )
            base = self.foundation(dict(native), training=training, compute_force=False)
        finally:
            for handle in handles:
                handle.remove()
        if not read_features:
            return base, None
        features = (
            self.last_mixer(last)
            if self.config.feature_source == "last_message"
            else sum(m(x) for m, x in zip(self.mixers, products))
        )
        return base, features

    def _energy_forward(
        self, data, training=False, compute_charges=False,
        execution_mode="optimized", neighbor_backend="reference",
        *, prepared_environment=None,
    ):
        """One energy graph, without cloning coordinates or taking derivatives.

        A prepared environment is (graph, potential, [potential, Ex, Ey, Ez]),
        with potential in eV/e and field in eV/(e Angstrom).
        Its geometry and fields must share the caller's
        coordinate graph. Periodic engine adapters differentiate the returned
        scalar, including their corrections, outside this method.
        """
        data = dict(data)
        data["atomic_numbers"] = self.atomic_numbers[data["node_attrs"].argmax(-1)]
        native = {key: value for key, value in data.items()
                  if not key.startswith("mm_")}
        embedded = len(data["mm_positions"]) > 0
        if prepared_environment is not None and not embedded:
            raise ValueError("A prepared environment requires background sites")
        if embedded:
            if prepared_environment is None:
                graph = build_graph(
                    data, float(self.r_max), self.config,
                    neighbor_backend=neighbor_backend, owner=self,
                    role_geometry=(execution_mode == "optimized"
                                   and hasattr(self, "env_encoder")),
                    unchunked=execution_mode == "optimized",
                )
                operation = grouped_potentials if execution_mode == "optimized" else potentials
                phi, far = operation(data, self.config)
            else:
                graph, phi, far = prepared_environment
            native["edge_index"] = graph.edge_index[
                :, (graph.edge_index < graph.qm_count).all(0)
            ]
            far = far / self.field_units
            if hasattr(self, "env_encoder"):
                ids = graph.env_indices
                scalar = self.env_embedding(data["mm_atomic_numbers"][ids])
                scalar = scalar + self.env_charge_embedding(data["mm_charges"][ids])
                geometry = self.env_encoder(
                    graph, scalar, data["mm_charges"][ids].flatten(),
                    edge_geometry=(prepare_once_edges(graph, data["mm_positions"])
                                   if execution_mode == "optimized" else None),
                )
            else:
                geometry = self.env_layers[0].prepare_geometry(graph)
        else:
            graph, phi, far, geometry = None, None, None, None
            native["edge_index"] = qm_edges(
                data, float(self.r_max), self.config.near_radius,
                neighbor_backend=neighbor_backend, owner=self,
            )
        native["shifts"] = data["positions"].new_zeros(native["edge_index"].shape[1], 3)
        native["unit_shifts"] = torch.zeros_like(native["shifts"])
        base, features = self._pass(
            native, data, graph, far, embedded, training,
            environment_geometry=geometry,
        )
        reconstructed = self.charge_reconstruction(features, data, diagnostics=compute_charges)
        ele = (sum_nodes(reconstructed["charges"] * phi, data["batch"], len(data["ptr"])-1)
               if embedded else torch.zeros_like(base["energy"]))
        return dict(energy=base["energy"] + ele, energy_ele=ele, **reconstructed), base

    def _forward_once(
        self,
        data,
        training=False,
        compute_force=True,
        compute_response=False,
        compute_charges=False,
        execution_mode="optimized",
        neighbor_backend="reference",
        *,
        compute_mm_force=False,
    ):
        with torch.enable_grad():
            data = dict(data)
            if data.get("pbc") is not None and data["pbc"].any():
                raise ValueError("EnvHead requires nonperiodic data")
            if data.get("total_spin") is not None and (data["total_spin"] != 1).any():
                raise ValueError("EnvHead supports singlet data")
            data["positions"] = data["positions"].clone().requires_grad_(True)
            data["mm_positions"] = data["mm_positions"].clone()
            if compute_mm_force:
                data["mm_positions"].requires_grad_(True)
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
            embedded = len(data["mm_positions"]) > 0
            result, base = self._energy_forward(
                data, training=training, compute_charges=compute_charges,
                execution_mode=execution_mode, neighbor_backend=neighbor_backend,
            )
            energy = result["energy"]
            if compute_mm_force and embedded:
                coordinates, names = [data["mm_positions"]], ["mm_forces"]
                if compute_force:
                    coordinates.insert(0, data["positions"])
                    names.insert(0, "forces")
                derivatives = torch.autograd.grad(
                    energy.sum(),
                    coordinates,
                    create_graph=training,
                    retain_graph=True,
                )
                result.update({name: -g for name, g in zip(names, derivatives)})
            elif compute_force:
                result["forces"] = -torch.autograd.grad(
                    energy.sum(),
                    data["positions"],
                    create_graph=training,
                    retain_graph=True,
                )[0]
            if compute_mm_force and not embedded:
                result["mm_forces"] = data["mm_positions"].new_empty((0, 3))
            if compute_response:
                result["response_potential"] = (
                    torch.autograd.grad(
                        energy.sum(),
                        data["mm_charges"],
                        create_graph=training,
                        retain_graph=True,
                    )[0].flatten()
                    if embedded
                    else data["mm_charges"].new_empty(0)
                )
            return result, base

    @staticmethod
    def _vacuum_data(data):
        vacuum = dict(data)
        vacuum["mm_positions"] = data["mm_positions"].new_empty((0, 3))
        vacuum["mm_charges"] = data["mm_charges"].new_empty((0, 1))
        vacuum["mm_atomic_numbers"] = torch.empty(
            0, device=data["positions"].device, dtype=torch.long
        )
        vacuum["mm_count"] = torch.zeros_like(data["mm_count"])
        return vacuum

    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_response=False,
        compute_charges=False,
        compute_vacuum_charges=False,
        execution_mode="optimized",
        neighbor_backend="reference",
        *,
        compute_mm_force=False,
    ):
        if execution_mode not in ("reference", "optimized"):
            raise ValueError("execution_mode must be reference or optimized")
        result, _ = self._forward_once(
            data,
            training,
            compute_force,
            compute_response,
            compute_charges,
            execution_mode=execution_mode,
            neighbor_backend=neighbor_backend,
            compute_mm_force=compute_mm_force,
        )
        if compute_vacuum_charges:
            result["charges_vacuum"] = (
                self._forward_once(
                    self._vacuum_data(data),
                    training,
                    compute_force=False,
                    execution_mode=execution_mode,
                    neighbor_backend=neighbor_backend,
                )[0]["charges"]
                if len(data["mm_positions"])
                else result["charges"]
            )
        return result

    def compare_environment(
        self,
        data,
        compute_charges=True,
        execution_mode="optimized",
        neighbor_backend="reference",
    ):
        """Two ordinary conditions, same QM geometry/Q/parameters; no force split.

        The response includes energy_ele. Subtract reference-free native terms
        in double precision, before adding the explicit cross interaction.
        """
        if execution_mode not in ("reference", "optimized"):
            raise ValueError("execution_mode must be reference or optimized")
        embedded, native_embedded = self._forward_once(
            data,
            compute_force=False,
            compute_charges=compute_charges,
            execution_mode=execution_mode,
            neighbor_backend=neighbor_backend,
        )
        if len(data["mm_positions"]):
            vacuum, native_vacuum = self._forward_once(
                self._vacuum_data(data),
                compute_force=False,
                compute_charges=compute_charges,
                execution_mode=execution_mode,
                neighbor_backend=neighbor_backend,
            )
            response = (
                native_embedded["interaction_energy"].double()
                - native_vacuum["interaction_energy"].double()
                + embedded["energy_ele"].double()
            )
        else:
            vacuum = embedded
            response = embedded["energy"].new_zeros(
                embedded["energy"].shape, dtype=torch.float64
            )
        return dict(
            **embedded,
            energy_vac=vacuum["energy"],
            energy_response=response,
            charges_vacuum=vacuum["charges"],
            delta_charges=embedded["charges"] - vacuum["charges"],
        )
