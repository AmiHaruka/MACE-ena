"""One conditioned foundation pass plus full-source QM--ENV monopole energy."""

import torch

from .config_v4 import EnvHeadConfigV4
from .charges import ChargeReconstructionV4
from .encoders import sum_nodes
from .electrostatics import potentials
from .graph import build_graph, qm_edges
from .model import EnvMACE


class EnvMACEV4(EnvMACE):
    format_version = 4

    def __init__(self, foundation, config=None, dtype=None):
        super().__init__(
            foundation,
            config or EnvHeadConfigV4(),
            dtype,
            _build_response_head=False,
            _charge_class=ChargeReconstructionV4,
        )
        # Conversion creates new contraction parameters. Select trainability
        # after conversion, before the training entry point creates Adam.
        self.foundation.requires_grad_(self.config.finetune == "full")
        if self.config.finetune == "partial":
            self.foundation.interactions[-1].requires_grad_(True)
            self.foundation.products[-1].requires_grad_(True)
            self.foundation.readouts.requires_grad_(True)
        self.foundation.atomic_energies_fn.requires_grad_(False)

    def _environment_geometry(self, graph):
        return self.env_layers[0].prepare_geometry(graph)

    def _forward_once(
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
                raise ValueError("EnvHead v4 requires nonperiodic data")
            if data.get("total_spin") is not None and (data["total_spin"] != 1).any():
                raise ValueError("EnvHead v4 currently supports singlet data")
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
            native = {
                key: value for key, value in data.items() if not key.startswith("mm_")
            }
            embedded = len(data["mm_positions"]) > 0
            if embedded:
                graph = build_graph(data, float(self.r_max), self.config)
                native["edge_index"] = graph.edge_index[
                    :, (graph.edge_index < graph.qm_count).all(0)
                ]
                phi, far = potentials(data, self.config)
                far = far / self.field_units
                geometry = self._environment_geometry(graph)
            else:
                graph, phi, far = None, None, None
                geometry = None
                native["edge_index"] = qm_edges(
                    data, float(self.r_max), self.config.near_radius
                )
            native["shifts"] = data["positions"].new_zeros(
                native["edge_index"].shape[1], 3
            )
            native["unit_shifts"] = torch.zeros_like(native["shifts"])
            base, features = self._pass(
                native,
                data,
                graph,
                far,
                embedded,
                training,
                complete_embedded_energy=True,
                environment_geometry=geometry,
            )
            reconstructed = self.charge_reconstruction(
                features, data, diagnostics=compute_charges
            )
            ele = (
                sum_nodes(
                    reconstructed["charges"] * phi, data["batch"], len(data["ptr"]) - 1
                )
                if embedded
                else torch.zeros_like(base["energy"])
            )
            energy = base["energy"] + ele
            result = dict(energy=energy, energy_ele=ele, **reconstructed)
            if compute_force:
                result["forces"] = -torch.autograd.grad(
                    energy.sum(),
                    data["positions"],
                    create_graph=training,
                    retain_graph=True,
                )[0]
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
    ):
        result, _ = self._forward_once(
            data, training, compute_force, compute_response, compute_charges
        )
        if compute_vacuum_charges:
            result["charges_vacuum"] = (
                self._forward_once(
                    self._vacuum_data(data), training, compute_force=False
                )[0]["charges"]
                if len(data["mm_positions"])
                else result["charges"]
            )
        return result

    def compare_environment(self, data, compute_charges=True):
        """Two ordinary conditions, same QM geometry/Q/parameters; no force split.

        The response includes energy_ele. Subtract reference-free native terms
        in double precision, before adding the explicit cross interaction.
        """
        embedded, native_embedded = self._forward_once(
            data, compute_force=False, compute_charges=compute_charges
        )
        if len(data["mm_positions"]):
            vacuum, native_vacuum = self._forward_once(
                self._vacuum_data(data),
                compute_force=False,
                compute_charges=compute_charges,
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
