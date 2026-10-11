"""Per-call differentiable reuse; native vacuum energy assembly stays intact."""

from dataclasses import dataclass, field
import torch
from .encoders import flatten_message


@dataclass
class VacuumTrace:
    """Local tensors captured before the vacuum and embedded products diverge."""

    interaction_kwargs: dict = field(default_factory=dict)
    first_message: torch.Tensor | None = None
    first_sc: torch.Tensor | None = None
    first_conditioned_message: torch.Tensor | None = None
    charge_terms: list = field(default_factory=list)


def run_shared_passes(model, native, data, graph, far, training):
    base, vacuum_features, trace = model._pass(
        native, data, graph, far, False, training, capture_common=True
    )
    ids = graph.env_indices
    scalar = model.env_embedding(
        data["mm_atomic_numbers"][ids]
    ) + model.env_charge_embedding(data["mm_charges"][ids].reshape(-1, 1))
    env = torch.nn.functional.pad(scalar, (0, model.hidden.dim - scalar.shape[1]))
    qm = trace.interaction_kwargs["node_feats"]
    products = []
    last = None
    for index, (interaction, product) in enumerate(
        zip(model.foundation.interactions, model.foundation.products)
    ):
        if index == 0:
            updated, sc = trace.first_conditioned_message, trace.first_sc
        else:
            kwargs = dict(trace.interaction_kwargs)
            kwargs["node_feats"] = qm
            kwargs["first_layer"] = False
            message, sc = interaction(**kwargs)
            updated = message + trace.charge_terms[index]
        message, env = model.env_layers[index](
            qm,
            env,
            graph,
            data["mm_charges"],
            update_env=index < len(model.env_layers) - 1,
        )
        field_message = model.env_layers[index].output_layout(
            model.far_layers[index](far)
        )
        updated = updated + model.config.epsilon * message + field_message
        last = flatten_message(updated, interaction.target_irreps)
        qm = product(node_feats=updated, sc=sc, node_attrs=native["node_attrs"])
        products.append(qm)
    features = (
        model.last_mixer(last)
        if model.config.feature_source == "last_message"
        else sum(m(x) for m, x in zip(model.mixers, products))
    )
    return base, vacuum_features, features
