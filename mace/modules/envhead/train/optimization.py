"""Named Adam groups and EMA weights for EnvHead training."""

from argparse import Namespace

import torch
from torch_ema import ExponentialMovingAverage

from mace.tools.scripts_utils import get_params_options


def build_optimizer(
    model,
    *,
    lr=0.01,
    weight_decay=5e-7,
    foundation_lr_factor=1.0,
    environment_lr_factor=1.0,
    amsgrad=True,
):
    """Preserve native MACE decay rules and the model's trainability mask.

    Parameter names are stored with each group so checkpoints and hypothetical
    updates can identify tensors independently of the optimizer group order.
    """
    named = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in named.items()}
    groups = []
    claimed = set()

    def append(name, parameters, decay, group_lr):
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        if not parameters:
            return
        for parameter in parameters:
            ident = id(parameter)
            if ident in claimed:
                raise ValueError(f"Duplicate optimizer parameter: {names[ident]}")
            claimed.add(ident)
        groups.append(
            dict(
                name=name,
                params=parameters,
                param_names=[names[id(parameter)] for parameter in parameters],
                weight_decay=decay,
                lr=group_lr,
            )
        )

    # freeze=None prevents the native helper from changing partial/frozen masks.
    # Recursive native product groups include converted cuEq contractions.
    options = get_params_options(
        Namespace(
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            beta=0.9,
            freeze=None,
            lr_params_factors="{}",
            train_one_body_contribution=True,
        ),
        model.foundation,
    )
    for group in options["params"]:
        append(
            "foundation_" + group["name"],
            group["params"],
            group["weight_decay"],
            group.get("lr", lr) * foundation_lr_factor,
        )

    for module_name in ("env_encoder", "env_layers"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        module_named = list(module.named_parameters())
        for decay in (True, False):
            append(
                module_name + ("_decay" if decay else "_no_decay"),
                [
                    parameter
                    for name, parameter in module_named
                    if (name.rsplit(".", 1)[-1] not in ("bias", "biases")) == decay
                ],
                weight_decay if decay else 0.0,
                lr * environment_lr_factor,
            )
    for module_name in (
        "env_embedding",
        "env_charge_embedding",
        "total_charge_embedding",
        "far_layers",
        "mixers",
        "last_mixer",
        "charge_reconstruction",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            append(module_name, module.parameters(), 0.0, lr * environment_lr_factor)

    missing = [
        name
        for name, parameter in named.items()
        if parameter.requires_grad and id(parameter) not in claimed
    ]
    if missing:
        raise ValueError(
            "Unregistered trainable optimizer parameters: " + ", ".join(missing)
        )
    return torch.optim.Adam(groups, lr=lr, amsgrad=amsgrad, betas=(0.9, 0.999))


def make_ema(model, decay=0.99):
    """Average trainable parameters in stable model order without altering it."""
    return ExponentialMovingAverage(
        [
            parameter
            for _, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        decay=decay,
    )
