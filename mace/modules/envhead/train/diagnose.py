"""Read-only E/F/q gradient and hypothetical Adam-update diagnostics."""

import argparse
import json
import math
from pathlib import Path
import torch
from mace.tools.torch_geometric.batch import Batch
from .checkpoint import load_checkpoint
from .training import EnvHeadLoss
from ..data.loading import load_dataset


def cosine(a, b):
    denom = a.norm() * b.norm()
    return float(torch.dot(a, b) / denom) if denom > 0 else None


def parameter_group(name):
    if name.startswith("foundation.readouts"):
        return "foundation_readouts"
    if name.startswith("foundation."):
        return "foundation_interactions_products"
    if name.startswith("charge_reconstruction"):
        return "charge_head"
    return "environment_and_mixers"


def adam_update(
    parameters,
    gradient,
    used,
    state_dict,
    scale=1.0,
    fresh=False,
    *,
    parameter_names=None,
):
    """Compute, without applying, an Adam step. Saved moments stay unchanged."""
    parameters = list(parameters)
    sizes = [parameter.numel() for parameter in parameters]
    if len(used) != len(parameters) or gradient.numel() != sum(sizes):
        raise ValueError("Gradient and usage mask must match the model parameters")
    slices = []
    offset = 0
    for size in sizes:
        slices.append(slice(offset, offset + size))
        offset += size
    groups = state_dict["param_groups"]
    named = ["param_names" in group for group in groups]
    if any(named):
        if not all(named) or parameter_names is None:
            raise ValueError(
                "Named optimizer groups require matching model parameter_names"
            )
        if len(parameter_names) != len(parameters) or len(set(parameter_names)) != len(
            parameters
        ):
            raise ValueError("Model parameter_names must identify each parameter once")
        indices = {name: index for index, name in enumerate(parameter_names)}
        ordered = []
        for group in groups:
            if len(group["param_names"]) != len(group["params"]):
                raise ValueError("Optimizer param_names must match each group's params")
            for ident, name in zip(group["params"], group["param_names"]):
                if name not in indices:
                    raise ValueError(
                        f"Optimizer parameter is absent from model: {name}"
                    )
                ordered.append((group, ident, indices[name]))
    else:
        if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
            raise ValueError(
                "Cannot map unnamed optimizer groups; saved param_names are required"
            )
        ordered = [
            (groups[0], ident, index) for index, ident in enumerate(groups[0]["params"])
        ]
    if len(ordered) != len(parameters) or len(
        {index for _, _, index in ordered}
    ) != len(parameters):
        raise ValueError(
            "Optimizer groups must cover each model parameter exactly once"
        )
    if len({ident for _, ident, _ in ordered}) != len(ordered):
        raise ValueError("Optimizer parameter identifiers must be unique")

    gradient = gradient.detach().cpu().double().flatten()
    result = [None] * len(parameters)
    for group, ident, index in ordered:
        b1, b2 = group["betas"]
        p = parameters[index]
        g = gradient[slices[index]] * scale
        if not used[index]:
            result[index] = torch.zeros_like(g)
            continue
        if group.get("maximize", False):
            g = -g
        if group.get("weight_decay", 0):
            g = g + group["weight_decay"] * p.detach().cpu().double().flatten()
        state = {} if fresh else state_dict["state"].get(ident, {})

        def moment(key):
            if key not in state:
                return torch.zeros_like(g)
            if state[key].shape != p.shape:
                raise ValueError(
                    f"Optimizer moment {key} shape does not match its model parameter"
                )
            return state[key].detach().cpu().double().flatten()

        step = float(state.get("step", 0)) + 1
        m = b1 * moment("exp_avg") + (1 - b1) * g
        v = b2 * moment("exp_avg_sq") + (1 - b2) * g.square()
        if group.get("amsgrad", False):
            v = torch.maximum(v, moment("max_exp_avg_sq"))
        result[index] = (
            -group["lr"]
            * (m / (1 - b1**step))
            / ((v / (1 - b2**step)).sqrt() + group["eps"])
        )
    return torch.cat(result)


def diagnose(model, batches, criterion, optimizer_state):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    sizes = [p.numel() for p in params]
    terms = ("energy_loss", "forces_loss", "charges_loss")
    vectors = {k: torch.zeros(sum(sizes), dtype=torch.float64) for k in terms}
    counts = {k: 0 for k in terms}
    numerators = {k: 0.0 for k in terms}
    used = [False] * len(params)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            batch = batch.to(device)
            pred = model(batch.to_dict(), training=True)
            batch_loss, parts, errors = criterion(batch, pred, return_components=True)
            denominators = dict(
                energy_loss=batch.energy.numel(),
                forces_loss=batch.forces.numel(),
                charges_loss=errors.numel(),
            )
            for i, key in enumerate(terms):
                gs = (
                    torch.autograd.grad(
                        parts[key], params, retain_graph=i < 2, allow_unused=True
                    )
                    if parts[key].requires_grad
                    else [None] * len(params)
                )
                flat = []
                for j, (p, g) in enumerate(zip(params, gs)):
                    used[j] |= g is not None
                    flat.append(
                        torch.zeros(p.numel(), dtype=torch.float64)
                        if g is None
                        else g.detach().flatten().cpu().double()
                    )
                count = denominators[key]
                vectors[key] += torch.cat(flat) * count
                numerators[key] += float(parts[key].detach()) * count
                counts[key] += count
            del pred, parts, gs, flat, errors, batch_loss
    finally:
        model.train(was_training)
    for key in terms:
        vectors[key] /= max(counts[key], 1)

    def summary(v):
        angles = {}
        for pair, a, b in [
            ("EF", "energy_loss", "forces_loss"),
            ("Eq", "energy_loss", "charges_loss"),
            ("Fq", "forces_loss", "charges_loss"),
        ]:
            c = cosine(v[a], v[b])
            angles[pair] = (
                None if c is None else math.degrees(math.acos(max(-1.0, min(1.0, c))))
            )
        return {
            "angles_degrees": angles,
            "norms": {k: float(x.norm()) for k, x in v.items()},
            "cos_EF": cosine(v["energy_loss"], v["forces_loss"]),
            "cos_Eq": cosine(v["energy_loss"], v["charges_loss"]),
            "cos_Fq": cosine(v["forces_loss"], v["charges_loss"]),
        }

    groups = {}
    offset = 0
    for (name, p), size in zip(named, sizes):
        groups.setdefault(parameter_group(name), []).append((offset, offset + size))
        offset += size
    result = dict(
        losses={k: numerators[k] / max(counts[k], 1) for k in terms},
        global_gradients=summary(vectors),
        parameter_groups={
            g: summary(
                {k: torch.cat([v[a:b] for a, b in slices]) for k, v in vectors.items()}
            )
            for g, slices in groups.items()
        },
    )
    total = sum(vectors.values())
    updates = {}
    for fresh in (False, True):
        for scale in (1.0, 10.0):
            delta = adam_update(
                params,
                total,
                used,
                optimizer_state,
                scale,
                fresh,
                parameter_names=[name for name, _ in named],
            )
            updates[f"{'fresh' if fresh else 'saved'}_scale_{int(scale)}"] = {
                "gradient_norm": float((total * scale).norm()),
                "update_norm": float(delta.norm()),
                "linearized_loss_changes": {
                    k: float(torch.dot(v, delta)) for k, v in vectors.items()
                },
            }
    result["hypothetical_updates"] = updates
    result[
        "note"
    ] = "No parameters or optimizer state updated. Saved-scale comparisons keep original moments; fresh comparisons use zero moments."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint, args.device)
    model = saved["model"]
    torch.set_default_dtype(next(model.parameters()).dtype)
    data = load_dataset(args.train_file, model)
    indices = torch.randperm(
        len(data), generator=torch.Generator().manual_seed(args.seed)
    )[: args.max_samples].tolist()
    batches = [
        Batch.from_data_list([data[j] for j in indices[i : i + args.batch_size]])
        for i in range(0, len(indices), args.batch_size)
    ]
    cfg = saved.get("training_args", {})
    stage = saved.get("stage", 1)
    prefix = "stage_two_" if stage == 2 else ""
    defaults = (
        (100.0, 10.0, cfg.get("charges_weight", 10.0))
        if stage == 2
        else (10.0, 100.0, 10.0)
    )
    criterion = EnvHeadLoss(
        *(
            cfg.get(prefix + key, default)
            for key, default in zip(
                ("energy_weight", "forces_weight", "charges_weight"), defaults
            )
        )
    )
    report = diagnose(model, batches, criterion, saved["optimizer"])
    report["sample_indices"] = indices
    report["checkpoint_epoch"] = saved["epoch"]
    report["checkpoint_stage"] = stage
    report["loss_weights"] = dict(
        energy=criterion.energy_weight,
        forces=criterion.forces_weight,
        charges=criterion.charges_weight,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
