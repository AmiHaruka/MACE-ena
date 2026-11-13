"""Training checkpoints and portable inference models."""

import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

FORMAT = "envhead-v4"  # Serialization identifier used by existing checkpoints.
METRICS_VERSION = 3  # RMSE excludes missing E/F labels; loss reductions are unchanged.


def _save_atomic(payload, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    training_args=None,
    best_loss=None,
    best_epoch=None,
    optimizer_steps=0,
    *,
    ema=None,
    stage=1,
    rank_states=None,
):
    if getattr(model, "format_version", 4) != 4:
        raise ValueError("Unsupported EnvHead model format")
    _save_atomic(
        dict(
            format=FORMAT,
            model=model,
            config=asdict(model.config),
            optimizer=optimizer.state_dict() if optimizer else None,
            scheduler=scheduler.state_dict() if scheduler else None,
            epoch=epoch,
            best_loss=best_loss,
            best_epoch=best_epoch,
            optimizer_steps=optimizer_steps,
            metrics_version=METRICS_VERSION,
            training_args=training_args or {},
            training_state_version=1,
            ema=ema.state_dict() if ema is not None else None,
            stage=stage,
            rank_states=rank_states,
            rng_state=torch.get_rng_state(),
            cuda_rng_state=[torch.cuda.get_rng_state()]
            if torch.cuda.is_initialized()
            else None,
        ),
        path,
    )


def _load(path, device):
    try:
        saved = torch.load(path, map_location=device, weights_only=False)
    except ModuleNotFoundError as error:
        if error.name in {
            "mace.modules.envhead.modules.model",
            "mace.modules.envhead.modules.config",
            "mace.modules.envhead.modules.readout",
            "mace.modules.envhead.modules.execution",
        }:
            raise ValueError(
                "Cannot load the saved model class; check its module path"
            ) from error
        raise
    return saved


def _validate_checkpoint(saved):
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError(
            "Expected a full EnvHead checkpoint"
        )
    if getattr(saved["model"], "format_version", 4) != 4:
        raise ValueError("Checkpoint and model formats do not match")


def _prepare_device(model):
    from ..backends import prepare_cueq_device

    return prepare_cueq_device(model)


def _copy_ema(model, state):
    parameters = [p for p in model.parameters() if p.requires_grad]
    shadow = state.get("shadow_params", [])
    if len(parameters) != len(shadow) or any(
        parameter.shape != average.shape
        for parameter, average in zip(parameters, shadow)
    ):
        raise ValueError("EMA state does not match the model's trainable parameters")
    with torch.no_grad():
        for parameter, average in zip(parameters, shadow):
            parameter.copy_(average)


def load_checkpoint(path, device="cpu"):
    """Load model parameters and optimizer state for training resume."""
    saved = _load(path, device)
    _validate_checkpoint(saved)
    _prepare_device(saved["model"])
    return saved


def export_model(path, model, ema=None, metadata=None):
    """Save a complete CPU eval model without changing any live training state.

    EMA values are copied only into the detached export. The trainability mask is
    retained so an exported cuEq model has the same parameter owners and bases.
    """
    if getattr(model, "format_version", 4) != 4:
        raise ValueError("Unsupported EnvHead model format")
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        exported = deepcopy(model).cpu().eval()
        if ema is not None:
            _copy_ema(exported, ema.state_dict())
        _prepare_device(exported)
        exported.training_metadata = deepcopy(metadata or {})
        _save_atomic(exported, path)
    return exported


def load_inference(path, device="cpu"):
    """Load an inference model with its metadata, preferring saved EMA weights."""
    saved = _load(path, device)
    if isinstance(saved, torch.nn.Module):
        model = saved
        if getattr(model, "format_version", None) != 4:
            raise ValueError("Expected an EnvHead model")
        saved = dict(getattr(model, "training_metadata", {}), model=model)
    else:
        _validate_checkpoint(saved)
        model = saved["model"]
        if saved.get("ema") is not None:
            _copy_ema(model, saved["ema"])
            saved["inference_weights"] = "ema"
        else:
            saved["inference_weights"] = "raw"
    _prepare_device(model)
    model.eval()
    return saved


def load_model(path, device="cpu"):
    """Load a complete .model export or the inference weights of a checkpoint."""
    return load_inference(path, device)["model"]
