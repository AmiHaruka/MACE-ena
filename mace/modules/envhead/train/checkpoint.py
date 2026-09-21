"""Strict foundation-bound head files and native MACE recovery checkpoints."""
from dataclasses import asdict

import torch

from mace.tools.checkpoint import CheckpointHandler

from ..adapters import foundation_adapter
from ..modules.config import EnvHeadConfig
from ..modules.model import EnvMACE

CONVENTIONS = dict(
    units={
        "energy": "eV",
        "length": "Angstrom",
        "charge": "e",
        "dipole": "e Angstrom",
        "quadrupole": "e Angstrom^2",
    },
    density="q*g_sigma - mu.grad(g_sigma) + 0.5*Theta:hessian(g_sigma); STF second moment; MM point sources",
    normalization={
        "charge_scale": 1.0,
        "dipole_scale": 1.0,
        "quadrupole_scale": 1.0,
        "energy_scale": 1.0,
    },
    format_version=4,
    charge_conditioning_features=[
        "geometry_0e",
        "Q/e",
        "Q/(N*e)",
        "group_Q/(group_N*e)",
    ],
    quadrupole_basis="sqrt(5)*e3nn.wigner_3j(1,1,2); orthonormal STF",
    internal_self_interaction=False,
)


def metadata(model):
    return dict(
        **CONVENTIONS,
        config=asdict(model.env_head.config),
        adapter=foundation_adapter(model).name,
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
    )


def save_head(path, model):
    torch.save(dict(metadata=metadata(model), head=model.env_head.state_dict()), path)


def load_head(path, foundation):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    if any(meta.get(k) != v for k, v in CONVENTIONS.items()):
        raise ValueError(
            "Only direct-delta EnvHead format_version=4 files are supported; old heads cannot be migrated"
        )
    settings = dict(meta["config"])
    if "adapter" not in meta:
        raise ValueError("Missing foundation adapter metadata")
    adapter = foundation_adapter(foundation).name
    if meta.get("adapter", "scale-shift-mace-v1") != adapter:
        raise ValueError("Foundation adapter mismatch")
    model = EnvMACE(
        foundation,
        EnvHeadConfig(**settings),
        dtype=getattr(torch, meta.get("dtype", "float64")),
    )
    model.env_head.load_state_dict(payload["head"], strict=True)
    return model


class EnvCheckpointHandler(CheckpointHandler):
    """Retain MACE file management, enforce metadata and optimizer restoration."""

    def save(self, state, epochs, keep_last=False):
        payload = self.builder.create_checkpoint(state)
        payload.update(
            validation_loss_version=1,
            envhead_metadata=metadata(state.model),
            epoch=epochs,
            rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else [],
        )
        self.io.save(payload, epochs, keep_last)

    def load_latest(self, state, swa=False, device=None, strict=True):
        result = self.io.load_latest(swa=swa, device=device)
        if result is None:
            return None
        payload, epoch = result
        saved_meta = dict(payload.get("envhead_metadata", {}))
        # Old v4 files may contain this obsolete field; ignore it.
        saved_meta.pop("foundation_fingerprint", None)
        if saved_meta.get("format_version") != 4:
            raise ValueError(
                "Only direct-delta EnvHead format_version=4 checkpoints are supported"
            )
        if saved_meta != metadata(state.model):
            raise ValueError(
                "Recovery checkpoint configuration mismatch; start a new run"
            )
        state.model.load_state_dict(payload["model"], strict=True)
        state.optimizer.load_state_dict(payload["optimizer"])
        state.lr_scheduler.load_state_dict(payload["lr_scheduler"])
        torch.set_rng_state(payload["rng_state"].cpu())
        if payload.get("cuda_rng_states") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng_states"]])
        return epoch
