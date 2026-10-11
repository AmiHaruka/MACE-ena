"""Full versioned checkpoints; trusted local model files only."""

from dataclasses import asdict
import torch

FORMAT = "envhead-v3"
FORMAT_V4 = "envhead-v4"
METRICS_VERSION = 2


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
):
    torch.save(
        dict(
            format=FORMAT_V4 if getattr(model, "format_version", 3) == 4 else FORMAT,
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
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        ),
        path,
    )


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(saved, dict) or saved.get("format") not in (FORMAT, FORMAT_V4):
        raise ValueError(
            "Expected full envhead-v3 or envhead-v4 checkpoint; v1/v2 checkpoints are incompatible"
        )
    expected = 4 if saved["format"] == FORMAT_V4 else 3
    if getattr(saved["model"], "format_version", 3) != expected:
        raise ValueError("Checkpoint format and model version disagree")
    if expected == 3 and "message_mode" not in saved["config"]:
        # Older v3 model objects contain the full tensor product parameters.
        saved["config"]["message_mode"] = "full"
        saved["model"].config.message_mode = "full"
    return saved
