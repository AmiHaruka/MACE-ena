"""Full v2 checkpoints; trusted local model files only."""

from dataclasses import asdict
import torch

FORMAT = "envhead-v2"


def save_checkpoint(
    path, model, optimizer=None, scheduler=None, epoch=0, training_args=None
):
    torch.save(
        dict(
            format=FORMAT,
            model=model,
            config=asdict(model.config),
            optimizer=optimizer.state_dict() if optimizer else None,
            scheduler=scheduler.state_dict() if scheduler else None,
            epoch=epoch,
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
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError(
            "Expected full envhead-v2 checkpoint; v1 head checkpoints are incompatible"
        )
    return saved
