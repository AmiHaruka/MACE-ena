"""Explicit v3 configuration; v2 checkpoints/configurations are not migrated."""

from dataclasses import dataclass


@dataclass
class EnvHeadConfig:
    channels: int = 16
    max_l: int = 2
    near_radius: float = 5.0
    switch_fraction: float = 0.8
    backend: str = "ace"
    message_mode: str = "channelwise"
    cueq: bool = False
    bidirectional: bool = True
    feature_source: str = "mixed_products"
    finetune: str = "partial"
    epsilon: float = 1.0
    radial: str = "constant"
    radial_shape: float = 2.0
    radial_baseline: float = 0.0
    radial_amplitude: float = 1.0
    radial_basis: int = 8
    correlation: int = 2
    charge_mode: str = "charge_projection"
    charge_iterations: int = 3
    charge_mixing: float = 0.5
    projection_width: float = 1.0
    projection_hardness: float = 1.0
    projection_regularization: float = 1e-8
    esp_order: int = 41
    esp_shells: tuple = (1.4, 1.8, 2.2)
    esp_chunk_size: int = 256
    mm_chunk_size: int = 1024
    residual_charges: bool = True

    def __post_init__(self):
        if self.message_mode not in ("channelwise", "full"):
            raise ValueError("message_mode must be channelwise or full")
        if self.backend != "ace":
            raise ValueError("v3 implements the ACE environment backend")
        if self.finetune not in ("partial", "full", "frozen"):
            raise ValueError("finetune must be partial, full or frozen")
        if self.charge_mode not in (
            "charge_projection",
            "esp_reconstruction",
            "iterative_response",
            "fukui_projection",
        ):
            raise ValueError("Unknown charge reconstruction")
        if (
            not isinstance(self.charge_iterations, int)
            or self.charge_iterations < 0
            or not 0 < self.charge_mixing <= 1
        ):
            raise ValueError(
                "Require nonnegative charge_iterations and 0 < charge_mixing <= 1"
            )
        if self.feature_source not in ("mixed_products", "last_message"):
            raise ValueError("Unknown feature source")
        if self.max_l not in (1, 2) or not 0 < self.switch_fraction < 1:
            raise ValueError("Require max_l=1/2 and 0 < switch_fraction < 1")
        if self.radial not in (
            "constant",
            "cosine",
            "polynomial",
            "mollifier",
            "switch",
            "bell",
        ):
            raise ValueError("Unknown radial modulation")
        if (
            min(
                self.channels,
                self.radial_basis,
                self.mm_chunk_size,
                self.esp_chunk_size,
                self.near_radius,
                self.projection_width,
                self.projection_hardness,
                self.projection_regularization,
                self.radial_shape,
            )
            <= 0
        ):
            raise ValueError(
                "Dimensions, radii and reconstruction parameters must be positive"
            )
        if not self.esp_shells or min(self.esp_shells) <= 1.2:
            raise ValueError(
                "ESP shells must lie outside the exclusion transition (1.2 vdW radii)"
            )
