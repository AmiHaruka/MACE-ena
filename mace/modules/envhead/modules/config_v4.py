"""Single-pass environment conditioning; only the two active charge routes."""

from dataclasses import dataclass


@dataclass
class EnvHeadConfigV4:
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
    charge_mode: str = "fukui_projection"
    projection_width: float = 1.0
    projection_hardness: float = 1.0
    projection_regularization: float = 1e-8
    mm_chunk_size: int = 1024

    def __post_init__(self):
        if self.backend != "ace" or self.message_mode not in ("channelwise", "full"):
            raise ValueError("v4 requires ACE with channelwise or full messages")
        if self.charge_mode not in ("fukui_projection", "charge_projection"):
            raise ValueError("v4 supports fukui_projection and charge_projection")
        if self.finetune not in ("partial", "full", "frozen"):
            raise ValueError("finetune must be partial, full or frozen")
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
                self.correlation,
                self.mm_chunk_size,
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
