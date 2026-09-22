"""Configuration for the unified-graph EnvHead v2 model."""

from dataclasses import dataclass


@dataclass
class EnvHeadConfig:
    channels: int = 16
    max_l: int = 2
    r_env: float = 5.0
    backend: str = "ace"
    triton: bool = False
    cueq: bool = False
    bidirectional: bool = True
    feature_source: str = "mixed_products"
    global_mode: str = "physical"
    freeze_foundation: bool = False
    epsilon: float = 1.0
    radial: str = "polynomial"
    radial_shape: float = 2.0
    radial_baseline: float = 0.0
    radial_amplitude: float = 1.0
    radial_basis: int = 8
    correlation: int = 2
    projection_width: float = 1.0
    projection_hardness: float = 1.0
    projection_regularization: float = 1e-4
    projection_condition_limit: float = 1e8
    mm_chunk_size: int = 1024

    def __post_init__(self):
        if self.backend not in ("ace", "painn", "e2former"):
            raise ValueError("backend must be ace, painn or e2former")
        if self.max_l < 1 or self.max_l > 2:
            raise ValueError("v2 supports max_l=1 or 2")
        if self.triton and self.backend != "e2former":
            raise ValueError("Triton aggregation is an E2Former option")
        if self.feature_source not in ("mixed_products", "last_message"):
            raise ValueError("Unknown feature_source")
        if self.global_mode not in ("physical", "learned"):
            raise ValueError("Unknown global_mode")
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
            min(self.channels, self.radial_basis, self.mm_chunk_size) < 1
            or self.r_env < 0
        ):
            raise ValueError(
                "Invalid channels, radial basis, chunk size or environment radius"
            )
        if (
            min(
                self.projection_width,
                self.projection_hardness,
                self.projection_regularization,
                self.radial_shape,
            )
            <= 0
        ):
            raise ValueError(
                "Widths, hardness, regularization and shape must be positive"
            )
