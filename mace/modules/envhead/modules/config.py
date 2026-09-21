"""Configuration for the direct-delta, fixed-depth multipole head."""
from dataclasses import dataclass
import math


@dataclass
class EnvHeadConfig:
    fusion: str = "product"
    charge_route: str = "fukui"
    density_max_l: int = 2
    mmc: bool = False
    channels: int = 32
    num_interactions: int = 2
    correlation: int = 2
    gaussian_width: float = 1.0
    mm_chunk_size: int = 1024

    def __post_init__(self):
        if self.fusion not in ("additive", "product", "message"):
            raise ValueError("fusion must be additive, product or message")
        if self.charge_route not in ("fukui", "eem", "sqe"):
            raise ValueError("charge_route must be fukui, eem or sqe")
        if self.density_max_l not in (1, 2):
            raise ValueError("density_max_l must be 1 or 2")
        if not isinstance(self.mmc, bool):
            raise ValueError("mmc must be boolean")
        for key in ("channels", "num_interactions", "correlation", "mm_chunk_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.gaussian_width) or self.gaussian_width <= 0:
            raise ValueError("gaussian_width must be finite and positive")
