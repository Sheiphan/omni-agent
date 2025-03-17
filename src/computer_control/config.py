"""Configuration management for the computer control system."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


def get_default_device() -> str:
    """Get the default device based on system capabilities."""
    if torch.backends.mps.is_available():
        return "mps"  # Use Metal Performance Shaders on MacOS
    elif torch.cuda.is_available():
        return "cuda"  # Use CUDA if available
    return "cpu"  # Fallback to CPU


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    icon_detect_path: Path
    icon_caption_model: str
    device: str = get_default_device()


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig
    debug: bool = False
    screenshot_interval: float = 1.0  # seconds

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            model=ModelConfig(
                icon_detect_path=Path(os.getenv("ICON_DETECT_PATH", "weights/icon_detect/best.pt")),
                icon_caption_model=os.getenv("ICON_CAPTION_MODEL", "Salesforce/blip2-opt-2.7b"),
                device=os.getenv("DEVICE", get_default_device()),
            ),
            debug=os.getenv("DEBUG", "").lower() == "true",
            screenshot_interval=float(os.getenv("SCREENSHOT_INTERVAL", "1.0")),
        )

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            model=ModelConfig(
                icon_detect_path=Path("weights/icon_detect/best.pt"),
                icon_caption_model="Salesforce/blip2-opt-2.7b",
            )
        )
