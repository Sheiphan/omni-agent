"""Computer Control package."""

from importlib import metadata

try:
    __version__ = metadata.version("computer-control")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

from .config import Config, ModelConfig
from .core.controller import ComputerController

__all__ = ["ComputerController", "Config", "ModelConfig"] 