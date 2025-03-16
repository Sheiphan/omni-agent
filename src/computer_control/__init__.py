"""Computer Control package."""

from importlib import metadata

try:
    __version__ = metadata.version("computer-control")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

from .core.controller import ComputerController
from .config import Config, ModelConfig

__all__ = ["ComputerController", "Config", "ModelConfig"]