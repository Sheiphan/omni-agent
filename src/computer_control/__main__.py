"""Main entry point for the computer control system."""

import logging
import sys
from typing import NoReturn

from .config import Config
from .core.controller import ComputerController
from .utils.vision import get_caption_model_processor, get_yolo_model


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> NoReturn:
    """Main entry point."""
    # Load configuration
    config = Config.default()

    # Setup logging
    setup_logging(config.debug)
    logger = logging.getLogger(__name__)

    try:
        # Load models
        logger.info("Loading models...")
        som_model = get_yolo_model(model_path=str(config.model.icon_detect_path))
        som_model.to(config.model.device)

        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=config.model.icon_caption_model,
            device=config.model.device,
        )

        # Initialize and run controller
        logger.info("Initializing controller...")
        controller = ComputerController(
            som_model=som_model,
            caption_model_processor=caption_model_processor,
            device=config.model.device,
        )

        logger.info("Starting computer control system...")
        controller.run()
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
