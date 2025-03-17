import os
from typing import Tuple
from unittest.mock import Mock, patch

import pytest

from computer_control.core.controller import ComputerController
from computer_control.models.element_extractor import ElementExtractor


@pytest.fixture
def mock_models() -> Tuple[Mock, dict[str, Mock]]:
    som_model = Mock()
    caption_model_processor = {"processor": Mock(), "model": Mock()}
    caption_model_processor["model"].device = "cpu"
    return som_model, caption_model_processor


@pytest.fixture
def controller(mock_models: Tuple[Mock, dict[str, Mock]]) -> ComputerController:
    som_model, caption_model_processor = mock_models
    return ComputerController(
        som_model=som_model,
        caption_model_processor=caption_model_processor,
        device="cpu",
    )


def test_controller_initialization(
    controller: ComputerController, mock_models: Tuple[Mock, dict[str, Mock]]
) -> None:
    som_model, caption_model_processor = mock_models
    assert controller.som_model == som_model
    assert controller.caption_model_processor == caption_model_processor
    assert controller.device == "cpu"
    assert os.path.exists(controller.image_folder)


@patch("computer_control.core.controller.time.sleep", return_value=None)
@patch("computer_control.core.controller.input", side_effect=["test question", "exit"])
def test_controller_run(mock_input: Mock, mock_sleep: Mock, controller: ComputerController) -> None:
    # Mock screenshot and processing
    with (
        patch.object(controller, "_take_screenshot") as mock_take_screenshot,
        patch.object(controller, "process_screenshot") as mock_process,
        patch.object(controller, "_move_and_click") as mock_move_click,
        patch("os.remove") as mock_remove,
    ):

        # Setup mock returns
        mock_take_screenshot.return_value = "test.png"
        mock_process.return_value = (Mock(), {"0": [100, 200]}, ["Button 1"])

        # Mock LLM response
        mock_llm_response = ElementExtractor(binary_score="Button 1")
        controller.retrieval_grader = Mock()
        controller.retrieval_grader.invoke = Mock(return_value=mock_llm_response)

        # Run controller
        controller.run()

        # Verify main loop execution
        assert mock_take_screenshot.call_count == 2  # One for each iteration
        assert mock_process.call_count == 2
        assert mock_move_click.call_count == 1  # Only for the first iteration
        assert mock_remove.call_count == 2  # Screenshot cleanup
        assert mock_sleep.call_count == 1  # Sleep between iterations
