import pytest
from unittest.mock import Mock, patch
from computer_control.core.controller import ComputerController

@pytest.fixture
def mock_models():
    som_model = Mock()
    caption_model_processor = Mock()
    return som_model, caption_model_processor

@pytest.fixture
def controller(mock_models):
    som_model, caption_model_processor = mock_models
    return ComputerController(
        som_model=som_model,
        caption_model_processor=caption_model_processor,
        device="cpu"
    )

def test_controller_initialization(controller, mock_models):
    som_model, caption_model_processor = mock_models
    assert controller.som_model == som_model
    assert controller.caption_model_processor == caption_model_processor
    assert controller.device == "cpu"

@patch("computer_control.core.controller.time.sleep", return_value=None)
def test_controller_run(mock_sleep, controller):
    # Mock the run loop to exit after one iteration
    with patch.object(controller, "_should_continue", side_effect=[True, False]):
        controller.run()
        # Add assertions based on your specific implementation
        assert mock_sleep.called 