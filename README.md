# Computer Control

An AI-powered computer control system that uses computer vision and natural language processing to understand and interact with your computer's interface automatically. The system can identify UI elements, process natural language commands, and perform automated actions.

## Features

- **Screen Element Detection**: Advanced computer vision using SOM model for accurate UI element identification
- **OCR Integration**: Uses PaddleOCR for robust text detection and recognition
- **Natural Language Processing**: GPT-4-mini powered element extraction for understanding user commands
- **Automated Control**: Intelligent cursor movement and click actions based on visual understanding
- **Real-time Processing**: Continuous screenshot analysis and interaction
- **Interactive CLI**: User-friendly command-line interface with rich output formatting

## Requirements

- Python 3.11 or higher
- CUDA-compatible GPU (recommended) or CPU
- PyAutoGUI for computer control
- Required AI models:
  - SOM model for element detection
  - Caption model processor
  - GPT-4-mini for command processing
  - PaddleOCR for text recognition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/computer-control.git
cd computer-control
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

## Dependencies

Key dependencies include:
- `pyautogui`: For computer control operations
- `Pillow`: Image processing
- `rich`: Terminal output formatting
- `langchain`: LLM integration
- `langchain_openai`: OpenAI model integration
- `paddleocr`: Text detection and recognition

## Usage

1. Start the computer control system:
```bash
python -m computer_control
```

2. The system will:
   - Take periodic screenshots of your screen
   - Process and analyze UI elements
   - Wait for your natural language commands
   - Execute the requested actions automatically

3. Example commands:
   - "Click on the start menu"
   - "Open the app folder"
   - "Go to settings"

## Technical Details

### Component Architecture

1. **Screenshot Processing**
   - Automated screenshot capture
   - Image preprocessing and conversion
   - Dynamic scaling based on screen resolution

2. **Element Detection**
   - SOM model-based element identification
   - OCR text detection with PaddleOCR
   - Coordinate mapping for precise interaction

3. **Command Processing**
   - Natural language understanding with GPT-4-mini
   - Element matching and validation
   - Coordinate calculation for cursor movement

4. **Action Execution**
   - Smooth cursor movement
   - Click action verification
   - Error handling and recovery

## Development

- Run tests: `pytest`
- Format code: `black . && isort .`
- Type checking: `mypy src tests`
- Lint code: `ruff check .`

## Project Structure

```
computer-control/
├── src/
│   └── computer_control/
│       ├── core/          # Core control and processing logic
│       │   └── controller.py  # Main controller implementation
│       ├── models/        # AI model implementations
│       │   └── element_extractor.py  # Element extraction logic
│       ├── utils/         # Utility functions
│       │   └── vision.py  # Computer vision utilities
│       └── __main__.py    # Entry point
├── tests/                 # Test suite
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- SOM model for element detection
- Florence-2 caption model
- OpenAI for GPT-4-mini
- PaddlePaddle team for PaddleOCR
