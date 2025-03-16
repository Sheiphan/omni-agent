# Computer Control

An AI-powered computer control system that uses computer vision to understand and interact with your computer's interface.

## Features

- Screen element detection using YOLO
- Icon and UI element captioning using Florence-2 model
- Automated computer control based on visual understanding

## Requirements

- Python 3.11 or higher
- CUDA-compatible GPU (recommended) or CPU
- Required model weights (see Setup section)

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

## Model Weights Setup

1. Create a `weights` directory in the project root
2. Download required model weights:
   - Icon detection model: Place in `weights/icon_detect/best.pt`
   - Florence-2 caption model: Place in `weights/icon_caption_florence/`

## Usage

Run the computer control system:
```bash
python -m computer_control
```

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
│       ├── core/          # Core functionality
│       ├── models/        # ML model implementations
│       ├── utils/         # Utility functions
│       └── __main__.py    # Entry point
├── tests/                 # Test suite
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
