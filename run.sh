#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source venv/bin/activate
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Check if model weights exist
if [ ! -f "weights/icon_detect/best.pt" ]; then
    echo "Error: YOLO model weights not found at weights/icon_detect/best.pt"
    exit 1
fi

if [ ! -d "weights/icon_caption_florence" ]; then
    echo "Error: Florence-2 model not found at weights/icon_caption_florence/"
    exit 1
fi

# Run the application
python -m computer_control 