#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if required model weights exist
if [ ! -d "weights/icon_detect" ] || [ ! -f "weights/icon_detect/best.pt" ]; then
    echo "Error: Icon detection model weights not found in weights/icon_detect/best.pt"
    exit 1
fi

if [ ! -d "weights/icon_caption_florence" ]; then
    echo "Error: Florence-2 caption model weights not found in weights/icon_caption_florence/"
    exit 1
fi

# Run the computer control system
python -m computer_control

# Deactivate virtual environment
deactivate