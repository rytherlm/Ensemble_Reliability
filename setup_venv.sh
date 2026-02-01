#!/bin/bash
# setup_venv.sh: Create .venv and install dependencies if needed
set -e

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
else
    echo "$REQ_FILE not found. Please create it with your dependencies."
    exit 1
fi

echo "Environment setup complete."
