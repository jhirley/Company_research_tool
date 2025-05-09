#!/bin/bash

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Activate the virtual environment
echo "Activating virtual environment..."
python -m venv .venv 
source .venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "Failed to activate virtual environment. Please check that .venv exists."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Run the application with: source .venv/bin/activate && python main.py"
