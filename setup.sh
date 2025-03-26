#!/bin/bash

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv.nosync/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != *".venv.nosync"* ]]; then
    echo "Failed to activate virtual environment. Please check that .venv.nosync exists."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Run the application with: source .venv.nosync/bin/activate && python main.py"
