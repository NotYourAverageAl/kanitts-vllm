#!/bin/bash
set -e

echo "=== KaniTTS-vLLM Setup ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Validate Python version (only 3.10-3.12 supported)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "Error: This project requires Python 3.10, 3.11, or 3.12"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo "Python version is supported"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo ""
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed or not in PATH"
        echo "Please restart your shell and run this script again, or install uv manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    echo "uv installed successfully"
fi

echo ""
echo "uv version: $(uv --version)"

# Verify CUDA is available
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. CUDA may not be available."
    echo "This project requires NVIDIA GPU with CUDA."
fi

# Add dependencies using uv
echo ""
echo "Adding dependencies..."

# Install FastAPI and Uvicorn
echo "Installing FastAPI and Uvicorn..."
uv pip install fastapi uvicorn

# Install nemo-toolkit (which will install transformers 4.53)
echo ""
echo "Installing nemo-toolkit[tts]..."
uv pip install "nemo-toolkit[tts]==2.4.0"

# Install vLLM with automatic torch backend detection
echo ""
echo "Installing vLLM (this will automatically install the correct PyTorch version)..."
uv pip install vllm --torch-backend=auto

# Force reinstall transformers to 4.57.1 (required for model compatibility)
echo ""
echo "Upgrading transformers to 4.57.1..."
echo "Note: nemo-toolkit[tts] requires transformers==4.53, but we need 4.57.1 for model compatibility"
uv pip install "transformers==4.57.1"

# Verify installation
echo ""
echo "=== Verifying Installation ==="
echo ""

uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CUDA not available')"
uv run python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
uv run python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
uv run python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now start the server with:"
echo "  uv run python server.py"
echo ""
echo "Or activate the virtual environment and run directly:"
echo "  source .venv/bin/activate"
echo "  python server.py"
echo ""
echo "Note: Models will be automatically downloaded on first run (~1.5GB)"
