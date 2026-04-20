#!/bin/bash
set -e

echo "========================================"
echo "Dual Engine Setup for 1-bit LLMs"
echo "========================================"

BITNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "Step 1: Creating Python 3.11 virtual environment for BitNet..."
if [ ! -d "$BITNET_ROOT/.venv-bitnet" ]; then
    /opt/homebrew/bin/python3.11 -m venv "$BITNET_ROOT/.venv-bitnet"
    echo "Created .venv-bitnet"
else
    echo ".venv-bitnet already exists"
fi

echo ""
echo "Step 2: Activating environment and installing dependencies..."
source "$BITNET_ROOT/.venv-bitnet/bin/activate"
pip install --upgrade pip -q

echo ""
echo "Step 3: Initializing BitNet submodules..."
cd "$BITNET_ROOT/3rdparty/bitnet-official"
git submodule update --init --recursive

echo ""
echo "Step 4: Installing BitNet requirements..."
pip install -r requirements.txt -q

echo ""
echo "Step 5: Building BitNet engine (I2_S support)..."
python setup_env.py -md ../../models/BitNet-b1.58-2B-4T -q i2_s

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use:"
echo "  source .venv-bitnet/bin/activate"
echo "  cd 3rdparty/bitnet-official"
echo "  python run_inference.py -m ../../models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p 'Hello'"
echo ""
echo "For Q1_0 models (Bonsai), use the prebuilt ggml-org engine:"
echo "  /tmp/llama-build/bin/llama-cli -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf -p 'Hello' -ngl 99"
