# bitnet.cpp - Dual Engine 1-bit LLM Inference

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-2.0-blue)
![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-brightgreen)

> **Dual-engine 1-bit LLM inference supporting both I2_S (BitNet) and Q1_0 (Bonsai) formats on Apple Silicon.**

This repository provides **two parallel inference engines** for 1-bit LLMs on Apple Silicon:

1. **ggml-org Engine** - Upstream llama.cpp with Q1_0 Metal support (Bonsai models)
2. **BitNet Engine** - Microsoft's optimized fork with I2_S support (BitNet models)

## Why Dual Engines?

| Feature | ggml-org (Q1_0) | BitNet (I2_S) |
|---------|-----------------|---------------|
| **Models** | Bonsai 1.7B/4B/8B | BitNet 2B/3B, Falcon3 |
| **Model Size** | 231-540 MiB | 1.7-3.0 GiB |
| **Prompt Speed** | 960-2,534 t/s | 490-1,455 t/s |
| **Token Speed** | 122-208 t/s | 65-68 t/s |
| **Memory Efficiency** | ~7x smaller | Larger but mature |

**Recommendation**: Use **Bonsai Q1_0** for best performance/size ratio, **I2_S** for existing BitNet models.

## Performance Comparison

### Bonsai Q1_0 (ggml-org Engine)

| Model | Size | pp512 (t/s) | tg64 (t/s) |
|-------|------|-------------|------------|
| Bonsai-1.7B | 231 MiB | **2,534** | **208** |
| Bonsai-4B | 540 MiB | **1,040** | **122** |

### BitNet I2_S (BitNet Engine)

| Model | Size | pp512 (t/s) | tg64 (t/s) |
|-------|------|-------------|------------|
| BitNet-2B | 1.71 GiB | **1,455** | **68** |
| Falcon3-7B | 3.05 GiB | **537** | **65** |

**Key Insight**: Q1_0 Bonsai models are **2-3x faster** for token generation while being **5-7x smaller**.

## Quick Start (macOS)

### Prerequisites

- macOS 12+ (Monterey or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Homebrew: `cmake` >= 4.3.1, Python >= 3.11

### Option 1: Q1_0 Models (Bonsai)

```bash
# Clone the repository
git clone --recursive https://github.com/your-repo/BitNet.git
cd BitNet

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build ggml-org engine (Q1_0 support)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Download Bonsai model
source .venv/bin/activate
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='prism-ml/Bonsai-1.7B-gguf', local_dir='models/Bonsai-1.7B', allow_patterns='*.gguf')"

# Run inference
/tmp/llama-build/bin/llama-cli -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf -p "Hello" -ngl 99
```

### Option 2: I2_S Models (BitNet)

```bash
# Build BitNet engine
cd 3rdparty/bitnet-official
python setup_env.py -md ../../models/BitNet-b1.58-2B-4T -q i2_s

# Run inference
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```

### Run Benchmarks

```bash
# Dual engine benchmark script
./benchmark_dual_engines.sh
```

## Supported Models

### Q1_0 Models (Bonsai - ggml-org Engine)

| Model | Parameters | Size | Metal GPU | Notes |
|-------|------------|------|-----------|-------|
| [Bonsai-1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) | 1.72B | 231 MiB | ✅ | **Fastest, fits in cache** |
| [Bonsai-4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | 4.02B | 540 MiB | ✅ | Good balance |
| [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | 8.19B | 1.07 GiB | ✅ | Maximum quality |

### I2_S Models (BitNet - BitNet Engine)

| Model | Parameters | Size | Metal GPU | Notes |
|-------|------------|------|-----------|-------|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4B | 1.71 GiB | ✅ | Original BitNet |
| [Falcon3-7B-1.58bit](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit) | 7.0B | 3.05 GiB | ✅ | Larger model |

**Download Falcon3 (GGUF):**
```bash
hf download tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF --local-dir models/Falcon3-7B-1.58bit
```

## macOS-Specific Notes

### Build Requirements

The successful build on Apple Silicon uses:
- Apple Clang (not Homebrew LLVM 18 - has macOS SDK header issues)
- CMake from Homebrew
- Python 3.11+ in virtual environment

If build hangs on M-series hardware, use:
```bash
CFLAGS="-mllvm -disable-interleaved-load-combine" \
CXXFLAGS="-mllvm -disable-interleaved-load-combine" \
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

### Metal GPU Activation

**No separate build required.** Metal is automatically enabled on Apple Silicon:
- Metal is built into `llama.cpp` via `GGML_METAL=1`
- Use `-ngl 31` to offload all 31 layers to GPU
- Use `-ngl 0` for CPU-only mode (still uses Metal for memory management)

### Known Issues

**Warning Messages:**
The following warnings are harmless and come from GGUF model metadata:
- `missing pre-tokenizer type, using: 'default'`
- `GENERATION QUALITY WILL BE DEGRADED!`
- `control token ... is not marked as EOG`

These do not affect actual performance.

### Troubleshooting

**Helper script not working:**
```bash
chmod +x run_bitnet_2b.sh
```

**Build hangs:**
Add compiler flags as shown above or use:
```bash
export CFLAGS="-mllvm -disable-interleaved-load-combine"
export CXXFLAGS="-mllvm -disable-interleaved-load-combine"
```

## Helper Scripts

### `run_bitnet_2b.sh`

One-shot inference helper with automatic model downloads:

```bash
./run_bitnet_2b.sh [options] "your prompt"

Options:
  --model 2b|falcon3    Select model (default: 2b)
  --chat                Enable interactive chat mode
  --verbose             Show full llama-cli output
  --help                Show help

Examples:
  ./run_bitnet_2b.sh "Explain quantum computing"
  ./run_bitnet_2b.sh --model falcon3 --chat "You are a helpful coding assistant"
```

### `utils/smoke_test.py`

Quick sanity check:
```bash
python utils/smoke_test.py --model 2b
```

## Original Documentation

For complete documentation on:
- Windows/Linux setup → [Microsoft BitNet](https://github.com/microsoft/BitNet)
- CUDA GPU support → [gpu/README.md](gpu/README.md)
- CPU kernel optimization → [src/README.md](src/README.md)
- Model conversion → See `utils/` directory

## Acknowledgements

This fork is based on [Microsoft BitNet](https://github.com/microsoft/BitNet), which builds on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. Optimized kernels use the Lookup Table methodologies from [T-MAC](https://github.com/microsoft/T-MAC/).

## License

MIT License - See upstream [LICENSE](https://github.com/microsoft/BitNet/blob/main/LICENSE) for details.
