# bitnet.cpp - Apple Silicon Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)
![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-brightgreen)

> **This fork is optimized for macOS and Apple Silicon (M1/M2/M3/M4) with enhanced Metal GPU support.**

This repository is a fork of the official [Microsoft BitNet](https://github.com/microsoft/BitNet) implementation, with specific optimizations and documentation for macOS users. It provides fast, lossless inference of 1.58-bit quantized LLMs using Apple Silicon's Metal Performance Shaders (MPS).

## Why This Fork

This fork adds:
- **Enhanced Apple Silicon documentation** with macOS-specific setup instructions
- **Metal GPU benchmarks** showing 13x performance improvement over CPU-only
- **Simplified helper scripts** (`run_bitnet_2b.sh`) with automatic model downloads
- **macOS-specific build notes** and troubleshooting
- **Expanded model support** tested on Apple Silicon

## Performance: Metal vs CPU

Benchmark results on Apple M2 Max:

| Configuration | GPU Layers | Prompt Processing | Text Generation | Speedup |
|--------------|-----------|-------------------|-----------------|---------|
| **Metal GPU (Full)** | 31 | **1,264 t/s** | 70.6 t/s | **13.2x** |
| **Metal GPU (Partial)** | 20 | 309.6 t/s | 69.3 t/s | 4.0x |
| **CPU-Only** | 0 | 77.5 t/s | 72.2 t/s | 1.0x |

**Key Insight:** Metal GPU acceleration provides massive benefits for **prompt processing** (13x faster), while text generation performance is similar between CPU and GPU. Use `-ngl 31` (all layers) for best performance with long prompts.

> **Note:** Unlike NVIDIA CUDA, no separate build is needed for Metal. The `llama.cpp` backend automatically enables Metal on Apple Silicon.

## Quick Start (macOS)

### Prerequisites

- macOS 12+ (Monterey or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Homebrew: `cmake` >= 4.3.1, Python >= 3.11
- Apple Clang (LLVM 18 has known issues)

```bash
# Clone the repository
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download and setup model
hf download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

### Run Inference

```bash
# Quick one-shot inference (recommended)
./run_bitnet_2b.sh "You are a helpful assistant"

# With specific model
./run_bitnet_2b.sh --model falcon3 "What is machine learning?"

# Interactive chat mode
./run_bitnet_2b.sh --chat "You are a coding assistant"

# Direct Python wrapper (more options)
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```

### Run Benchmarks

```bash
# Quick benchmark
python utils/e2e_benchmark.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 64 -p 128 -t 8

# Comprehensive comparison (CPU vs Metal)
./build/bin/llama-bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -ngl 0,31 -p 128 -n 64 -t 8
```

## Supported Models

This fork includes expanded model support tested specifically on Apple Silicon:

| Model | Parameters | Apple Silicon | Metal GPU | Notes |
|-------|-----------|---------------|-----------|-------|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4B | ✅ | ✅ | **Recommended default** |
| [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) | 3.3B | ✅ | ✅ | Larger model |
| [Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit) | 7.0B | ✅ | ✅ | Use GGUF version |
| [Llama3-8B-1.58](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | 8.0B | ✅ | ✅ | 100B tokens |
| [Falcon3 Family](https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026) | 1B-10B | ✅ | ✅ | Multiple sizes |

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
