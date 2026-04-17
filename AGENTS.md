# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-15
**Commit:** 01eb415
**Branch:** main

## OVERVIEW
BitNet is a mixed CPU and GPU inference repo for 1-bit LLMs. The root workflow is CMake + top-level Python orchestration on top of a `3rdparty/llama.cpp` submodule boundary.

## STRUCTURE
```text
BitNet/
├── src/              # CPU kernels and CMake-owned native sources
├── include/          # shared kernel config and BitNet ggml headers
├── gpu/              # separate Python/CUDA inference workflow
├── utils/            # conversion, benchmarking, tuning, codegen scripts
├── preset_kernels/   # committed generated kernel presets by model
├── docs/             # focused technical docs, not an implementation surface
├── assets/ media/    # images and demo media only
├── setup_env.py      # root orchestration for codegen, build, quantization
├── run_inference.py  # wrapper around build/bin/llama-cli
└── run_inference_server.py  # wrapper around build/bin/llama-server
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Repo bootstrap, model prep, CPU build | `setup_env.py`, `README.md`, `CMakeLists.txt` | Main entry path for CPU workflow |
| CPU kernel implementation or tuning | `src/`, `include/gemm-config.h` | See `src/AGENTS.md` |
| GPU kernel build or GPU inference flow | `gpu/` | Separate env and commands; see `gpu/AGENTS.md` |
| Conversion, benchmarks, perplexity, tuning | `utils/` | Script-heavy operational surface; see `utils/AGENTS.md` |
| Generated preset headers/configs | `preset_kernels/` | Treat as outputs, not the primary edit surface |
| Security reporting process | `SECURITY.md` | Do not use public GitHub issues |

## CODE MAP
| Symbol / Surface | Type | Location | Role |
|------------------|------|----------|------|
| `setup_env.py` | CLI orchestrator | `./setup_env.py` | Installs gguf tooling, runs codegen, builds, prepares models |
| `run_inference.py` | CLI wrapper | `./run_inference.py` | Launches `build/bin/llama-cli` |
| `run_inference_server.py` | CLI wrapper | `./run_inference_server.py` | Launches `build/bin/llama-server` |
| `ggml-bitnet-mad.cpp` | native hotspot | `src/ggml-bitnet-mad.cpp` | Main CPU kernel implementation hotspot |
| `ggml-bitnet-lut.cpp` | native source | `src/ggml-bitnet-lut.cpp` | LUT-backed CPU kernel path |
| `gemm-config.h` | config header | `include/gemm-config.h` | Central CPU tuning knob file |
| `generate.py` | GPU app | `gpu/generate.py` | GPU inference entrypoint |
| `compile.sh` | CUDA build | `gpu/bitnet_kernels/compile.sh` | Builds `libbitnet.so` |
| `convert-hf-to-gguf-bitnet.py` | converter | `utils/convert-hf-to-gguf-bitnet.py` | HF → GGUF conversion path |
| `tune_gemm_config.py` | tuner | `utils/tune_gemm_config.py` | Rewrites `gemm-config.h`, rebuilds `llama-bench` |

## CONVENTIONS
- Root `requirements.txt` is a compatibility shim for top-level Python scripts and llama.cpp requirement fragments; do not treat it like a normal app requirements file.
- Build defaults to Release and expects Clang/GCC-family toolchains for native code.
- CPU and GPU workflows are intentionally split: root + `src/` for CPU/native work, `gpu/` for CUDA/Python work.
- Validation is script-driven. There is no dedicated `tests/` tree or formal CI workflow checked in.
- Models, checkpoints, build outputs, logs, and generated binaries are intentionally ignored in `.gitignore`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not add packages directly to `requirements.txt`; keep top-level dependency policy centralized.
- Do not treat `preset_kernels/` as the source of truth; those files are generated/committed outputs.
- Do not file security vulnerabilities in public GitHub issues; follow `SECURITY.md`.
- Do not assume `3rdparty/llama.cpp` internals are editable first-party code in every checkout; it is a submodule boundary and may be absent.
- Do not put GPU-only commands in root guidance when `gpu/AGENTS.md` owns them.

## UNIQUE STYLES
- Root scripts are thin wrappers around built llama.cpp binaries or orchestration steps, not reusable Python packages.
- `setup_env.py --use-pretuned` copies per-model preset artifacts from `preset_kernels/` into `include/`.
- Many operational scripts hardcode CPU execution flags like `-ngl 0`, benchmark-style workflows, and local output directories.

## COMMANDS
```bash
pip install -r requirements.txt
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
python run_inference_server.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf
python utils/e2e_benchmark.py -m /path/to/model.gguf -n 128 -p 512 -t 2
```

## NOTES
- `src/README.md` and `gpu/README.md` are the authoritative local workflow docs for CPU and GPU specifics.
- There is no `.github/workflows/`, `Makefile`, or `pyproject.toml` in this checkout; prefer documented scripts over guessed commands.
- If work touches `include/gemm-config.h`, also inspect `utils/tune_gemm_config.py` and `src/README.md`.
- If work touches model conversion, check `utils/` first before editing root scripts.
