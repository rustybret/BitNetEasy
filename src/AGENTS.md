# CPU KERNEL KNOWLEDGE

## OVERVIEW
`src/` owns the native CPU BitNet kernels and their CMake integration with llama.cpp.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main CPU kernel changes | `ggml-bitnet-mad.cpp` | Largest native hotspot |
| LUT path changes | `ggml-bitnet-lut.cpp` | Alternate native kernel path |
| Native build wiring | `CMakeLists.txt` | Compiler gate and include dependency |
| Tuning knobs | `../include/gemm-config.h` | Shared config file used by native path |
| CPU optimization context | `README.md` | Performance rationale and manual commands |

## CONVENTIONS
- Native CPU work assumes Clang or GCC; `src/CMakeLists.txt` fails fast on other compilers.
- `include/gemm-config.h` is part of the effective `src/` workflow even though it lives outside this directory.
- Performance-sensitive edits should keep `src/README.md` in sync when they change tuning guidance or recommended settings.
- This directory is tightly coupled to the llama.cpp submodule layout through include paths and ggml integration.

## ANTI-PATTERNS
- Do not change kernel code without checking whether `gemm-config.h` assumptions still hold.
- Do not document or route tests here as if a native unit-test suite exists; validation is driven from `utils/` scripts and benchmark binaries.
- Do not duplicate GPU workflow notes here.
- Do not treat `include/` as an independent child scope when changes are really part of CPU kernel tuning.

## COMMANDS
```bash
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
python setup_env.py --quant-embd
build/bin/llama-quantize --token-embedding-type Q6_K models/BitNet-b1.58-2B-4T/ggml-model-f32.gguf models/BitNet-b1.58-2B-4T/ggml-model-i2_s-embed-q6_k.gguf I2_S 1 1
```

## NOTES
- `README.md` documents CPU performance data, architecture-specific tuning, and embedding quantization recommendations.
- `ggml-bitnet-mad.cpp` is the highest-risk file here because it concentrates the core CPU kernel implementation.
- Review neighboring `include/ggml-bitnet.h` warnings before changing supported quantization assumptions.
