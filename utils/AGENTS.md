# UTILITIES WORKFLOW KNOWLEDGE

## OVERVIEW
`utils/` owns the operational scripts: conversion, benchmarking, perplexity evaluation, code generation, tuning, and shell-based performance harnesses.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| HF / GGUF conversion | `convert-hf-to-gguf-bitnet.py`, `convert-ms-to-gguf-bitnet.py`, `convert.py` | Main conversion cluster |
| Helper conversion pipeline | `convert-helper-bitnet.py`, `preprocess-huggingface-bitnet.py` | Pre/post-processing wrappers |
| Benchmarks | `e2e_benchmark.py`, `test_power.sh`, `test_gemm_kernel.sh` | Perf-oriented validation |
| Perplexity evaluation | `test_perplexity.py` | Dataset-based CPU validation |
| Kernel/code generation | `codegen_tl1.py`, `codegen_tl2.py` | Produces preset kernel outputs |
| Config tuning | `tune_gemm_config.py` | Rewrites `include/gemm-config.h` and rebuilds |
| Synthetic model generation | `generate-dummy-bitnet-model.py` | Experimental benchmarking path |

## CONVENTIONS
- Most files here are standalone CLIs; treat them as task-specific tools, not a cohesive library.
- Benchmarks and tests are mixed intentionally: `test_*` names often mean performance harnesses, not unit tests.
- `test_perplexity.py` expects datasets as `data/<dataset>/test.txt` and writes outputs under `perplexity_results/`.
- `tune_gemm_config.py` mutates `include/gemm-config.h`, rebuilds `llama-bench`, and writes results under `stats/`.
- `codegen_tl1.py` and `codegen_tl2.py` are the source-of-truth for generated preset kernel artifacts.

## ANTI-PATTERNS
- Do not hand-edit generated tokenizer/hash sections or generated preset outputs when the generator script is the real source of truth.
- Do not assume scripts here are safe read-only helpers; several mutate config files, create models, rebuild binaries, or write result directories.
- Do not document these as formal CI tests; this directory is mostly local tooling and benchmark automation.
- Do not duplicate CPU kernel implementation rules here; use this file for operational workflows around that code.

## COMMANDS
```bash
python utils/convert-hf-to-gguf-bitnet.py /path/to/model_dir --outtype f32
python utils/e2e_benchmark.py -m /path/to/model.gguf -n 128 -p 512 -t 2
python utils/test_perplexity.py --help
python utils/tune_gemm_config.py --help
python utils/codegen_tl1.py --model bitnet_b1_58-large --BM 256,128,256 --BK 128,64,128 --bm 32,64,32
python utils/codegen_tl2.py --model bitnet_b1_58-large --BM 256,128,256 --BK 96,192,96 --bm 32,32,32
bash utils/test_gemm_kernel.sh
```

## NOTES
- `convert.py` and `convert-ms-to-gguf-bitnet.py` contain unresolved TODO/FIXME hotspots; treat model-detection logic there as fragile.
- `test_power.sh` assumes Linux-specific tooling; do not treat it as portable validation.
- Changes that affect `preset_kernels/` should usually start from the codegen scripts or `setup_env.py --use-pretuned`, not by editing committed outputs directly.
