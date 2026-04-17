# GPU WORKFLOW KNOWLEDGE

## OVERVIEW
`gpu/` is a separate Python/CUDA inference stack with its own dependencies, conversion flow, runtime modules, and kernel build step.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| GPU setup and run flow | `README.md` | Authoritative local commands |
| End-to-end inference entrypoint | `generate.py` | Main GPU app |
| GPU checkpoint conversion | `convert_safetensors.py`, `convert_checkpoint.py` | Model prep path |
| GPU kernel validation | `test.py` | Manual benchmark/correctness script |
| CUDA shared library build | `bitnet_kernels/compile.sh` | Builds `libbitnet.so` |
| Runtime helpers | `model.py`, `tokenizer.py`, `sample_utils.py`, `stats.py` | Flat module tree, not a package |

## CONVENTIONS
- Use the GPU-specific environment from `gpu/README.md`; root `requirements.txt` does not replace `gpu/requirements.txt`.
- Kernel build documentation lives at the `gpu/` level; keep `bitnet_kernels/` details here unless the subtree grows much more complex.
- `test.py` is a manual CUDA benchmark/correctness script, not a pytest-style suite.
- This directory is a flat script/module tree with sibling imports, not a packaged Python library.

## ANTI-PATTERNS
- Do not route GPU dependency changes through root `requirements.txt`; use `gpu/requirements.txt`.
- Do not assume CPU wrapper flags like `-cnv` or `-ngl 0` apply to GPU flows the same way.
- Do not duplicate root bootstrap/build instructions here unless they are GPU-specific.
- Do not edit `bitnet_kernels/` artifacts without also checking `test.py` and the README build instructions.

## COMMANDS
```bash
conda create --name bitnet-gpu "python<3.13"
conda activate bitnet-gpu
pip install -r gpu/requirements.txt
bash gpu/bitnet_kernels/compile.sh
python gpu/test.py
python gpu/convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_state.pt --model_name 2B
python gpu/convert_checkpoint.py --input ./checkpoints/model_state.pt
python gpu/generate.py ./checkpoints/ --interactive --chat_format
```

## NOTES
- `bitnet_kernels/compile.sh` hardcodes the CUDA build of `libbitnet.so`; keep GPU architecture assumptions in mind when editing it.
- `test.py` loads `bitnet_kernels/libbitnet.so` directly via `ctypes`, so path/layout changes can break local validation quickly.
- If `gpu/AGENTS.md` ever becomes too dense with low-level CUDA notes, only then split out `gpu/bitnet_kernels/AGENTS.md`.
