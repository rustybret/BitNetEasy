# Local CPU setup notes

Verified on this machine with:

- Apple Silicon macOS
- Homebrew `cmake` 4.3.1
- Homebrew Python 3.11 in repo-local `.venv`
- Apple clang 17 for the successful native build

## What worked

The verified default model path is:

```text
models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

An additional supported model option is:

```text
models/Falcon3-7B-1.58bit/ggml-model-i2_s.gguf
```

The native binaries are in:

```text
build/bin/
```

## Activate the environment

```bash
source .venv/bin/activate
```

## Run CPU inference

```bash
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```

Or point `run_inference.py` at the Falcon3 GGUF once it is downloaded locally:

```bash
python run_inference.py -m models/Falcon3-7B-1.58bit/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```

Or use the local helper for a one-shot prompt that exits when generation finishes:

```bash
./run_bitnet_2b.sh "You are a helpful assistant"
```

The helper now supports model aliases:

```bash
./run_bitnet_2b.sh --model 2b "You are a helpful assistant"
./run_bitnet_2b.sh --model falcon3 "You are a helpful assistant"
```

If the selected 2B or Falcon3 model is missing, the helper now calls `setup_env.py` automatically to download and prepare it.

By default, the helper suppresses the repetitive `llm_load_vocab` tokenizer/EOG warnings so the startup output is easier to read. If you want the full raw loader output, add `--verbose`:

```bash
./run_bitnet_2b.sh --model falcon3 --verbose "You are a helpful assistant"
```

If you want the old interactive chat-style behavior, opt into it explicitly:

```bash
./run_bitnet_2b.sh --chat "You are a helpful assistant"
```

## Why this uses the GGUF repo

The raw Hugging Face checkpoint repo `microsoft/BitNet-b1.58-2B-4T` downloaded successfully, but this checkout's converter rejects its architecture string `BitNetForCausalLM`. The working path here is the README-supported GGUF repo `microsoft/BitNet-b1.58-2B-4T-gguf`.

For Falcon3, the upstream GGUF repo to download is:

```bash
hf download tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF --local-dir models/Falcon3-7B-1.58bit
```

After the download, the helper can target it with `--model falcon3`.

## Build note

Homebrew LLVM 18 failed against the current macOS SDK headers during compilation. The successful build on this machine used Apple clang instead.

## Runtime note

Warnings like these are noisy, but they are not the reason the helper appeared to stall:

- `missing pre-tokenizer type, using: 'default'`
- `GENERATION QUALITY WILL BE DEGRADED!`
- `control token ... is not marked as EOG`

Those come from the GGUF model metadata/tokenizer handling in this build. The confusing behavior was that the first version of `run_bitnet_2b.sh` always passed `-cnv`, which made `llama-cli` enter interactive conversation mode after startup. The helper now defaults to one-shot generation and only enters chat mode when run with `--chat`.

## Smoke test

To check whether a local model can produce one minimally coherent response to a trivial prompt, run:

```bash
python utils/smoke_test.py --model 2b
```

You can also point it at the Falcon3 alias once that GGUF is downloaded:

```bash
python utils/smoke_test.py --model falcon3
```

## Local generated files

Setup generated these untracked files in the repo:

- `include/bitnet-lut-kernels.h`
- `include/kernel_config.ini`

These are normal outputs of the setup/codegen flow.
