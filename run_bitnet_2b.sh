#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

DEFAULT_MODEL_KEY="2b"
MODEL_KEY="$DEFAULT_MODEL_KEY"
MODEL_SETUP_REPO=""
AUTO_SETUP="${BITNET_AUTO_SETUP:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "Missing virtualenv Python at $PYTHON_BIN"
  print -u2 "Activate or recreate the local venv first."
  exit 1
fi

CONVERSATION_MODE=0
VERBOSE_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      if [[ $# -lt 2 ]]; then
        print -u2 "Missing value for $1"
        exit 1
      fi
      MODEL_KEY="$2"
      shift 2
      ;;
    --chat)
      CONVERSATION_MODE=1
      shift
      ;;
    --verbose|-v)
      VERBOSE_MODE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

case "$MODEL_KEY" in
  2b|bitnet-2b|bitnet)
    MODEL_NAME="BitNet 2B"
    MODEL_PATH="$ROOT_DIR/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    MODEL_SETUP_REPO="microsoft/BitNet-b1.58-2B-4T"
    ;;
  falcon3|falcon3-7b|falcon)
    MODEL_NAME="Falcon3 7B"
    MODEL_PATH="$ROOT_DIR/models/Falcon3-7B-1.58bit/ggml-model-i2_s.gguf"
    MODEL_SETUP_REPO="tiiuae/Falcon3-7B-1.58bit"
    ;;
  *)
    print -u2 "Unknown model option: $MODEL_KEY"
    print -u2 "Supported values: 2b, falcon3"
    exit 1
    ;;
esac

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ "$AUTO_SETUP" != "0" && -n "$MODEL_SETUP_REPO" ]]; then
    print -u2 "Missing model for $MODEL_NAME at $MODEL_PATH"
    print -u2 "Running setup_env.py to download and prepare it..."
    "$PYTHON_BIN" "$ROOT_DIR/setup_env.py" --hf-repo "$MODEL_SETUP_REPO" --model-dir "$ROOT_DIR/models" -q i2_s
  fi

  if [[ ! -f "$MODEL_PATH" ]]; then
    print -u2 "Missing model for $MODEL_NAME at $MODEL_PATH"
    print -u2 "Run the documented setup flow first."
    exit 1
  fi
fi

PROMPT="${1:-You are a helpful assistant}"
THREADS="${BITNET_THREADS:-2}"
N_PREDICT="${BITNET_N_PREDICT:-128}"
CTX_SIZE="${BITNET_CTX_SIZE:-2048}"

COMMAND=(
  "$PYTHON_BIN" "$ROOT_DIR/run_inference.py"
  -m "$MODEL_PATH"
  -p "$PROMPT"
  -t "$THREADS"
  -n "$N_PREDICT"
  -c "$CTX_SIZE"
)

if (( CONVERSATION_MODE )); then
  COMMAND+=( -cnv )
fi

if (( VERBOSE_MODE )); then
  exec "${COMMAND[@]}"
fi

exec "${COMMAND[@]}" 2>&1 | while IFS= read -r line; do
  case "$line" in
    "llm_load_vocab: special tokens cache size ="*| \
    "llm_load_vocab: token to piece cache size ="*)
      print -r -- "$line"
      ;;
    "llm_load_vocab:"*)
      ;;
    *)
      print -r -- "$line"
      ;;
  esac
done
