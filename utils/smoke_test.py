#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT = "The capital of France is"
DEFAULT_EXPECT = "Paris"
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ALIASES = {
    "2b": ROOT_DIR / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
    "falcon3": ROOT_DIR / "models/Falcon3-7B-1.58bit/ggml-model-i2_s.gguf",
}
STOP_MARKERS = (
    "llama_perf_sampler_print:",
    "llama_perf_context_print:",
    "ggml_metal_free:",
    "main: ",
    "system_info:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a local BitNet model response")
    parser.add_argument(
        "-m",
        "--model",
        default="2b",
        help="Model alias (2b, falcon3) or explicit GGUF path",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "-e",
        "--expect",
        default=DEFAULT_EXPECT,
        help="Case-insensitive keyword expected in the generated response",
    )
    parser.add_argument(
        "-n",
        "--n-predict",
        type=int,
        default=8,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=2,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "-c",
        "--ctx-size",
        type=int,
        default=2048,
        help="Context size",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature; defaults to greedy decoding",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    alias_path = DEFAULT_MODEL_ALIASES.get(model_arg.lower())
    if alias_path is not None:
        return alias_path
    return Path(model_arg).expanduser()


def build_command(args: argparse.Namespace, model_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT_DIR / "run_inference.py"),
        "-m",
        str(model_path),
        "-p",
        args.prompt,
        "-n",
        str(args.n_predict),
        "-t",
        str(args.threads),
        "-c",
        str(args.ctx_size),
        "-temp",
        str(args.temp),
    ]


def extract_response(output: str, prompt: str) -> str:
    normalized = output.replace("\r", "")
    prompt_index = normalized.rfind(prompt)
    tail = normalized[prompt_index + len(prompt):] if prompt_index != -1 else normalized

    cut_points = [len(tail)]
    for marker in STOP_MARKERS:
        marker_index = tail.find(marker)
        if marker_index != -1:
            cut_points.append(marker_index)

    candidate = tail[: min(cut_points)]
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def has_excessive_repetition(response: str) -> bool:
    words = re.findall(r"[A-Za-z']+", response.lower())
    if not words:
        return False

    streak = 1
    for previous, current in zip(words, words[1:]):
        if previous == current:
            streak += 1
            if streak >= 4:
                return True
        else:
            streak = 1
    return False


def validate_response(response: str, expected_keyword: str) -> list[str]:
    failures: list[str] = []

    if not response:
        failures.append("generated response was empty")
        return failures

    if len(response) < 2 or len(response) > 120:
        failures.append(f"response length {len(response)} outside 2-120 characters")

    if expected_keyword.lower() not in response.lower():
        failures.append(f"response did not contain expected keyword '{expected_keyword}'")

    if has_excessive_repetition(response):
        failures.append("response repeated the same word 4+ times consecutively")

    if any(ord(ch) < 32 and ch not in "\t\n" for ch in response):
        failures.append("response contained non-printable control characters")

    return failures


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    if not model_path.exists():
        print(f"SMOKE FAIL: model file not found: {model_path}")
        return 1

    command = build_command(args, model_path)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"SMOKE FAIL: inference timed out after {args.timeout}s")
        return 1

    stdout_output = result.stdout or ""
    stderr_output = result.stderr or ""
    response_source = stdout_output if stdout_output else stdout_output + ("\n" + stderr_output if stderr_output else "")
    response = extract_response(response_source, args.prompt)

    failures: list[str] = []
    if result.returncode != 0:
        failures.append(f"inference exited with status {result.returncode}")

    failures.extend(validate_response(response, args.expect))

    if failures:
        print("SMOKE FAIL")
        print(f"Model: {model_path}")
        print(f"Prompt: {args.prompt}")
        print(f"Extracted response: {response or '<empty>'}")
        for failure in failures:
            print(f"- {failure}")
        if stderr_output:
            print("stderr tail:")
            print(stderr_output[-500:])
        return 1

    print("SMOKE PASS")
    print(f"Model: {model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Extracted response: {response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
