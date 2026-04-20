#!/usr/bin/env python3
"""
MPS/Metal vs CPU Benchmark Comparison for BitNet models.
Runs benchmarks with different GPU layer configurations and compares results.
"""

import subprocess
import sys
import json
import os
from datetime import datetime

MODEL = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
BENCH_BIN = "./build/bin/llama-bench"
RESULTS_FILE = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def run_benchmark(ngl, name):
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {name} (ngl={ngl})")
    print(f"{'='*60}")

    cmd = [
        BENCH_BIN,
        '-m', MODEL,
        '-ngl', str(ngl),
        '-p', '128',
        '-n', '64',
        '-t', '8',
        '-r', '3',
        '-o', 'json'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        # Parse JSON output (find the JSON array in the output)
        output = result.stdout

        # Find the JSON data (it's usually at the end after all the loading messages)
        lines = output.strip().split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            if line.strip().startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.strip().endswith(']') and in_json:
                break

        if json_lines:
            json_str = '\n'.join(json_lines)
            data = json.loads(json_str)
            return data
        else:
            print(f"Warning: No JSON output found for {name}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {name}")
        return None
    except Exception as e:
        print(f"Error running benchmark for {name}: {e}")
        return None

def main():
    """Run all benchmarks and save results."""
    print("="*60)
    print("BitNet MPS/Metal vs CPU Benchmark")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("")

    # Check prerequisites
    if not os.path.exists(MODEL):
        print(f"Error: Model not found at {MODEL}")
        sys.exit(1)

    if not os.path.exists(BENCH_BIN):
        print(f"Error: Benchmark binary not found at {BENCH_BIN}")
        sys.exit(1)

    # Define benchmark configurations
    configs = [
        (0, "CPU Only"),
        (10, "Metal GPU - 10 layers"),
        (20, "Metal GPU - 20 layers"),
        (31, "Metal GPU - All 31 layers"),
    ]

    all_results = []

    for ngl, name in configs:
        result = run_benchmark(ngl, name)
        if result:
            all_results.extend(result)

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"{'='*60}")

    # Print summary
    if all_results:
        print("\nSummary:")
        print("-" * 80)
        print(f"{'Config':<30} {'Test':<10} {'Tokens/sec':<15} {'Backend':<10}")
        print("-" * 80)

        for r in all_results:
            config_name = f"ngl={r.get('n_gpu_layers', 'N/A')}"
            test_type = f"pp {r.get('n_prompt', 0)}" if r.get('n_prompt', 0) > 0 else f"tg {r.get('n_gen', 0)}"
            tps = f"{r.get('avg_ts', 0):.2f}"
            backend = r.get('backend', 'N/A')
            print(f"{config_name:<30} {test_type:<10} {tps:<15} {backend:<10}")

if __name__ == "__main__":
    main()
