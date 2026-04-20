#!/bin/bash
# MPS/Metal Shader Acceleration Benchmark Script
# Compares CPU vs Metal GPU performance on Apple Silicon

set -e

MODEL="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
BENCH_BIN="./build/bin/llama-bench"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "BitNet MPS/Metal Benchmark"
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo "============================================"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

# Check if benchmark binary exists
if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: Benchmark binary not found at $BENCH_BIN"
    exit 1
fi

# Function to run benchmark
run_benchmark() {
    local ngl=$1
    local name=$2
    local output_file="$RESULTS_DIR/${TIMESTAMP}_${name}.md"

    echo ""
    echo "Running benchmark: $name (ngl=$ngl)"
    echo "Output: $output_file"

    $BENCH_BIN \
        -m "$MODEL" \
        -ngl "$ngl" \
        -p 128 \
        -n 64 \
        -t 8 \
        -r 3 \
        -o md \
        2>&1 | tee "$output_file"

    echo "Completed: $name"
}

# 1. CPU-only benchmark (no GPU layers)
echo ""
echo "=== Test 1: CPU Only (0 GPU layers) ==="
run_benchmark 0 "cpu_only"

# 2. Metal GPU benchmark (all layers)
echo ""
echo "=== Test 2: Metal GPU (all 31 layers) ==="
run_benchmark 31 "metal_full"

# 3. Partial GPU offload benchmarks
echo ""
echo "=== Test 3: Partial GPU offload ==="
run_benchmark 10 "metal_10_layers"
run_benchmark 20 "metal_20_layers"

# 4. Summary
echo ""
echo "============================================"
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_DIR/"
echo "============================================"
echo ""
echo "Summary of results:"
ls -la "$RESULTS_DIR/${TIMESTAMP}"_*.md
