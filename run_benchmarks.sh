#!/bin/bash
# Run MPS/Metal benchmarks comparing CPU vs GPU performance

set -e

MODEL="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
BENCH="./build/bin/llama-bench"
OUTDIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "BitNet MPS/Metal Shader Acceleration Benchmark"
echo "============================================================"
echo "Model: $MODEL"
echo "Output directory: $OUTDIR"
echo ""

# Configuration
PROMPT_TOKENS=128
GEN_TOKENS=64
THREADS=8
REPETITIONS=3

echo "Running benchmarks with:"
echo "  Prompt tokens: $PROMPT_TOKENS"
echo "  Generation tokens: $GEN_TOKENS"
echo "  Threads: $THREADS"
echo "  Repetitions: $REPETITIONS"
echo ""

# 1. CPU Only (0 GPU layers)
echo "============================================================"
echo "1. CPU ONLY (ngl=0)"
echo "============================================================"
$BENCH -m "$MODEL" -ngl 0 -p $PROMPT_TOKENS -n $GEN_TOKENS -t $THREADS -r $REPETITIONS -o md 2>&1 | tee "$OUTDIR/cpu_ngl0.md"
echo ""

# 2. Metal GPU - All layers (31 layers for this model)
echo "============================================================"
echo "2. METAL GPU - All Layers (ngl=31)"
echo "============================================================"
$BENCH -m "$MODEL" -ngl 31 -p $PROMPT_TOKENS -n $GEN_TOKENS -t $THREADS -r $REPETITIONS -o md 2>&1 | tee "$OUTDIR/metal_ngl31.md"
echo ""

# 3. Partial offload - 15 layers
echo "============================================================"
echo "3. METAL GPU - Partial (ngl=15)"
echo "============================================================"
$BENCH -m "$MODEL" -ngl 15 -p $PROMPT_TOKENS -n $GEN_TOKENS -t $THREADS -r $REPETITIONS -o md 2>&1 | tee "$OUTDIR/metal_ngl15.md"
echo ""

echo "============================================================"
echo "Benchmark Complete!"
echo "Results saved to: $OUTDIR/"
echo "============================================================"

# Extract and display key results
echo ""
echo "SUMMARY OF RESULTS:"
echo "-------------------"
for f in "$OUTDIR"/*.md; do
    echo ""
    echo "File: $(basename $f)"
    grep "t/s" "$f" | head -5
done
