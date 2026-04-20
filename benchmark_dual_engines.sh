#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results_$(date +%Y%m%d_%H%M%S)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Dual Engine Benchmark for 1-bit LLMs"
echo "========================================"

mkdir -p "$RESULTS_DIR"

GGML_BENCH="/tmp/llama-build/bin/llama-bench"
BITNET_BENCH="${SCRIPT_DIR}/3rdparty/bitnet-official/build/bin/llama-bench"

if [ ! -f "$GGML_BENCH" ]; then
    echo -e "${YELLOW}Building ggml-org engine...${NC}"
    mkdir -p /tmp/llama-build
    cd /tmp/llama-build
    cmake "${SCRIPT_DIR}/3rdparty/llama.cpp" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(sysctl -n hw.ncpu) llama-bench llama-cli
fi

echo ""
echo -e "${GREEN}=== Q1_0 Models (ggml-org/llama.cpp) ===${NC}"

Q10_MODELS=(
    "${SCRIPT_DIR}/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf"
    "${SCRIPT_DIR}/models/Bonsai-4B/Bonsai-4B-Q1_0.gguf"
)

for model in "${Q10_MODELS[@]}"; do
    if [ -f "$model" ]; then
        model_name=$(basename "$model" .gguf)
        echo "Benchmarking: $model_name"
        "$GGML_BENCH" -m "$model" -ngl 99 -p 128,512 -n 64 -t 8 2>&1 | \
            tee "${RESULTS_DIR}/${model_name}_q10_metal.txt"
    fi
done

echo ""
echo -e "${GREEN}=== I2_S Models (Microsoft BitNet) ===${NC}"

I2S_MODELS=(
    "${SCRIPT_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    "${SCRIPT_DIR}/models/Falcon3-7B-1.58bit/ggml-model-i2_s.gguf"
)

if [ -f "$BITNET_BENCH" ]; then
    for model in "${I2S_MODELS[@]}"; do
        if [ -f "$model" ]; then
            model_name=$(basename "$model" .gguf)
            echo "Benchmarking: $model_name"
            "$BITNET_BENCH" -m "$model" -ngl 31 -p 128,512 -n 64 -t 8 2>&1 | \
                tee "${RESULTS_DIR}/${model_name}_i2s_metal.txt"
        fi
    done
else
    echo -e "${RED}BitNet engine not built. Run setup_bitnet_engine.sh first:${NC}"
    echo "  ./setup_bitnet_engine.sh"
fi

echo ""
echo "========================================"
echo "Benchmark Summary"
echo "========================================"
echo ""

echo "| Model | Format | Size | pp512 (t/s) | tg64 (t/s) |"
echo "|-------|--------|------|-------------|------------|"

for result in "${RESULTS_DIR}"/*.txt; do
    if [ -f "$result" ]; then
        pp512=$(grep "pp512" "$result" | tail -1 | awk '{print $NF}' | sed 's/±.*//')
        tg64=$(grep "tg64" "$result" | tail -1 | awk '{print $NF}' | sed 's/±.*//')
        model_name=$(basename "$result" .txt | sed 's/_q10\|_i2s//')
        
        if [ -n "$pp512" ] && [ -n "$tg64" ]; then
            if echo "$result" | grep -q "q10"; then
                format="Q1_0"
            else
                format="I2_S"
            fi
            echo "| $model_name | $format | - | $pp512 | $tg64 |"
        fi
    fi
done

echo ""
echo "Results saved to: $RESULTS_DIR"
