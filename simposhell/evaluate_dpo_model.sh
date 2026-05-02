#!/bin/bash
# DPO model evaluation script - AlpacaEval, Arena-Hard, MT-Bench
# Usage: bash evaluate_dpo_model.sh [model_path]

set -x

# Configuration parameters
MODEL_PATH=${1:-"./checkpoint/llama3-8b-dpo-rrhf"}  # Default DPO model path
OUTPUT_DIR="./evaluation_results"
EVAL_DATASETS_DIR="/home/ubuntu/datasets/eval_dataset"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting DPO model evaluation: $MODEL_PATH"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please ensure the model has been trained or provide the correct model path"
    exit 1
fi

# Check if evaluation datasets exist
echo "=== Checking evaluation datasets ==="

if [ ! -f "$EVAL_DATASETS_DIR/alpacaeval_data.json" ]; then
    echo "Error: AlpacaEval dataset does not exist: $EVAL_DATASETS_DIR/alpacaeval_data.json"
    exit 1
fi

if [ ! -f "$EVAL_DATASETS_DIR/arena_hard_data.json" ]; then
    echo "Error: Arena-Hard dataset does not exist: $EVAL_DATASETS_DIR/arena_hard_data.json"
    exit 1
fi

if [ ! -f "$EVAL_DATASETS_DIR/mt_bench_data.json" ]; then
    echo "Error: MT-Bench dataset does not exist: $EVAL_DATASETS_DIR/mt_bench_data.json"
    exit 1
fi

echo "All evaluation datasets checked!"

# 2. Generate model outputs
echo "=== Generating model outputs ==="

# AlpacaEval output
echo "Generating AlpacaEval output..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/alpacaeval_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/alpacaeval_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

# Arena-Hard output
echo "Generating Arena-Hard output..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/arena_hard_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/arena_hard_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

# MT-Bench output
echo "Generating MT-Bench output..."
python -m openrlhf.cli.batch_inference \
    --pretrain "$MODEL_PATH" \
    --dataset "$EVAL_DATASETS_DIR/mt_bench_data.json" \
    --input_key "instruction" \
    --output_path "$OUTPUT_DIR/mt_bench_outputs.json" \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --micro_batch_size 8 \
    --bf16 \
    --flash_attn

echo "Model output generation complete!"

# 3. Run evaluation
echo "=== Running Evaluation ==="

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set"
    echo "Please set: export OPENAI_API_KEY=your_api_key"
    echo "Or modify the API key in the script"
    echo "Skipping evaluations that require API..."
else
    # AlpacaEval evaluation
    echo "Running AlpacaEval evaluation..."
    alpaca_eval \
        --model_outputs "$OUTPUT_DIR/alpacaeval_outputs.json" \
        --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
        --reference_outputs "$EVAL_DATASETS_DIR/alpacaeval_data.json" \
        --output_path "$OUTPUT_DIR/alpacaeval_results"
    
    # Arena-Hard evaluation
    echo "Running Arena-Hard evaluation..."
    python -m arena_hard.evaluate \
        --model_outputs "$OUTPUT_DIR/arena_hard_outputs.json" \
        --reference_outputs "$EVAL_DATASETS_DIR/arena_hard_data.json" \
        --output_path "$OUTPUT_DIR/arena_hard_results"
    
    # MT-Bench evaluation
    echo "Running MT-Bench evaluation..."
    python -m mt_bench.evaluate \
        --model_outputs "$OUTPUT_DIR/mt_bench_outputs.json" \
        --reference_outputs "$EVAL_DATASETS_DIR/mt_bench_data.json" \
        --output_path "$OUTPUT_DIR/mt_bench_results"
fi

echo ""
echo "=== Evaluation Complete ==="
echo "All results saved in: $OUTPUT_DIR"
echo ""
echo "Result files:"
echo "- AlpacaEval: $OUTPUT_DIR/alpacaeval_results"
echo "- Arena-Hard: $OUTPUT_DIR/arena_hard_results"
echo "- MT-Bench: $OUTPUT_DIR/mt_bench_results"
echo ""
echo "Model output files:"
echo "- AlpacaEval: $OUTPUT_DIR/alpacaeval_outputs.json"
echo "- Arena-Hard: $OUTPUT_DIR/arena_hard_outputs.json"
echo "- MT-Bench: $OUTPUT_DIR/mt_bench_outputs.json"
