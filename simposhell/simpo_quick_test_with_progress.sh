#!/bin/bash
# SimPO Instruct Setup - Quick test with progress
# Validates full pipeline with a small number of prompts

set -x

echo "=========================================="
echo "🧪 SimPO Instruct Setup Quick Test"
echo "=========================================="

# Parameters
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"
OUTPUT_DIR="/home/ubuntu/rrhf/ultrafeedback_onpolicy_test"
REWARD_MODEL="llm-blender/PairRM"

# 5 seeds (one response per seed)
SEEDS=(13 21 42 79 100)

echo "📋 Test config:"
echo "   SFT model: $SFT_MODEL"
echo "   Output dir: $OUTPUT_DIR"
echo "   Reward model: $REWARD_MODEL"
echo "   Seeds: ${SEEDS[@]}"
echo "   Prompts: 5 (test)"
echo ""

# Create test directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "📝 Step 1: Generate 5 different responses"
echo "=========================================="
echo "📌 Generating 5 responses for 5 prompts"
echo ""

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    step_num=$((i + 1))
    echo "🔄 Progress: $step_num/5 - Generating responses for seed=$seed..."
    
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_decode_final.py \
        --model $SFT_MODEL \
        --temperature 0.8 \
        --top_p 0.95 \
        --max_tokens 512 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        --max_prompts 5
    
    if [ $? -eq 0 ]; then
        echo "✅ seed=$seed response generation complete"
        if [ -f "$OUTPUT_DIR/output_$seed.json" ]; then
            echo "📊 Generated $(jq length $OUTPUT_DIR/output_$seed.json) responses"
        fi
    else
        echo "❌ seed=$seed response generation failed"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "📝 Step 2: Post-process generated results"
echo "=========================================="
echo "📌 Merging 5 response files, filtering duplicates"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_post_process.py \
    --generation_file_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ Post-processing complete"
    if [ -f "$OUTPUT_DIR/all_outputs.json" ]; then
        echo "📊 Post-processed: $(jq length $OUTPUT_DIR/all_outputs.json) samples"
        echo "📄 Post-processing result preview:"
        head -n 5 $OUTPUT_DIR/all_outputs.json
    fi
else
    echo "❌ Post-processing failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "📝 Step 3: Preference annotation with PairRM reward model"
echo "=========================================="
echo "📌 Scoring 5 responses with PairRM, selecting best and worst"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_reward_annotate_no_datasets.py \
    --generation_file $OUTPUT_DIR/all_outputs.json \
    --reward_model $REWARD_MODEL \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ Reward model annotation complete"
    if [ -f "$OUTPUT_DIR/all_outputs_bin.json" ]; then
        echo "📊 Annotated: $(jq length $OUTPUT_DIR/all_outputs_bin.json) preference pairs"
        echo "📄 Annotation result preview:"
        head -n 5 $OUTPUT_DIR/all_outputs_bin.json
    fi
else
    echo "❌ Reward model annotation failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "📝 Step 4: Convert to DPO training format"
echo "=========================================="
echo "📌 Converting to DPO JSONL format"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/convert_to_dpo_format.py \
    --input_file $OUTPUT_DIR/all_outputs_bin.json \
    --output_file $OUTPUT_DIR/test_dpo.jsonl

if [ $? -eq 0 ]; then
    echo "✅ DPO format conversion complete"
    if [ -f "$OUTPUT_DIR/test_dpo.jsonl" ]; then
        echo "📊 DPO data: $(wc -l < $OUTPUT_DIR/test_dpo.jsonl) preference pairs"
        echo "📄 DPO data sample:"
        head -n 2 $OUTPUT_DIR/test_dpo.jsonl
    fi
else
    echo "❌ DPO format conversion failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "🎉 SimPO Quick Test Complete!"
echo "=========================================="
echo "📁 Test results saved in: $OUTPUT_DIR"
echo "📊 Final file listing:"
ls -la $OUTPUT_DIR/
echo ""
echo "✅ Quick test passed! Ready to run full pipeline."
