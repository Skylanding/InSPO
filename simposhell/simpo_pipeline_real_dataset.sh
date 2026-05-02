#!/bin/bash
# SimPO Instruct Setup - Full UltraFeedback dataset
# Large-scale preference dataset generation (61,000+ prompts)

set -x

echo "=========================================="
echo "🚀 SimPO Instruct Setup - Full UltraFeedback Dataset"
echo "=========================================="

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Parameters
SFT_MODEL="/home/ubuntu/basemodels/llama3/llama3-8b-instruct"
OUTPUT_DIR="/home/ubuntu/rrhf/ultrafeedback_onpolicy_full"
REWARD_MODEL="llm-blender/PairRM"
DATASET_DIR="HuggingFaceH4/ultrafeedback_binarized"

# 5 seeds for 5 different responses
9
SEEDS=(13 21 42 79 100)

echo "📋 Config:"
echo "   SFT model: $SFT_MODEL"
echo "   Output dir: $OUTPUT_DIR"
echo "   Reward model: $REWARD_MODEL"
echo "   Dataset: $DATASET_DIR (61,000+ prompts)"
echo "   Seeds: ${SEEDS[@]}"
echo "   Expected output: ~300,000 preference pairs"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "📝 Step 1: Generate 5 different responses with SFT model"
echo "=========================================="
echo "📌 Generating 5 responses for each UltraFeedback prompt"
echo "📌 Params: temperature=0.8, max_tokens=4096"
echo "📌 Estimated time: ~2-3 hours per seed (depends on GPU)"
echo ""

# Generate responses for each seed
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    step_num=$((i + 1))
    echo "🔄 Progress: $step_num/5 - Generating responses for seed=$seed..."
    echo "⏰ Estimated time: 2-3 hours"
    
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_decode_hf_hub.py \
        --model $SFT_MODEL \
        --temperature 0.8 \
        --top_p 0.95 \
        --max_tokens 4096 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        --gpu_ids "0,1,2,3" \
        --batch_size 1
    
    if [ $? -eq 0 ]; then
        echo "✅ seed=$seed response generation complete"
        if [ -f "$OUTPUT_DIR/output_$seed.json" ]; then
            echo "📊 Responses generated: $(jq length $OUTPUT_DIR/output_$seed.json)"
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
echo "📌 Reward model: $REWARD_MODEL"
echo "📌 Estimated time: 1-2 hours"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/simpo_reward_annotate_no_datasets.py \
    --generation_file $OUTPUT_DIR/all_outputs.json \
    --reward_model $REWARD_MODEL \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ Reward model annotation complete"
    if [ -f "$OUTPUT_DIR/all_outputs_bin.json" ]; then
        echo "📊 Annotated: $(jq length $OUTPUT_DIR/all_outputs_bin.json) preference pairs"
    fi
else
    echo "❌ Reward model annotation failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "📝 Step 4: Convert to SimPO and DPO format"
echo "=========================================="
echo "📌 Converting to SimPO and DPO training format"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh && conda activate handbook && python /home/ubuntu/Open/rrhf/convert_to_simpo_format.py \
    --input_file $OUTPUT_DIR/all_outputs_bin.json \
    --simpo_output $OUTPUT_DIR/ultrafeedback_onpolicy_simpo.json \
    --dpo_output $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl

if [ $? -eq 0 ]; then
    echo "✅ Format conversion complete"
    if [ -f "$OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl" ]; then
        echo "📊 DPO data: $(wc -l < $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl) preference pairs"
    fi
else
    echo "❌ Format conversion failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "🎉 SimPO Instruct Setup Complete!"
echo "=========================================="
echo "📁 Output files:"
echo "   SimPO data: $OUTPUT_DIR/ultrafeedback_onpolicy_simpo.json"
echo "   DPO data: $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl"
echo "   HuggingFace format: $OUTPUT_DIR/"
echo ""

# Show result statistics
if [ -f "$OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl" ]; then
    echo "📊 Final stats:"
    echo "   DPO preference pairs: $(wc -l < $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl)"
    echo "   Data size: $(du -h $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl | cut -f1)"
    echo ""
    echo "📄 Data sample preview:"
    head -n 2 $OUTPUT_DIR/ultrafeedback_onpolicy_dpo.jsonl
    echo ""
fi

echo "✅ All steps complete! DPO data is ready for training."
echo "💡 Tip: This dataset is much larger than previous test runs, suitable for full SimPO training."