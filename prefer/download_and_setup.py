#!/usr/bin/env python3
"""
Download SFT model and UltraFeedback dataset, convert to OpenRLHF format.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_sft_model(model_name="OpenRLHF/Llama-3-8b-sft-mixture", save_dir="./models"):
    """Download SFT model to save_dir."""
    print(f"Downloading SFT model: {model_name}")

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"Princeton_{model_name.split('/')[-1]}")

    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return model_path

    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)

        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.save_pretrained(model_path)

        print(f"SFT model saved to: {model_path}")
        return model_path

    except Exception as e:
        print(f"SFT model download failed: {e}")
        return None


def download_ultrafeedback_dataset(dataset_name="princeton-nlp/llama3-ultrafeedback-armorm", save_dir="./data"):
    """Download Llama3-UltraFeedback-ArmoRM dataset to save_dir."""
    print(f"Downloading dataset: {dataset_name}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"Dataset downloaded: {len(dataset)} samples")

        dataset_path = os.path.join(save_dir, "Princeton_llama3_ultrafeedback_armorm")
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to: {dataset_path}")

        return dataset_path

    except Exception as e:
        print(f"Dataset download failed: {e}")
        return None


def convert_to_openrlhf_format(dataset_path, output_file, max_samples=None):
    """Convert dataset at dataset_path to OpenRLHF DPO JSONL format."""
    print("Converting to OpenRLHF DPO format...")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)

        dpo_data = []
        sample_count = 0

        for sample in dataset:
            if max_samples and sample_count >= max_samples:
                break

            if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
                prompt = sample['prompt']

                chosen_response = ""
                rejected_response = ""

                if isinstance(sample['chosen'], list):
                    for msg in sample['chosen']:
                        if msg.get('role') == 'assistant':
                            chosen_response = msg.get('content', '')
                            break
                else:
                    chosen_response = str(sample['chosen'])

                if isinstance(sample['rejected'], list):
                    for msg in sample['rejected']:
                        if msg.get('role') == 'assistant':
                            rejected_response = msg.get('content', '')
                            break
                else:
                    rejected_response = str(sample['rejected'])

                if chosen_response and rejected_response and chosen_response != rejected_response:
                    dpo_sample = {
                        'prompt': prompt,
                        'chosen': chosen_response,
                        'rejected': rejected_response
                    }
                    dpo_data.append(dpo_sample)
                    sample_count += 1
                else:
                    print(f"Skipping invalid sample {sample_count}: chosen/rejected identical or empty")
                    continue

            else:
                print(f"Skipping sample with unexpected keys: {sample.keys()}")
                continue

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dpo_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Converted {len(dpo_data)} samples, saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Format conversion failed: {e}")
        return False


def create_config_file(model_path, dataset_path, output_dir="./config"):
    """Create OpenRLHF config JSON file."""
    print("Creating OpenRLHF config...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "model": {
            "pretrain": model_path,
            "ref_pretrain": model_path
        },
        "dataset": {
            "path": dataset_path,
            "split": "train",
            "max_samples": 100000
        },
        "training": {
            "train_batch_size": 128,
            "micro_train_batch_size": 4,
            "max_epochs": 3,
            "max_len": 1024,
            "learning_rate": 5e-7,
            "beta": 0.1
        },
        "optimization": {
            "zero_stage": 2,
            "adam_offload": True,
            "flash_attn": True,
            "gradient_checkpointing": True
        },
        "logging": {
            "save_path": "./checkpoint/Princeton_llama3-8b-dpo-ultrafeedback-armorm",
            "save_steps": 500,
            "logging_steps": 10,
            "eval_steps": 200,
            "use_wandb": True,
            "wandb_project": "princeton_dpo",
            "wandb_run_name": "Princeton_llama3-8b-dpo-ultrafeedback-armorm"
        },
        "gpu": {
            "cuda_visible_devices": "0,1,2,3"
        }
    }
    
    config_file = os.path.join(output_dir, "Princeton_openrlhf_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Config saved to: {config_file}")
    return config_file


def main():
    parser = argparse.ArgumentParser(description="Download and configure SFT model and dataset")
    parser.add_argument("--model_name", type=str, default="OpenRLHF/Llama-3-8b-sft-mixture",
                       help="SFT model name")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/llama3-ultrafeedback-armorm",
                       help="Dataset name")
    parser.add_argument("--model_dir", type=str, default="./models",
                       help="Model save directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Data save directory")
    parser.add_argument("--config_dir", type=str, default="./config",
                       help="Config save directory")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Max number of samples")
    parser.add_argument("--skip_model", action="store_true",
                       help="Skip model download")
    parser.add_argument("--skip_dataset", action="store_true",
                       help="Skip dataset download")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SFT Model & Dataset Download Setup")
    print("=" * 60)
    
    model_path = None
    if not args.skip_model:
        model_path = download_sft_model(args.model_name, args.model_dir)
        if not model_path:
            print("SFT model download failed, exiting")
            return

    dataset_path = None
    if not args.skip_dataset:
        dataset_path = download_ultrafeedback_dataset(args.dataset_name, args.data_dir)
        if not dataset_path:
            print("Dataset download failed, exiting")
            return

    if dataset_path:
        output_file = os.path.join(args.data_dir, "Princeton_llama3_ultrafeedback_armorm_dpo.jsonl")
        if not convert_to_openrlhf_format(dataset_path, output_file, args.max_samples):
            print("Dataset conversion failed, exiting")
            return

    if model_path and dataset_path:
        config_file = create_config_file(model_path, output_file, args.config_dir)
        print(f"Setup complete! Config: {config_file}")

    print("=" * 60)
    print("All done!")
    print("Run DPO training:")
    print("bash train_dpo_rto_sft_improved.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
