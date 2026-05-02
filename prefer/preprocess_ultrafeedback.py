#!/usr/bin/env python3
"""
Preprocess Llama3-UltraFeedback-ArmoRM into OpenRLHF DPO format.
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm


def convert_llama3_ultrafeedback_armorm_to_dpo_format(input_dataset, output_file, max_samples=None):
    """Convert Llama3-UltraFeedback-ArmoRM dataset to DPO format."""
    dpo_data = []
    
    for i, sample in enumerate(tqdm(input_dataset, desc="Converting Llama3-UltraFeedback-ArmoRM")):
        if max_samples and i >= max_samples:
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
            else:
                print(f"Warning: Skipping sample {i} due to invalid chosen/rejected responses")
                continue
        else:
            print(f"Warning: Skipping sample {i} due to unexpected format: {sample.keys()}")
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dpo_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(dpo_data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert Llama3-UltraFeedback-ArmoRM dataset to DPO format")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/llama3-ultrafeedback-armorm",
                       help="Llama3-UltraFeedback-ArmoRM dataset name")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--output_file", type=str, default="./data/Princeton_llama3_ultrafeedback_armorm_dpo.jsonl",
                       help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to convert")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.split)

    convert_llama3_ultrafeedback_armorm_to_dpo_format(dataset, args.output_file, args.max_samples)
    
    print("Conversion completed!")


if __name__ == "__main__":
    main()
