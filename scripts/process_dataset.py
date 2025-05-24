import argparse
import os
from datasets import load_dataset, DatasetDict, Dataset
import json


def make_conversations(examples):
    """Transforms CSV row (with 'system', 'prompt', 'output') to messages format."""
    return {
        "messages": [
            {"role": "system", "content": examples["system"]},
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant",  "content": examples["output"]}
        ]
    }


def load_and_process_dataset(file_path):
    """Loads a CSV split and transforms it to the messages format."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}, returning None for this split.")
        
        return None

    data = load_dataset("csv", data_files=file_path, split="train")
    
    return data.map(make_conversations, remove_columns=data.column_names)


def main():
    parser = argparse.ArgumentParser(description="Process CSV data to Hugging Face messages format.")
    parser.add_argument("--data_dir", required=True, help="Root directory of raw CSV files.")
    parser.add_argument("--train_file", default="train.csv", help="Training CSV filename.")
    parser.add_argument("--validation_file", default="valid.csv", help="Validation CSV filename.")
    parser.add_argument("--test_file", default=None, help="Optional test CSV filename.")
    parser.add_argument("--save_data_dir", required=True, help="Output directory for processed dataset.")
    
    args = parser.parse_args()

    processed_dataset = {}

    split_definations = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file
    }

    for split, filename in split_definations.items():
        file_path = os.path.join(args.data_dir, filename)
        processed_split = load_and_process_dataset(file_path)
        if processed_split:
            processed_dataset[split] = processed_split
            print(f"Processed {split} split from {file_path}.")
        
    dataset = DatasetDict(processed_dataset)
    save_path = os.path.join(args.save_data_dir, "processed_data_hf")
    dataset.save_to_disk(save_path)



if __name__ == "__main__":
    main()
