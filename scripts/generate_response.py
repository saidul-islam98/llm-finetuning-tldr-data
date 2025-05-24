import argparse
import os
from datasets import load_dataset, DatasetDict, Dataset
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch

def main():
    parser = argparse.ArgumentParser(description="Process CSV data to Hugging Face messages format.")
    parser.add_argument("--data_dir", required=True, help="Root directory of raw CSV files.")
    parser.add_argument("--model_path", required=True, help="Path to trained model.")
    parser.add_argument("--test_file", default=None, help="Optional test CSV filename.")
    
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file))


    def predict(messages, model, tokenizer):
        device = "cuda"
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        return response
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row["system"]
        input_value = row["prompt"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"},
        ]

        response = predict(messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(result_text)

    print(test_text_list)


if __name__ == "__main__":
    main()