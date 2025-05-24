from datasets import load_from_disk
import argparse
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TrainingArguments
import os

def main():
    parser = argparse.ArgumentParser(description="Process CSV data to Hugging Face messages format.")
    parser.add_argument("--data_dir", required=True, help="Root directory of raw CSV files.")
    parser.add_argument("--model_path", default="train.csv", help="Training CSV filename.")
    parser.add_argument("--out_dir", required=True, help="Training Save Directory")
   
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ################
    # Dataset
    ################

    dataset = load_from_disk(args.data_dir)

    print("-"*60)
    print("Dataset Loaded Successfully!!!")
    print("-"*60)
        
    ################
    # Model init kwargs & Tokenizer
    ################
    
    model_id = args.model_path  
    # print(model_id)             

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    processor = AutoProcessor.from_pretrained(model_id)

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
    
    print("-"*60)
    print("Model and Tokenizer Loaded Successfully!!!")
    print("-"*60)

    ################
    # Training
    ################
    max_seq_length = 2048
    training_args = SFTConfig(
                output_dir=args.out_dir,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                optim="adamw_torch_fused",
                learning_rate=2e-5,
                save_strategy="steps",
                save_total_limit=3,
                metric_for_best_model="eval_loss",
                bf16=True,
                logging_steps=10,
                logging_strategy="steps",
                save_steps=50,
                report_to="tensorboard",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.1,
                warmup_ratio=0.03,
                weight_decay=0.1,
                eval_strategy="steps",
                eval_steps=50,
                max_seq_length=max_seq_length
    )

    trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=tokenizer,
    )


    # --- Checkpoint Resumption Logic (Improved Robustness) ---
    checkpoint_dir = args.out_dir # Use the output_dir from SFTConfig
    checkpoint_path = None # Initialize path to None

    # Check if the output directory exists
    if os.path.isdir(checkpoint_dir):
        # Find directories starting with "checkpoint-"
        checkpoints = [
            d for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Find the checkpoint with the highest step number
            try:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                # Construct the full path to the latest checkpoint
                potential_path = os.path.join(checkpoint_dir, latest_checkpoint)
                # Optional: Add a basic check if it looks like a valid HF Trainer checkpoint
                # DeepSpeed validation happens internally in trainer.train()
                if os.path.exists(os.path.join(potential_path, "trainer_state.json")) or \
                os.path.exists(os.path.join(potential_path, "pytorch_model.bin")) or \
                os.path.exists(os.path.join(potential_path, "latest")): # DeepSpeed indicator
                    checkpoint_path = potential_path
                    print(f"Found potential checkpoint: {checkpoint_path}")
                else:
                    print(f"Directory {potential_path} exists but missing key checkpoint files.")
            except (ValueError, IndexError):
                print(f"Could not parse step number from checkpoint directories in {checkpoint_dir}")
        else:
            print(f"Output directory '{checkpoint_dir}' exists but contains no directories starting with 'checkpoint-'.")
    else:
        print(f"Output directory '{checkpoint_dir}' does not exist. Will start training from scratch.")
    # --- End Checkpoint Resumption Logic ---


    # --- Call trainer.train ---
    # The Trainer automatically handles resuming if resume_from_checkpoint is a valid path,
    # or starts from scratch if it's None or False.
    print("Starting trainer.train()...")
    if checkpoint_path:
        print(f"Attempting to resume training from: {checkpoint_path}")
        # Trainer expects a path string or True/False. Passing the path is preferred.
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No valid checkpoint found or specified. Starting training from scratch.")
        trainer.train() # resume_from_checkpoint defaults to False




if __name__ == "__main__":

    main()