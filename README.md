# 🧠 Instruction-Finetuning Pipeline for Language Models

This repository provides a complete pipeline to process datasets, train instruction-tuned models using full fine-tuning or LoRA-based fine-tuning, and generate responses using the trained models.

---

## 📁 Directory Structure
```
├── data/ # Processed and raw data directory
│ ├── new_test.csv
│ ├── new_train.zip
│ ├── new_valid.csv
│
├── notebooks/
│ └── Process_data.ipynb # Notebook to convert JSONL to CSV format
│
├── scripts/ # Scripts for preprocessing and generation
│ ├── init.py
│ ├── commands.txt # Commands for preprocessing and training
│ ├── generate_response.py # Script to generate model responses
│ └── process_dataset.py # Converts CSV to Huggingface dataset format
│
├── src/ # Training scripts
│ ├── sft.py # Full finetuning
│ └── sft_lora.py # LoRA finetuning
│
└── README.md
```

---

## 📊 Data Format

The `csv` files (`new_train.csv`, `new_valid.csv`, `new_test.csv`) contain structured instruction tuning data with the following format:

| system                     | prompt                                      | output                                     |
|---------------------------|---------------------------------------------|--------------------------------------------|
| System message to the AI  | Instruction for the model                   | Ground-truth output                        |
| You are an AI assistant…  | You will be given a title and a post…      | I still have contact with an old ex’s…     |

Each row corresponds to a supervised instruction-response pair.

---

## 📘 Notebooks

### `notebooks/Process_data.ipynb`

Use this notebook to **convert `.jsonl` files into `.csv` format** that aligns with the system-prompt-output structure required by the training scripts.

---

## 🔧 Scripts

### Dataset Preprocessing

- `scripts/process_dataset.py`: Converts CSV files into Huggingface `datasets` format.
- `scripts/generate_response.py`: Uses trained models to generate responses from prompts.

### Preprocessing Command

```bash
python scripts/process_dataset.py \
--data_dir data \
--train_file new_train.csv \
--validation_file new_valid.csv \
--test_file new_test.csv \
--save_data_dir data/processed_data_hf
```
## 🧠 Training
### Full Finetuning

Use src/sft.py for full model finetuning.
```
python src/sft.py \
--data_dir data/processed_data_hf \
--model_path Qwen/Qwen3-0.6B \
--out_dir path_to_output/training_runs
```
## LoRA Finetuning

Use src/sft_lora.py to perform LoRA-based parameter-efficient finetuning. Adjust script arguments based on your model and hardware setup.
📂 Commands Reference

All relevant commands are provided in scripts/commands.txt, including:
- Preprocessing CSV to Huggingface format
- Running full SFT
- Placeholder for LoRA commands (can be added as needed)

## 🚀 Getting Started
- Place your original .jsonl files in the desired location.
- Use notebooks/Process_data.ipynb to convert .jsonl to .csv.
- Use process_dataset.py to convert the CSV into Huggingface format.
- Choose your preferred training script: sft.py or sft_lora.py.
- Train your model and use generate_response.py to produce outputs.

## 📌 Notes
- Ensure the dataset follows the correct format: system, prompt, output.
- new_train.zip may contain large-scale training data; unzip it before use.
- All paths can be adjusted via command-line arguments.


