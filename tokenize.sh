#!/bin/bash

# Set variables
DATASET_DIR="/workspace/SlimPajama-627B"
LLAMA_TOKENIZER_DIR="/workspace/Samba/scripts/checkpoints/meta-llama/Llama-2-7b"

# Change to the Samba directory (assuming Samba repo is cloned in home directory)
cd /workspace/Samba

# Prepare the dataset
python scripts/prepare_slimpajama.py --source_path $DATASET_DIR --tokenizer_path $LLAMA_TOKENIZER_DIR --destination_path /workspace/data/slim --split validation --percentage 1.0
python scripts/prepare_slimpajama.py --source_path $DATASET_DIR --tokenizer_path $LLAMA_TOKENIZER_DIR --destination_path /workspace/data/slim --split train --percentage 0.04