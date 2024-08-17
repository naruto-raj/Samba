#!/bin/bash

# Set variables
REPO_URL="https://github.com/microsoft/Samba.git"
DATASET_URL="https://huggingface.co/datasets/cerebras/SlimPajama-627B"
DATASET_DIR="/workspace/dataset"
REPO_DIR="/workspace/Samba"
HF_TOKEN=""  # Replace with your actual Hugging Face token

# Export HF_TOKEN environment variable
export HF_TOKEN=$HF_TOKEN

# Clone the Samba repository
if [ ! -d "$REPO_DIR" ]; then
    git clone $REPO_URL $REPO_DIR
else
    echo "Samba repository already cloned."
fi

# Check if Git LFS is installed
if ! git lfs --version &> /dev/null
then
    echo "Git LFS is not installed. Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
    git lfs install
else
    echo "Git LFS is already installed."
fi

# Attempt to run a Lit-LLaMA script (Note: Ensure the path and script are correct)
if [ -d "$REPO_DIR/scripts" ]; then
    cd $REPO_DIR/scripts
    if [ -f "download.py" ]; then
        python download.py --repo_id "meta-llama/Llama-2-7b" --tokenizer_only True --access_token $HF_TOKEN
    else
        echo "download.py script not found in Samba repository."
    fi
else
    echo "Scripts directory not found in Samba repository."
fi

# Download the SlimPajama dataset
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p $DATASET_DIR
    git clone $DATASET_URL $DATASET_DIR
else
    echo "Dataset directory already exists."
fi
