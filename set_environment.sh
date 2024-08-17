#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install necessary Python packages
pip install azureml-mlflow tensorboard
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install lightning==2.1.2 lightning[app]
pip install jsonargparse[signatures] tokenizers sentencepiece wandb lightning[data] torchmetrics
pip install tensorboard zstandard pandas pyarrow huggingface_hub
pip install flash-attn --no-build-isolation

# Clone and install Flash Attention components
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/csrc/rotary
pip install .
cd ../layer_norm
pip install .
cd ../xentropy
pip install .
cd ../../

# Install Mamba packages
pip install causal-conv1d
# git clone https://github.com/state-spaces/mamba.git
pip install mamba_ssm

# Install Triton and other packages
pip install triton-nightly --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
pip install einops
pip install opt_einsum
pip install git+https://github.com/sustcsonglin/flash-linear-attention@98c176e
