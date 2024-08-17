#!/bin/bash

# Get the MASTER_ADDR and MASTER_PORT from the environment variables
echo $MASTER_ADDR
echo $MASTER_PORT

# Check if the environment variables are set, otherwise set defaults
if [ -z "$MASTER_ADDR" ]; then
  MASTER_ADDR="localhost"
  echo "Setting default MASTER_ADDR to $MASTER_ADDR"
fi

if [ -z "$MASTER_PORT" ]; then
  MASTER_PORT="29500"
  echo "Setting default MASTER_PORT to $MASTER_PORT"
fi

# Set variables
DATA_DIR="/workspace/data/slim"

# Change to the Samba directory (assuming Samba repo is cloned in home directory)
cd Samba

# Start the training process
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=samba-421M --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain.py --train_data_dir $DATA_DIR --val_data_dir $DATA_DIR
