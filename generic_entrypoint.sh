#!/bin/bash
source ./env

MODEL_DIR=${MODEL_BASE_DIR}/${1}
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${2}
echo "Training with GPUS: $CUDA_VISIBLE_DEVICES"

python -m ${3} \
--model_save_path ${MODEL_DIR} \
--dataset_save_path ${DATA_DIR} \
"${@:4}"