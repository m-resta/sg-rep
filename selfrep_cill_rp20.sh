#!/bin/bash

source ./env
MODEL_DIR=${MODEL_BASE_DIR}/${1}

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${2}
echo "Training with GPUS: $CUDA_VISIBLE_DEVICES"

python -m src.strategies.selfrep_unified \
--model_save_path ${MODEL_DIR} \
--dataset_save_path ${DATA_DIR} \
--dataset_name iwslt2017 \
--lang_pairs fr-en nl-it en-ro it-ro \
--replay_memory 360730 \
--train_epochs 50 \
--pairs_in_experience 2 \
--metric_for_best_model bleu \
--batch_size 150 \
--early_stopping 10 \
--topk 32000 \
--save_steps 5000 \
--eval_steps 5000 \
--num_selfgenerated_samples 25000 \
--filtering 'delete_enchant'   \
--generation_strategy 1 \
--fp16 --logging_dir ${MODEL_DIR}/train_logs \
        2>&1 | tee -a ${MODEL_DIR}/stout-err.log