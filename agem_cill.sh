#!/bin/bash

source ./env
MODEL_DIR=${MODEL_BASE_DIR}${1}

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${2}
echo "Training with GPUS: $CUDA_VISIBLE_DEVICES"

python -m src.strategies.agem_cill_unified \
--model_save_path ${MODEL_DIR} \
--dataset_save_path ${DATA_DIR} \
--dataset_name iwslt2017 \
--train_epochs 50 \
--lang_pairs fr-en nl-it en-ro it-ro \
--pairs_in_experience 2 \
--metric_for_best_model bleu \
--batch_size 150 \
--gradient_accumulation_steps 1 \
--early_stopping 10 \
--logging_dir ${EXP_HOME}/${MODEL_DIR}/train_logs \
--save_steps 5000 \
--eval_steps 5000 \
--agem_sample_size 150 \
--agem_patterns_per_exp 180365 \
--fp16 --logging_dir ${MODEL_DIR}/train_logs \
        2>&1 | tee -a ${MODEL_DIR}/stout-err.log