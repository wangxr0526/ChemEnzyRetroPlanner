#!/bin/bash

if [ -z "${PARROT_REGISTRY}" ]; then
  export PARROT_REGISTRY=parrot_image:latest
fi

export DATA_NAME="USPTO_condition"
export PROCESSED_DATA_PATH=$PWD/dataset/source_dataset/${DATA_NAME}_final
export MODEL_PATH=$PWD/outputs/Parrot_train_in_USPTO_Condition_enhance
export MODEL_STRUCTURE_PATH=$PWD/models

export EXTRA_FILES="outputs/Parrot_train_in_USPTO_Condition_enhance/vocab.txt,outputs/Parrot_train_in_USPTO_Condition_enhance/config.json,outputs/Parrot_train_in_USPTO_Condition_enhance/eval_results.txt,outputs/Parrot_train_in_USPTO_Condition_enhance/tokenizer_config.json,outputs/Parrot_train_in_USPTO_Condition_enhance/training_args.bin,outputs/Parrot_train_in_USPTO_Condition_enhance/scheduler.pt,outputs/Parrot_train_in_USPTO_Condition_enhance/optimizer.pt,outputs/Parrot_train_in_USPTO_Condition_enhance/special_tokens_map.json,outputs/Parrot_train_in_USPTO_Condition_enhance/split_topk.csv,outputs/Parrot_train_in_USPTO_Condition_enhance/model_args.json,config_inference_use_uspto.yaml,dataset/source_dataset/USPTO_condition_final/USPTO_condition_alldata_idx.pkl"

mkdir -p "$PWD/mars"

docker run --rm \
  -v "$PROCESSED_DATA_PATH/USPTO_condition_alldata_idx.pkl":/app/parrot/dataset/source_dataset/USPTO_condition_final/USPTO_condition_alldata_idx.pkl \
  -v "$PWD":/app/parrot/ \
  -v "$MODEL_PATH":/app/parrot/outputs/Parrot_train_in_USPTO_Condition_enhance \
  -v "$PWD/mars":/app/parrot/mars \
  -t "${PARROT_REGISTRY}" \
  torch-model-archiver \
  --serialized-file /app/parrot/outputs/Parrot_train_in_USPTO_Condition_enhance/pytorch_model.bin \
  --model-name="$DATA_NAME" \
  --model-file model.py \
  --version=1.0 \
  --handler=/app/parrot/handler.py \
  --extra-files="$EXTRA_FILES" \
  --export-path=/app/parrot/mars \
  --force
