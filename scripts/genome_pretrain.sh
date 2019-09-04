#!/bin/bash

export MODEL="genome10M-b2048"
export TPU_IP="10.0.102.2"
export ROOT="/home/athon/genomeXL"
export GS_ROOT="gs://athon-research/genomeXL"
export RECORD_INFO_DIR="models/"${MODEL}"/tfrecords"
export MODEL_DIR="models/${MODEL}"

cd ~/xlnet && \
python train.py \
  --use_tpu=True \
  --tpu="grpc://"${TPU_IP}":8470" \
  --gcp_project="iop-mk" \
  --tpu_zone="us-central1-b" \
  --num_hosts=1 \
  --track_mean=True \
  --record_info_dir=${GS_ROOT}/${RECORD_INFO_DIR} \
  --model_dir=${GS_ROOT}/${MODEL_DIR} \
  --train_batch_size=128 \
  --uncased=True \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=24 \
  --d_model=1024 \
  --d_embed=1024 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85
