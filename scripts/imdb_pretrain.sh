#!/bin/bash

export DATA="/home/athon/xlnet/data/imdb/aclImdb"
export MODEL="/home/athon/xlnet/models/xlnet_imdb_L-24_H-1024_A-16"
cd ~/xlnet && \
python train.py \
  --tpu="gprc://10.240.1.2:8470" \
  --track_mean=True \
  --model_dir=$MODEL \
  --save_steps=10 \
  --record_info_dir=$DATA/tfrecords \
  --train_batch_size=2048 \
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