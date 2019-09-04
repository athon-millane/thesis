#!/bin/bash 
export INPUT_GLOB="/home/athon/xlnet/data/imdb/aclImdb/train/unsup/*.txt"
export SAVE_DIR="/home/athon/xlnet/data/imdb/aclImdb/tfrecords"
export SP_PATH="/home/athon/xlnet/models/xlnet_cased_L-24_H-1024_A-16/spiece.model"

cd ~/xlnet && \
python data_utils.py \
    --bsz_per_host=32 \
    --num_core_per_host=16 \
    --seq_len=512 \
    --reuse_len=256 \
    --input_glob=${INPUT_GLOB} \
    --save_dir=${SAVE_DIR} \
    --num_passes=20 \
    --bi_data=True \
    --sp_path=${SP_PATH} \
    --mask_alpha=6 \
    --mask_beta=1 \
    --num_predict=85