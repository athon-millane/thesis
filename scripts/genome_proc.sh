#!/bin/bash 

export MODEL="genome10M-b2048"
export INPUT_GLOB="/home/athon/genomeXL/data/human/GRCh38_p13_5000.txt"
export SAVE_DIR="/home/athon/genomeXL/models/"/${MODEL}
export SP_PATH="/home/athon/genomeXL/models/"${MODEL}"/spiece.model"

cd ~/xlnet && \
python data_utils.py \
    --bsz_per_host=128 \
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
