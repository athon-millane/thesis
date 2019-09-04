#!/bin/bash
export GRCH38_P13="/home/athon/genomeXL/data/human/GRCh38_p13.txt"
export GENOME_SPM="sp.genome.v1"

spm_train \
    --input=${GRCH38_P13} \
    --model_prefix=${GENOME_SPM} \
    --vocab_size=1024 \
    --character_coverage=0.99995 \
    --model_type=unigram \
    --control_symbols="<cls>,<sep>,<pad>,<mask>,<eod>" \
    --user_defined_symbols="<eop>,.,(,),-,–,£,€" \
    --shuffle_input_sentence \
    --input_sentence_size=10000000