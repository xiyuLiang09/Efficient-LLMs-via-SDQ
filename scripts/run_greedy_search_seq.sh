#!/bin/bash
python bit_search.py \
    --model_dir model/tuned_gpt2_sp/jfestrnx \
    --output_path configs/greedy_search_seq.json \
    --batch_size 24 \
    --alpha 0.7
