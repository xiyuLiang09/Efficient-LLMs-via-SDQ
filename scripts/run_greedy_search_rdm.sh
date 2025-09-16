#!/bin/bash
python bit_search.py \
    --use_random \
    --model_dir model/tuned_gpt2_sp/jfestrnx \
    --output_path configs/greedy_search_rdm.json \
    --batch_size 24 \
    --search_method greedy \
    --alpha 0.6
