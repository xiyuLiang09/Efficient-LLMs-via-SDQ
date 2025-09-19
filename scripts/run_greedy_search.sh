#!/bin/bash
python bit_search.py \
    --model_dir $1 \
    --search_method greedy \
    --output_path configs/greedy_search_config.json \
    --low_bits 4 \
    --high_bits 8 \
    --lower_bound 40 \
    --alpha 0.7 \
    --batch_size 24 \