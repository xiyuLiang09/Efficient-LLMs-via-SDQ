#!/bin/bash
python bit_search.py \
    --search_method random \
    --num_fixed_layers 6 \
    --repeats 2 \
    --config_path configs/default_bits_config.json \
    --output_path configs/random_search_config.json
