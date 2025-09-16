#!/bin/bash
cd "$(dirname "$0")/.."  # 确保在 SDQ 目录下执行
PYTHONPATH=. python scripts/eval_shuffle_opt.py \
    --bits_schedule sp \
    --run_id jfestrnx \
    --repeats 2 \
