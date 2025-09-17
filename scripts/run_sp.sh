#!/bin/bash
python train.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --training_strategy sp \
  --num_epochs 1 \
  --batch_size 24 \
  --gradient_accumulation_steps 3 \
  --max_steps 1000 \
  --learning_rate 1e-3 \
  --lr_schedule cosine \
  --distill_weight 1 \
  --output_dir $1 \
  --do_eval
