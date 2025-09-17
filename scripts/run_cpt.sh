#!/bin/bash
python train.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --training_strategy cpt \
  --num_epochs 1 \
  --batch_size 24 \
  --gradient_accumulation_steps 3 \
  --max_steps 1000 \
  --lr_schedule cosine \
  --learning_rate 1e-4 \
  --num_cyclic_period 4 \
  --cyclic_num_bits_schedule 3 8 \
  --output_dir $1 \
  --do_eval
