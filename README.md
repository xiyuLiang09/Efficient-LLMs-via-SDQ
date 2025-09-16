# Efficient LLMs via Switcble and Dynamic Quantization (SDQ)

This is the implementation of the coding test for `EIC Lab`.

## Overview

- Model: [GPT-2 small](https://arxiv.org/pdf/2305.17888)
- Dataset: [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad)
- Task: Question Answering
- Quantization Function: [LLM-QAT](https://arxiv.org/pdf/2305.17888)

## Run

### Requirements

- Python 3.12, PyTorch 2.0 compatible with local CUDA version
- pip install -r requirements.txt

### Start

```sh
git clone https://github.com/xiyuLiang09/Efficient-LLMs-via-SDQ.git

cd Efficient-LLMs-via-SDQ
```

### Training

#### 1. Switchable Precision Training (SP)

  The training function not only supports switchable precision training but also incorporates cascade distillation from [InstantNet](https://arxiv.org/pdf/2104.10853) to improve model performance at low precision. Use the script `run_sp.sh` in the `./scripts` folder to tune the model with SP:

```sh
bash scripts/run_sp.sh
```
  
  or use the following command to run the customized training parameters. Note that the `--training_strategy` should be set to `sp`:

```sh
python train.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --training_strategy sp \
  --num_epochs 1 \
  --batch_size 24 \
  --gradient_accumulation_steps 3 \
  --max_steps 1000 \
  --lr_schedule cosine \
  --distill_weight 1 \
  --learning_rate 1e-3 \
  --output_dir model/tuned_gpt2_sp \
```

#### 2. Cyclic Precision Training (CPT)

  To enable the training bit-widths to be changed dynamically (i.e., use the cyclic precision training), use the script `run_cpt.sh` in the `./scripts` folder directly to tune the model with cyclic precision:

```sh
bash scripts/run_cpt.sh
```

  Similar to using switchable precision training, you can also use the following command, but make sure to set `--training_strategy` to `cpt`:

```sh
python train.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --training_strategy cpt \
  --num_cyclic_period 5 \
  --num_epochs 1 \
  --batch_size 24 \
  --max_steps 1000 \
  --gradient_accumulation_steps 3 \
  --lr_schedule cosine \
  --learning_rate 1e-3 \
  --output_dir model/tuned_gpt2_cpt \
  --do_eval
```

After training, the model weights, LoRA module weights (two `.pt` files), and training configuration (one `.json` file) are all saved in corresponding `$output_dir`.