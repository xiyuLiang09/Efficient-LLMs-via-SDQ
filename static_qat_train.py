import argparse
import json
import math
import os
import time

import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers.optimization import get_scheduler

import wandb

# from dataloader import create_data_loaders
from dataloader import create_data_loaders

# from eval import inference
from loralib.utils import (
    mark_only_lora_as_trainable,
    replace_qlinear_with_loralayer,
)

from model.configuration_sdq import SDQConfig
from model.modeling_sdq_gpt2 import GPT2ForQuestionAnswering
from eval import compute_score_per_batch, inference
from utils import save_tuned_model


def parse_args():
    parser = argparse.ArgumentParser(description="Static Quantization for GPT-2 Model")

    # model and dataset path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained GPT2 model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")

    # training parameters
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, nargs=3, default=(4, 6, 16))

    return parser.parse_args()


def train(
    model,
    run: wandb.Run,
    train_dataloader,
    optimizer,
    lr_scheduler,
    device,
    num_epochs,
    max_steps: int = -1,
    eval_steps: int = -1,
    do_eval: bool = True,
    gradient_accumulation_steps: int = 1,
    eval_examples=None,
    eval_features=None,
    eval_data_loader=None,
    eval_bits_config_list=None,
):
    model.train()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    start_time = time.time()
    global_step = 0

    # Train loop
    step_loss = 0.0
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            offset_mapping = batch.pop("offset_mapping")
            context_positions = batch.pop("context_positions")
            batch = {k: batch[k].to(device) for k in batch.keys()}

            outputs = model(**batch)

            # compute score per batch
            if (i + 1) % gradient_accumulation_steps == 0:
                score = compute_score_per_batch(
                    tokenizer=tokenizer,
                    batch=batch,
                    outputs=outputs,
                    offset_mapping=offset_mapping,
                    context_positions=context_positions,
                )
                print(score)

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            step_loss += loss.item()

            del outputs, loss

            # Due to limited GPU memory, add param `gradient_accumulation_steps` to meet the requirement of:
            # 1. 1000 iterations
            # 2. covering more data samples.
            if (i + 1) % gradient_accumulation_steps == 0:
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                batch_time = time.time() - batch_start_time
                progress_bar.set_postfix(
                    {
                        "loss": f"{step_loss:.4f}",
                        "step": global_step,
                        "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    }
                )

                log_data = {
                    "train/loss": step_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/batch_time": batch_time,
                    "eval/exact_match": score["exact_match"],
                    "eval/f1": score["f1"],
                }
                if run:
                    run.log(log_data, step=global_step)

                global_step += 1
                step_loss = 0

                # Validate per cycle which includes `eval_steps` steps
                if do_eval and eval_steps > 0 and global_step % eval_steps == 0:
                    scores = inference(
                        model=model,
                        eval_data_loader=eval_data_loader,
                        examples=eval_examples,
                        features=eval_features,
                        device=device,
                        exp_name="static_qat_train",
                    )
                    print(scores)
                    model.train()  # switch to training mode after validation

                if max_steps > 0 and global_step >= max_steps:
                    break

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")


def main():
    args = parse_args()
    w_bits, a_bits, kv_bits = args.bits

    with wandb.init(
        project="sdq-gpt2",
        name=f"static_qat_w{w_bits}a{a_bits}kv{kv_bits}",
        config={**args.__dict__},
        settings=wandb.Settings(init_timeout=120),
    ) as run:
        bits_config = {
            "default": {
                "attn": {"w_bits": w_bits, "a_bits": a_bits, "kv_bits": kv_bits},
                "mlp": {"w_bits": w_bits, "a_bits": a_bits},
            },
            "layers": {},
        }
        lora_bits_configs = {
            "default": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
        }
        model_config = SDQConfig.from_pretrained(args.model_path)
        model_config.bits_config = bits_config
        model_config.lora_bit_configs = lora_bits_configs
        model = GPT2ForQuestionAnswering.from_pretrained(args.model_path, config=model_config)

        replace_qlinear_with_loralayer(model, lora_bits_configs)
        # model = load_tuned_qa_model(model_dir="model/static_qat_train/w8a8kv16")
        mark_only_lora_as_trainable(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load data
        train_dataloader, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
            args.dataset_path, args.batch_size, batch_eval=True
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_training_steps = args.max_steps if args.max_steps > 0 else args.num_epochs * num_update_steps_per_epoch
        warmup_steps = int(num_training_steps * 0.1)

        lr_scheduler = get_scheduler(
            name=args.lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Save config.json and model.pt
        output_dir = args.output_dir
        if output_dir is not None:
            output_dir = os.path.join(output_dir, f"w{w_bits}a{a_bits}kv{kv_bits}_8000/")
            os.makedirs(output_dir, exist_ok=True)
            config_path = os.path.join(output_dir, "train_config.json")
            with open(config_path, "w") as f:
                json.dump(args.__dict__, f, indent=4)

        # Train
        train(
            model,
            run=run,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            do_eval=args.do_eval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # if GPU memory is not enough, increase this param.
            eval_examples=eval_examples,
            eval_features=eval_dataset,
            eval_data_loader=eval_data_loader,
        )

        # Save fine-tuned model
        if output_dir is not None:
            save_tuned_model(model, output_dir=output_dir)

        # Evaluation
        if args.do_eval:
            scores = inference(
                model,
                eval_data_loader=eval_data_loader,
                examples=eval_examples,
                features=eval_dataset,
                device=device,
                exp_name="static_qat_train",
            )
            print(scores)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    main()

"""
python static_qat_train.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --num_epochs 1 \
  --batch_size 24 \
  --gradient_accumulation_steps 3 \
  --max_steps 1000 \
  --learning_rate 1e-4 \
  --output_dir model/static_qat_train/ \
  --bits 8 8 16 \
  --do_eval
  --eval_steps 1000
"""
