import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from transformers.optimization import get_scheduler

from config import Config
from dataloader import create_data_loaders
from eval import compute_score_per_batch, inference
from loralib.utils import (
    mark_only_lora_as_trainable,
    replace_qlinear_with_loralayer,
)
from model.configuration_sdq import SDQConfig
from model.modeling_sdq_gpt2 import GPT2ForQuestionAnswering
from utils import cyclic_adjust_bits_config, cyclic_adjust_precision, get_bits_config, save_tuned_model
from utils import update_model_bits_config as update_model_bits


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDQ-GPT2 Model")

    # model and dataset path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained GPT2 model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")

    # general training parameters
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_steps", type=int, default=-1)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)

    # training parameters for SP
    parser.add_argument("--distill_weight", type=float, default=1)
    
    parser.add_argument("--training_strategy", type=str, choices=["sp", "cpt", "full"], default="sp")

    # training parameters for CDT
    parser.add_argument("--num_cyclic_period", type=int, default=5)
    parser.add_argument("--cyclic_num_bits_schedule", type=int, nargs=2, default=(4, 8))

    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def train(
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    device,
    num_epochs,
    bits_config_list,
    loss_scale,
    output_dir: str | Path | None = None,
    max_steps: int = -1,
    eval_steps: int = -1,
    ckpt_steps: int = -1,
    do_eval: bool = True,
    gradient_accumulation_steps: int = 1,
    distill_weight: float = 1,
    training_strategy: Literal["sp", "cpt", "full"] = "sp",
    eval_examples=None,
    eval_features=None,
    eval_data_loader=None,
    eval_bits_config_list=None,
    num_cyclic_period=4,
    cyclic_num_bits_schedule=(3, 8),
):
    model.train()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    start_time = time.time()
    global_step = 0

    # calculate cyclic_period for cyclic precision training
    if max_steps is not None and max_steps > 0:
        total_opt_steps = max_steps
    else:
        total_opt_steps = (len(train_dataloader) * num_epochs) // max(1, gradient_accumulation_steps)
    cyclic_period = max(1, total_opt_steps // max(1, num_cyclic_period))

    loss_fct = nn.MSELoss()

    # If eval_bits_config_list is None, use bits_config_list to evaluate
    if eval_bits_config_list is None:
        eval_bits_config_list = bits_config_list

    # Train loop
    step_loss = 0.0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_step = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        loss_value = [-1 for _ in bits_config_list]  # for switchable precision
        last_bits_config_idx = -1  # for cyclic precision
        last_num_bits = -1  # for cyclic precision

        for i, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            batch = {k: batch[k].to(device) for k in batch.keys()}

            # Jointly training to support Switchable Precision
            if training_strategy == "sp":
                teacher_start_logits_list = []
                teacher_end_logits_list = []

                for bits_config in bits_config_list[::-1]:  # high-bit to low-bit
                    update_model_bits(model, bits_config)

                    outputs = model(**batch)
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    loss = outputs.loss

                    # knowledge distillation
                    if len(teacher_start_logits_list) > 0:
                        for start_logits_t, end_logits_t in zip(teacher_start_logits_list, teacher_end_logits_list):
                            s_loss = loss_fct(start_logits, start_logits_t)
                            e_loss = loss_fct(end_logits, end_logits_t)
                            loss += distill_weight * (s_loss + e_loss) / 2

                    loss = loss * loss_scale[bits_config_list.index(bits_config)]
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # accumulate loss_value in gradient_accumulation_steps
                    if loss_value[bits_config_list.index(bits_config)] == -1:
                        loss_value[bits_config_list.index(bits_config)] = loss.item()
                    else:
                        loss_value[bits_config_list.index(bits_config)] += loss.item()

                    teacher_start_logits_list.append(start_logits.detach())
                    teacher_end_logits_list.append(end_logits.detach())

                    del outputs, loss, start_logits, end_logits

                # Due to limited GPU memory, add param `gradient_accumulation_steps` to meet the requirement of:
                # 1. 1000 iterations
                # 2. covering more data samples.
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # calculate average loss
                    total_loss = sum(value for value in loss_value if value != -1)
                    count = sum(1 for value in loss_value if value != -1)
                    average_loss = (total_loss / count) if count else 0.0

                    epoch_step += 1
                    global_step += 1

                    progress_bar.set_postfix(
                        {
                            "loss": f"{average_loss:.4f}",
                            "step": global_step,
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                    # reset loss_value per gradient_accumulation_steps
                    loss_value = [-1 for _ in bits_config_list]

                    # control iterations
                    if max_steps > 0 and global_step >= max_steps:
                        break

            # Cyclic Precision Training
            elif training_strategy == "cpt":
                offset_mapping = batch.pop("offset_mapping")
                context_positions = batch.pop("context_positions")
                # when the input bit-width is a set of integers, use `cyclic_adjust_precision`
                num_bits = cyclic_adjust_precision(
                    cyclic_num_bits_schedule=cyclic_num_bits_schedule,
                    iter=global_step,
                    cyclic_period=cyclic_period,
                )
                if num_bits != last_num_bits:
                    bits_config = get_bits_config(num_bits)
                    update_model_bits(model, bits_config)
                    print(f"Update num_bits from {last_num_bits} to {num_bits}")
                    last_num_bits = num_bits

                # when the input bit-width is a dict, use logic below:
                # bits_config_idx = cyclic_adjust_bits_config(
                #     bits_config_list=bits_config_list,
                #     current_iter=global_step,
                #     cyclic_period=cyclic_period,
                # )
                # if bits_config_idx != last_bits_config_idx:
                #     update_model_bits(model, bits_config_list[bits_config_idx])
                #     print(f"update num_bits from config[{last_bits_config_idx}] to config[{bits_config_idx}]")
                #     last_bits_config_idx = bits_config_idx

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

                # Due to limited GPU memory, add param `gradient_accumulation_steps` to meet the requirement of coding test.
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.set_postfix(
                        {
                            "loss": f"{step_loss:.4f}",
                            "step": global_step,
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                    epoch_step += 1
                    global_step += 1
                    step_loss = 0

            # note that `max_loss` strategy is only used for comparison test
            elif training_strategy == "max_loss":
                loss_list = []
                teacher_start_logits_list = []
                teacher_end_logits_list = []

                for i, bits_config in enumerate(bits_config_list):
                    update_model_bits(model, bits_config)
                    outputs = model(**batch)
                    loss_list.append(outputs.loss.item())
                    teacher_start_logits_list.append(outputs.start_logits.detach())
                    teacher_end_logits_list.append(outputs.end_logits.detach())

                bits_config_max = bits_config_list[np.array(loss_list).argmax()]
                update_model_bits(model, bits_config_max)
                outputs = model(**batch)

                for start_logits_t, end_logits_t in zip(teacher_start_logits_list, teacher_end_logits_list):
                    s_loss = nn.MSELoss()(outputs.start_logits, start_logits_t)
                    e_loss = nn.MSELoss()(outputs.end_logits, end_logits_t)
                    loss += distill_weight * (s_loss + e_loss) / 2

                loss = loss * loss_scale[bits_config_list.index(bits_config)]
                loss = loss / gradient_accumulation_steps
                loss.backward()
            else:
                raise ValueError(f"Unknown training_strategy: {training_strategy}")

            if (i + 1) % gradient_accumulation_steps == 0:
                # Validate per cycle which includes `eval_steps` steps
                if do_eval and eval_steps > 0 and global_step % eval_steps == 0:
                    scores = inference(
                        model=model,
                        eval_data_loader=eval_data_loader,
                        examples=eval_examples,
                        features=eval_features,
                        device=device,
                        exp_name=f"training_{training_strategy}",
                        eval_bits_config_list=eval_bits_config_list,
                    )
                    print(scores)
                    model.train()  # switch to training mode after validation

                if ckpt_steps > 0 and global_step % ckpt_steps == 0 and output_dir is not None:
                    # if output_dir is not None:
                    ckpt_dir = f"{output_dir}/ckpt_{(global_step + 1) // ckpt_steps}"
                    save_tuned_model(model, output_dir=ckpt_dir)
                
                if max_steps > 0 and global_step >= max_steps:
                    break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed. Time consumed: {epoch_time:.2f}s")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")


def main():
    args = parse_args()

    bits_config_list = Config.bits_config_list
    args.bits_config_list = bits_config_list
    if hasattr(Config, "eval_bits_config_list"):
        eval_bits_config_list = Config.eval_bits_config_list
    else:
        eval_bits_config_list = bits_config_list
    loss_scale = Config.get("loss_scale", [1] * len(bits_config_list))
    lora_bits_configs = Config.get(
        "lora_bits_configs",
        {"default": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1}},
    )
    if args.training_strategy == "cpt":
        lora_bits_configs = {"default": lora_bits_configs["default"]}

    
    # use highest precision to eval
    if args.training_strategy == "cpt":
        eval_bits_config_list = [get_bits_config(args.cyclic_num_bits_schedule[1])]

    # Load model, tokenizer
    model_config = SDQConfig.from_pretrained(args.model_path)
    model_config.bits_config = bits_config_list[0]
    model_config.lora_bit_configs = lora_bits_configs
    model_config.loss_scale = loss_scale

    model = GPT2ForQuestionAnswering.from_pretrained(args.model_path, config=model_config)
    replace_qlinear_with_loralayer(model, lora_bits_configs)
    mark_only_lora_as_trainable(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare dataset
    if args.training_strategy == "cpt":
        train_dataloader, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
            args.dataset_path, args.batch_size, batch_eval=True
        )
    else:
        train_dataloader, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
            args.dataset_path, args.batch_size
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

    # Add suffix to output_dir and save config.json and model.pt
    output_dir = args.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "train_config.json")
        with open(config_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)

    # Train
    train(
        model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=args.num_epochs,
        bits_config_list=bits_config_list,
        training_strategy=args.training_strategy,
        loss_scale=loss_scale,  # weighted average of loss
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        ckpt_steps=args.ckpt_steps,
        do_eval=args.do_eval,
        num_cyclic_period=args.num_cyclic_period,
        output_dir=output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # if GPU memory is not enough, increase the value of this parameter.
        distill_weight=args.distill_weight,
        eval_examples=eval_examples,
        eval_features=eval_dataset,
        eval_data_loader=eval_data_loader,
        eval_bits_config_list=eval_bits_config_list,
        cyclic_num_bits_schedule=args.cyclic_num_bits_schedule,
    )

    # Save
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
            exp_name=args.training_strategy,
            eval_bits_config_list=eval_bits_config_list,
        )
        print(scores)


if __name__ == "__main__":
    main()
