import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from dataloader import create_data_loaders
from eval import compute_score_per_batch, inference

# from dataloader import create_data_loaders
from loralib.utils import (
    mark_only_lora_as_trainable,
    replace_qlinear_with_loralayer,
)
from model.configuration_sdq import SDQConfig
from model.modeling_sdq_gpt2 import GPT2ForQuestionAnswering
from utils import (
    cyclic_adjust_precision,
    get_bits_config,
    save_tuned_model,
)
from utils import update_model_bits_config as update_model_bits

records = {
    "exact_match": [],
    "f1": [],
}
def parse_args():
    parser = argparse.ArgumentParser(description="Precision Range Test")

    # model and dataset path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained GPT2 model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")

    # training parameters
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_steps", type=int, default=-1)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--precision_range", type=int, nargs=2, required=True, help="Precision range with format (low_bits, high_bits)"
    )

    return parser.parse_args()

def train(
    model,
    precision_range: Tuple,
    train_dataloader,
    optimizer,
    device,
    num_epochs: int = 1,
    output_dir: str | Path | None = None,
    max_steps: int = -1,
    eval_steps: int = -1,
    ckpt_steps: int = -1,
    do_eval: bool = True,
    gradient_accumulation_steps: int = 1,
    eval_examples=None,
    eval_features=None,
    eval_data_loader=None,
):
    model.train()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    start_time = time.time()
    global_step = 0

    total_iters = (
        max_steps if max_steps > 0 else (len(train_dataloader) * num_epochs // max(1, gradient_accumulation_steps))
    )

    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = learning_rate

    # Training loop
    step_loss = 0.0
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, batch in enumerate(progress_bar):
            model.train()
            batch_start_time = time.time()
            offset_mapping = batch.pop("offset_mapping")
            context_positions = batch.pop("context_positions")
            batch = {k: batch[k].to(device) for k in batch.keys()}

            num_bits = cyclic_adjust_precision(precision_range, global_step, total_iters)
            bits_config = get_bits_config(num_bits)
            update_model_bits(model, bits_config)

            outputs = model(**batch)

            # Compute score per batch
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

            if (i + 1) % gradient_accumulation_steps == 0:
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

                # log_data = {
                #     "train/loss": step_loss,
                #     "train/learning_rate": optimizer.param_groups[0]["lr"],
                #     "train/batch_time": batch_time,
                #     "eval/exact_match": score["exact_match"],
                #     "eval/f1": score["f1"],
                # }
                # if run:
                #     run.log(log_data)
                records["exact_match"].append(score["exact_match"])
                records["f1"].append(score["f1"])

                global_step += 1
                step_loss = 0

                # Validate per eval_steps
                if do_eval and eval_steps > 0 and global_step % eval_steps == 0:
                    score = inference(
                        model,
                        eval_data_loader=eval_data_loader,
                        examples=eval_examples,
                        features=eval_features,
                        device=device,
                        exp_name="precision_range_test",
                    )
                    print(score)

                # Control iterations
                if max_steps > 0 and global_step >= max_steps:
                    break

            # Save checkpoint per ckpt_steps if output_dir is provided
            if ckpt_steps > 0 and global_step % ckpt_steps == 0 and output_dir is not None:
                ckpt_dir = f"{output_dir}/ckpt_{(global_step + 1) // ckpt_steps}"
                save_tuned_model(model, output_dir=ckpt_dir)

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

def acc_visualization(output_dir: str, dpi=300):
    """Visualization"""
    title_fs = 16  
    axis_label_fs = 18  
    bit_label_fs = 14  
    tick_fs = 12   

    def interp_nan(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = len(y)

        if n == 0:
            return y
        idx = np.arange(n)
        good = np.isfinite(y)

        if not np.any(good):
            return np.zeros_like(y, dtype=float)
        y_interp = y.copy()

        first, last = idx[good][0], idx[good][-1]
        y_interp[:first] = y[good][0]
        y_interp[last + 1 :] = y[good][-1]

        mask = ~good
        y_interp[mask] = np.interp(idx[mask], idx[good], y[good])
        return y_interp

    # smoothing (default window length = 100)
    def running_average(y: np.ndarray, window_size: int) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return y
        
        w = min(window_size, n)
        if w < 3:
            w = 3
       
        if w % 2 == 0:
            w += 1
        smoothed = np.convolve(y, np.ones(w) / w, mode="same")
        
        if w > 1:
            tail_idx = min(w, n) - 1
            smoothed[-(w - 1) :] = smoothed[-tail_idx]
        return smoothed

    em_raw = np.asarray(records.get("exact_match", []), dtype=float)
    f1_raw = np.asarray(records.get("f1", []), dtype=float)

    em = interp_nan(em_raw)
    f1 = interp_nan(f1_raw)

    n = len(em)
    if n == 0:
        raise ValueError("Records['exact_match'] is empty, cannot create the curve")

    x = np.arange(n)

    win = 100  
    em_smooth = running_average(em, win)
    f1_smooth = running_average(f1, win)

    # create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # EM：raw + smooth
    ax.plot(x, em, label="Exact Match (Raw)")
    ax.plot(x, em_smooth, linewidth=2.0, label=f"Exact Match (Smoothed, Window={min(win, len(em))})")

    # # F1：raw + smooth
    # ax.plot(x, f1, label="F1 (Raw)")
    # ax.plot(x, f1_smooth, linewidth=2.0, label=f"F1 (Smoothed, Window={min(win, len(f1))})")

    ax.grid(False)

    start_v = int(np.floor(x.min() / 100.0) * 100)
    end_v = int(np.ceil(x.max() / 100.0) * 100)
    for v in range(start_v, end_v + 1, 100):
        ax.axvline(x=v, color="gray", linestyle="--", linewidth=1)

    num_segments = max(1, (end_v - start_v) // 100) 
    for i in range(num_segments):
        bit = 2 + i 
        left, right = i * 100, (i + 1) * 100
        xc = (left + right) / 2.0
        ax.text(
            xc,
            0.95,
            f"{bit}-bit",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=bit_label_fs,
        )

    ax.set_title("Exact Match per Step", fontsize=title_fs)
    ax.set_xlabel("Iterations", fontsize=axis_label_fs)
    ax.set_ylabel("Score", fontsize=axis_label_fs)

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.legend()
    plt.tight_layout()
    
    png_save_path = os.path.join(output_dir, "prt_result.png")
    plt.savefig(png_save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saving PRT result to {png_save_path}")

def main():
    args = parse_args()

    init_bits_config = {
        "default": {"attn": {"w_bits": 4, "a_bits": 4, "kv_bits": 16}, "mlp": {"w_bits": 4, "a_bits": 4}},
        "layers": {},
    }
    lora_bits_config = {
        "default": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
    }

    model_config = SDQConfig.from_pretrained(args.model_path)
    model_config.bits_config = init_bits_config
    model = GPT2ForQuestionAnswering.from_pretrained(args.model_path, config=model_config)
    replace_qlinear_with_loralayer(model, lora_bits_config)
    mark_only_lora_as_trainable(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    train_dataloader, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
        args.dataset_path, args.batch_size, batch_eval=True
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Save training configuration to $output_dir/train_config.json
    output_dir = args.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "train_config.json")
        with open(config_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)

    # Train
    train(
        model,
        precision_range=args.precision_range,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        ckpt_steps=args.ckpt_steps,
        do_eval=args.do_eval,
        output_dir=output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # if GPU memory is not enough, increase this param.
        eval_examples=eval_examples,
        eval_features=eval_dataset,
        eval_data_loader=eval_data_loader,
    )
    
    # Visualize and save PRT result
    acc_visualization(output_dir=output_dir)
    json.dump(records, open(os.path.join(output_dir, "records.json"), "w"))


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "offline"
    main()
"""
python precision_range_test.py \
  --model_path model/linear_gpt2 \
  --dataset_path rajpurkar/squad \
  --precision_range 2 12 \
  --num_epochs 1 \
  --batch_size 24 \
  --gradient_accumulation_steps 3 \
  --max_steps 1000 \
  --learning_rate 1e-3 \
  --output_dir model/precision_range_test/w2a6-w12a12/ \
  --do_eval
"""
