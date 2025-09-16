import argparse
import copy
import datetime
import json
import os
import random
from typing import Literal

import torch

from dataloader import create_data_loaders
from eval import inference
from utils import load_tuned_qa_model

# Exact Match Score of the full-precision GPT2 model after 1000 iterations of fine-tuning on SQuAD
EM_BASELINE = 55.75

# The ratio of model size between different bit-width quantizations and the full-precision model
# e.g., the "w4" entry indicates that the model size after 4-bit weight quantization is 0.38 times that of the full-precision model
SIZE_RATIO_MAP = {
    "w4": 0.38,
    "w8": 0.47,
    "w16": 0.65,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run randomized evaluation with quantization configs")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the tuned model")
    parser.add_argument("--alpha", type=float, default=0.7, help="Alpha parameter for greedy evaluation")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for evaluation dataloader")
    parser.add_argument("--search_method", type=str, choices=["greedy", "random"], default="greedy", help="Search method")
    parser.add_argument("--use_random", action="store_true", help="Whether to use random greedy search")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/random_search_bits.json",
        help="Path to load bits_config_list used for random search",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="configs/greedy_search_seq.json",
        help="Output path of the optimal bits config obtained via greedy search",
    )

    return parser.parse_args()


def greedy_search(model_dir: str, output_path: str, alpha: float = 0.7, batch_size: int = 24, use_random: bool = False):
    # load model
    model = load_tuned_qa_model(model_dir=model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_config_path = os.path.join(model_dir, "train_config.json")
    bits_config_list = json.load(open(train_config_path)).get("bits_config_list", None)
    assert bits_config_list is not None

    # load validation dataset
    _, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
        "rajpurkar/squad",
        batch_size=batch_size,
    )

    opt_eval_bits_config = {
        "default": bits_config_list[-1]["default"],
        "layers": {},
    }

    # Search sequentially or randomly based on flag `use_random`
    layers = list(range(12))
    if use_random:
        random.shuffle(layers)
    for layer_idx in layers:
        eval_bits_config_list = [copy.deepcopy(opt_eval_bits_config) for _ in range(len(bits_config_list) - 1)]
        for i, bits_config in enumerate(bits_config_list[:-1]):
            low_bits_config = bits_config["default"]
            eval_bits_config_list[i]["layers"].update({f"layer_{layer_idx}": low_bits_config})
        
        # inference and get exact match scores
        scores = inference(
            model,
            eval_data_loader=eval_data_loader,
            examples=eval_examples,
            features=eval_dataset,
            device=device,
            exp_name=f"greedy_search_{'rdm' if use_random else 'seq'}",
            eval_bits_config_list=eval_bits_config_list,
        )
        em_scores = [score["exact_match"] for score in scores]

        assert len(em_scores) == len(bits_config_list) - 1

        # compute utilities (higher is better)
        utilities = []
        for i, em in enumerate(em_scores):
            acc_norm = em / EM_BASELINE
            eff_norm = SIZE_RATIO_MAP[f"w{bits_config_list[i]['default']['attn']['w_bits']}"]
            u = alpha * acc_norm + (1.0 - alpha) * (1.0 - eff_norm)
            utilities.append(u)

        opt_idx = utilities.index(max(utilities))
        opt_bits_config = bits_config_list[opt_idx]["default"]
        opt_eval_bits_config["layers"].update({f"layer_{layer_idx}": opt_bits_config})

    with open(output_path, "w") as f:
        json.dump(opt_eval_bits_config, f, indent=4)

def random_search(bits_config_list, output_path: str, repeats: int = 1):
    """
    Configure each model layer with a bit-width randomly selected from the input `bits_config_list`. A data example can be found in `./configs/random_search_bits.json`.
    """
    eval_bits_config_list = [{"default": bits_config_list[-1]["default"], "layers": {}} for _ in range(repeats)]
    
    num_layers = 12
    for eval_bits_config in eval_bits_config_list:
        for layer_idx in range(num_layers):
            # Pick a random bits_config
            random_bits_config = random.choice(bits_config_list[:-1])
            eval_bits_config["layers"][f"layer_{layer_idx}"] = random_bits_config["default"]
    with open(output_path, "w") as f:
        result = dict()
        result["eval_bits_config_list"] = eval_bits_config_list
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    if args.search_method == "greedy":
        greedy_search(
            model_dir=args.model_dir,
            output_path=args.output_path,
            alpha=args.alpha,
            batch_size=args.batch_size,
            use_random=args.use_random,
        )
    elif args.search_method == "random":
        bits_config_list = json.load(open(args.config_path))["bits_config_list"]
        random_search(bits_config_list=bits_config_list, output_path=args.output_path, repeats=args.repeats)
    else:
        raise ValueError(f"Unknown mode: {args.search_method}")
