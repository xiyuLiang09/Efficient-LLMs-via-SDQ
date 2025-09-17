import argparse
import json
import os
import random

import torch

import utils
from dataloader import create_data_loaders
from eval import inference
from utils import load_tuned_qa_model, update_model_bits_config

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
    parser.add_argument("--low_bits", type=int, default=4, help="Lowwer bits for greedy search")
    parser.add_argument("--high_bits", type=int, default=8, help="Higher bits for greedy search")
    parser.add_argument("--lower_bound", type=float, help="Lower bound for greedy search")
    parser.add_argument("--use_random", action="store_true", help="Whether to use random greedy search")
    parser.add_argument("--repeats", type=int, default=1, help="Number of random bits config")
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


def greedy_search(
    model_dir: str,
    output_path: str,
    low_bits: int,
    high_bits: int,
    lower_bound: float,
    alpha: float = 0.7,
    batch_size: int = 24,
    use_random: bool = False,
):
    """采用贪心搜索策略，逐步将模型每一层的量化位宽从 `high_bits` 降为 `low_bits`，直至模型的`em_score`达到 `lower_bound`"""
    # load model
    model = load_tuned_qa_model(model_dir=model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_config_path = os.path.join(model_dir, "train_config.json")
    bits_config_list = json.load(open(train_config_path)).get("bits_config_list", None)
    assert bits_config_list is not None

    # 初始化模型所有层为的量化位宽为 `high_bits`
    high_bits_config = utils.get_bits_config(high_bits)
    update_model_bits_config(model, high_bits_config)

    low_bits_config = utils.get_bits_config(low_bits)["default"]
    high_bits_config = high_bits_config["default"]

    # load validation dataset
    _, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
        "rajpurkar/squad",
        batch_size=batch_size,
    )
    
    num_layers = len(model.transformer.h)
    fixed_layers = set()  # 记录量化位宽已经固定为`low_bits`的层
    eval_bits_config_list = []

    em_score_base = EM_BASELINE # 记录替换前的 em_score，用于评估替换前后 em_score 的降低比例
    for step in range(num_layers):
        print(f"Step {step} is running......")
        print(f"Fixed layers: {fixed_layers}")
        testing_layers = set(range(num_layers)) - fixed_layers
        print(f"Testing layers: {testing_layers}")
        for i in testing_layers:
            eval_bits_config_list.append(
                {
                    "default": high_bits_config,
                    "layers": {
                        # 依次替换 `testing_layers` 中的层的量化比特位宽为 `low_bits`
                        f"layer_{i}": low_bits_config,
                        # 固定 `fixed_layers` 中的层的量化比特位宽为 `low_bits`
                        **{f"layer_{j}": low_bits_config for j in fixed_layers},
                    },
                }
            )
        scores = inference(
            model,
            eval_data_loader=eval_data_loader,
            examples=eval_examples,
            features=eval_dataset,
            device=device,
            exp_name=f"greedy_search_step_{step}",
            eval_bits_config_list=eval_bits_config_list,
        )
        em_scores = [score["exact_match"] for score in scores]
        print(em_scores)
        # compute utilities (higher is better)
        utilities = []
        for i, em in enumerate(em_scores):
            acc_norm = em / em_score_base
            eff_norm = SIZE_RATIO_MAP[f"w{low_bits}"] 
            u = alpha * acc_norm + (1.0 - alpha) * (1.0 - eff_norm)
            utilities.append(u)
        print(utilities)
        opt_idx = utilities.index(max(utilities))
        if em_scores[opt_idx] > lower_bound:
            fixed_layers.add(opt_idx)
            em_score_base = em_scores[opt_idx]
        else:
            break

    # 保存最终结果
    result = {
        "fixed_layers": fixed_layers,
        "em_score": em_score_base,
    }
    print(json.dumps(result, indent=4))
    result["bits_config"] = {
        "default": high_bits_config,
        "layers": {
            f"layer_{i}": low_bits_config,
            **{f"layer_{j}": low_bits_config for j in fixed_layers},
        },
    }

    result["eval_bits_config_list"] = [{"default": high_bits_config, "layers": {}}]

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

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
            low_bits=args.low_bits,
            high_bits=args.high_bits,
            lower_bound=args.lower_bound,
            alpha=args.alpha,
            batch_size=args.batch_size,
            use_random=args.use_random,
        )
    elif args.search_method == "random":
        bits_config_list = json.load(open(args.config_path))["bits_config_list"]
        random_search(bits_config_list=bits_config_list, output_path=args.output_path, repeats=args.repeats)
    else:
        raise ValueError(f"Unknown mode: {args.search_method}")

"""
python bit_search.py \
    --model_dir model/tuned_gpt2_sp/y5sonb2h \
    --search_method greedy \
    --output_path configs/greedy_search.json \
    --low_bits 4 \
    --high_bits 8 \
    --lower_bound 40.34 \
    --alpha 0.7 \
    --batch_size 24 \
"""