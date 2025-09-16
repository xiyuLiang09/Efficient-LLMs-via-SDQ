import collections
import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from tqdm import tqdm
from transformers.trainer_utils import EvalPrediction

from loralib.utils import lora_state_dict, replace_qlinear_with_loralayer
from model.configuration_sdq import SDQConfig
from model.modeling_sdq_gpt2 import GPT2ForQuestionAnswering
from quantize.quantizer import QuantizeLinear


def parse_bits_config_list(bits_config_list):
    """
    Parse `bits_config_list` to support batch parameter configuration for multiple layers using a format like `1-5`.

    Args:
        bits_config_list (list): a list of bits configs.
    Returns:
        list: a list of parsed bits configs.
    """
    for bits_config in bits_config_list:
        layers_bits_config = bits_config.get("layers", {})
        res = EasyDict()
        for layer_range, config in layers_bits_config.items():
            layer_range = list(map(int, layer_range.split("-")))
            if len(layer_range) == 1:
                layer_idx = layer_range[0]
                res.update({f"layer_{layer_idx}": config})
            elif len(layer_range) == 2 and layer_range[0] <= layer_range[1]:
                for layer_idx in range(layer_range[0], layer_range[1] + 1):
                    res.update({f"layer_{layer_idx}": config})
            else:
                raise ValueError("Invalid layer range")
        bits_config["layers"] = res


def get_layer_bits_config(bits_config, layer_idx: int) -> Dict[str, Dict[str, int]]:
    """
    Get layer-specific bits configuration for a given layer index.

    Args:
        bits_config: The bits configuration for the model.
        layer_idx(int): The index of the layer.

    Returns:
        A dictionary containing the layer-specific bits configuration.
    """
    layer_bits_config = copy.deepcopy(bits_config["default"])
    layer_name = f"layer_{layer_idx}"
    if layer_name in bits_config["layers"].keys():
        layer_bits_config.update(bits_config["layers"][layer_name])
    return layer_bits_config


def update_model_bits_config(model, bits_config) -> None:
    bits_config = copy.deepcopy(bits_config)
    for layer_idx, block in enumerate(model.transformer.h):
        # Get layer-specific config or default config
        layer_bits_config = get_layer_bits_config(bits_config, layer_idx)
        attn_config = layer_bits_config["attn"]
        mlp_config = layer_bits_config["mlp"]

        # Update attention layer bits configuration
        block.attn.kv_bits = attn_config["kv_bits"]
        for m in block.attn.modules():
            if isinstance(m, QuantizeLinear):
                m.w_bits = attn_config["w_bits"]
                m.a_bits = attn_config["a_bits"]

        # Update MLP layer bits configuration
        for m in block.mlp.modules():
            if isinstance(m, QuantizeLinear):
                m.w_bits = mlp_config["w_bits"]
                m.a_bits = mlp_config["a_bits"]


def cyclic_adjust_precision(cyclic_num_bits_schedule, iter, cyclic_period):
    assert len(cyclic_num_bits_schedule) == 2

    num_bit_min = cyclic_num_bits_schedule[0]
    num_bit_max = cyclic_num_bits_schedule[1]

    # ref: `http://arxiv.org/abs/2101.09868` (cyclic scheduling method from the paper's example)
    num_bits = np.rint(
        num_bit_min
        + 0.5 * (num_bit_max - num_bit_min) * (1 + np.cos(np.pi * ((iter % cyclic_period) / cyclic_period) + np.pi))
    )
    return int(num_bits)


def cyclic_adjust_bits_config(bits_config_list: List, current_iter: int, cyclic_period: int) -> int:
    """
    Return the index of bits_config_list for the current iteration
    """
    steps_per_bits = cyclic_period / len(bits_config_list)
    return int((current_iter % cyclic_period) / steps_per_bits)


def save_tuned_model(model: GPT2ForQuestionAnswering, output_dir: str | Path) -> None:
    """
    Saves a LoRA-tuned model by separating and storing the base model weights and LoRA adapter weights.

    This function saves two components of a fine-tuned model:
    1. `model.pt`: The full state dictionary of the base model (including any modifications from tuning)
    2. `adapter.pt`: The LoRA-specific adapter weights, isolated from the base model parameters

    Args:
        model: The LoRA-tuned model. Should be a PEFT-wrapped model or compatible
            with the `lora_state_dict` function to extract adapter weights.
        output_dir: Path to the directory where the model files will be saved.
            The directory will be created if it does not already exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)
    model_path = os.path.join(output_dir, "model.pt")
    adapter_path = os.path.join(output_dir, "adapter.pt")
    torch.save(model.state_dict(), model_path)
    torch.save(lora_state_dict(model), adapter_path)


def load_tuned_qa_model(model_dir: str | Path, active_lora: bool = True) -> GPT2ForQuestionAnswering:
    """
    Loads a LoRA-tuned model by restoring both base model weights and LoRA adapter weights.

    This function reconstructs a previously saved LoRA-tuned model by loading two components:
    1. `model.pt`: The full state dictionary of the base model (with tuning modifications)
    2. `adapter.pt`: The LoRA-specific adapter weights

    - The loaded weights are applied to the provided base model in sequence, first loading
    the base model weights followed by the LoRA adapter weights. Non-matching keys between
    the saved state dicts and the model are ignored by setting `strict=False`.

    Args:
        model: Base model architecture to load the weights into. Should match the
            architecture of the original model used for fine-tuning.
        model_dir: Path to the directory containing the saved model files (`model.pt`
            and `adapter.pt`).
    """
    model_path = os.path.join(model_dir, "model.pt")
    config_path = os.path.join(model_dir, "config.json")
    adapter_path = os.path.join(model_dir, "adapter.pt")

    # model_config = SDQConfig.from_pretrained("gpt2")
    model_config = SDQConfig.from_json_file(config_path)
    model = GPT2ForQuestionAnswering(config=model_config)

    if active_lora and hasattr(model_config, "lora_bits_configs"):
        replace_qlinear_with_loralayer(model, model_config.lora_bits_configs)
    model.load_state_dict(torch.load(model_path), strict=False)

    if active_lora and hasattr(model_config, "lora_bits_configs"):
        model.load_state_dict(torch.load(adapter_path), strict=False)

    return model


def load_tuned_model(model: nn.Module, model_dir: str | Path) -> None:
    """
    Loads a LoRA-tuned model by restoring both base model weights and LoRA adapter weights.

    This function reconstructs a previously saved LoRA-tuned model by loading two components:
    1. `model.pt`: The full state dictionary of the base model (with tuning modifications)
    2. `adapter.pt`: The LoRA-specific adapter weights

    - The loaded weights are applied to the provided base model in sequence, first loading
    the base model weights followed by the LoRA adapter weights. Non-matching keys between
    the saved state dicts and the model are ignored by setting `strict=False`.

    Args:
        model: Base model architecture to load the weights into. Should match the
            architecture of the original model used for fine-tuning.
        model_dir: Path to the directory containing the saved model files (`model.pt`
            and `adapter.pt`).
    """
    model_path = os.path.join(model_dir, "model.pt")
    adapter_path = os.path.join(model_dir, "adapter.pt")
    model.load_state_dict(torch.load(model_path), strict=False)
    model.load_state_dict(torch.load(adapter_path), strict=False)


# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)

    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array And after every iteration we have to change the step
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def postprocess_preds(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    # null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
):
    """
    Post-processing utilities for question-answering tasks(SQuAD v1 only).

    Args:
        examples (:obj:`List[datasets.Dataset]`): The non-preprocessed dataset.
        features (:obj:`List[datasets.Dataset]`): The preprocessed dataset.
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            `predictions[0]` contains the start logits and `predictions[1]` contains the end logits respectively.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions (and n_best predictions) will be saved with `prefix` in their file names.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    else:
        all_start_logits, all_end_logits = predictions

    if len(all_start_logits) != len(features):
        raise ValueError(f"Got {len(all_start_logits)} predictions and {len(features)} features.")

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # Containers for outputs
    all_preds = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    # TODO: add logger
    # logger.setLevel(log_level)
    # logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for examples_idx, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[examples_idx]
        prelim_preds = []

        for feature_idx in feature_indices:
            start_logits = all_start_logits[feature_idx]
            end_logits = all_end_logits[feature_idx]

            offset_mapping = features[feature_idx]["offset_mapping"]
            token_is_max_context = features[feature_idx].get(
                "token_is_max_context", None
            )  # TODO: maybe we need to delete this param.

            # Get the index of top-k start and end logits
            start_idxes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_idxes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            # Filtering
            for start_idx in start_idxes:
                for end_idx in end_idxes:
                    # 1) answers that are out-of-scope. either because
                    # the indices are out of bounds or
                    # correspond to part of the input_ids that are not in the context.
                    if (
                        start_idx >= len(offset_mapping)
                        or end_idx >= len(offset_mapping)
                        or offset_mapping[start_idx] is None
                        or offset_mapping[end_idx] is None
                    ):
                        continue

                    # 2) answers with a length that is out of [0, max_answer_length]
                    if end_idx < start_idx or (end_idx - start_idx + 1) > max_answer_length:
                        continue

                    s_char, _ = offset_mapping[start_idx]
                    _, e_char = offset_mapping[end_idx]

                    prelim_preds.append(
                        {
                            "offsets": (s_char, e_char),
                            "score": start_logits[start_idx] + end_logits[end_idx],
                            "start_logit": start_logits[start_idx],
                            "end_logit": end_logits[end_idx],
                        }
                    )

        # Only keep the best `n_best_size` predictions (descending order).
        preds_sorted = sorted(prelim_preds, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer in the original context.
        ctx = example["context"]
        for pred in preds_sorted:
            offsets = pred.pop("offsets")
            pred["text"] = ctx[offsets[0] : offsets[1]]

        # If there are no effective predictions, create a fake prediction to avoid failure.
        if len(preds_sorted) == 0 or (len(preds_sorted) == 1 and preds_sorted[0]["text"] == ""):
            preds_sorted.insert(
                0,
                {
                    "text": "empty",
                    "score": 0.0,
                    "start_logit": 0.0,
                    "end_logit": 0.0,
                },
            )

        # Compute softmax scores
        # TODO: why do we use torch.nn.functional.softmax here?
        scores = np.array([pred.pop("score") for pred in preds_sorted])
        exp_scores = np.exp(scores - np.max(scores))  # exp(x - Logsumexp(x))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, preds_sorted):
            pred["probability"] = float(prob)

        # Pick the best prediction (only for SQuAD v1, because we omit the "no answer" option)
        all_preds[example["id"]] = preds_sorted[0]["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in preds_sorted
        ]

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )

        # TODO: add logger
        # logger.info(f"Saving predictions to {prediction_file}.")
        print(f"Saving predictions to {prediction_file}")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_preds, ensure_ascii=False, indent=4) + "\n")

        # logger.info(f"Saving nbest_preds to {nbest_file}.")
        print(f"Saving nbest predictions to {nbest_file}")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_nbest_json, ensure_ascii=False, indent=4) + "\n")

    return all_preds


def format_preds(
    examples,
    features,
    predictions,
    n_best_size=20,
    max_answer_length=30,
    output_dir="exp_outputs",
    stage="eval",
    answer_col_name="answers",
):
    """
    Data Examples:
        - formatted_preds = [
            {
                'prediction_text': '1976',
                'id': '56e10a3be3433e1400422b22'
            }, ...
        ]
        - references = [
            {
                'id': '56e10a3be3433e1400422b22',
                'answers': {'answer_start': [79], 'text': ['1976']}
            }, ...
        ]
    """
    predictions = postprocess_preds(
        examples=examples,
        features=features,
        predictions=predictions,
        # TODO: these 4 input args should can be freely modified in the config file (e.g., n_best_size = config.n_best_size).
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        output_dir=output_dir,
        prefix=stage,
    )

    formatted_preds = [
        {
            "id": k,
            "prediction_text": v,
        }
        for k, v in predictions.items()
    ]
    references = [{"id": e["id"], "answer": e[answer_col_name]} for e in examples]

    return EvalPrediction(predictions=formatted_preds, label_ids=references)


def get_bits_config(num_bits: int):
    # According to the LLM-QAT paper, the minimum applicable bit-width of activation is 6;
    # values lower than 6 will severely affect model performance
    a_bits = max(num_bits, 6)
    return {
        "default": {
            "attn": {"w_bits": num_bits, "a_bits": a_bits, "kv_bits": 16},
            "mlp": {"w_bits": num_bits, "a_bits": a_bits},
        },
        "layers": {},
    }


"""Functions for computing modol size and kv size"""
SCALE_BITS_DEFAULT = 16  # bit width for storing scaling (symmetric quantization without zero point)
EMBED_FP_BITS = 32  # embedding stored as fp32 if not quantized (can be changed to 16)
LN_BITS = 16  # bit width for LayerNorm and bias (typically 16)


def compute_model_size(
    model,
    bits_cfg,
    *,
    scale_bits: int = SCALE_BITS_DEFAULT,
    embed_fp_bits: int = EMBED_FP_BITS,
    ln_bits: int = LN_BITS,
) -> float:
    """
    Calculate the approximate memory footprint of a model under a given bit-width configuration.

    Returns:
        float: Estimated memory size in bytes.
    """
    _layer_pat = re.compile(r"transformer\.h\.(\d+)\.")

    def _get_layer_idx(name: str):
        m = _layer_pat.search(name)
        return int(m.group(1)) if m else None

    def _which_block(name: str):
        if ".attn." in name:
            return "attn"
        if ".mlp." in name:
            return "mlp"
        return "mlp"

    total = 0.0

    # 1. Embedding (not quantized by default)
    wte = getattr(model.transformer, "wte", None)
    if wte is not None and hasattr(wte, "weight"):
        V, d = wte.weight.shape
        total += V * d * (embed_fp_bits / 8.0)
    wpe = getattr(model.transformer, "wpe", None)
    if wpe is not None and hasattr(wpe, "weight"):
        V, d = wpe.weight.shape
        total += V * d * (embed_fp_bits / 8.0)

    # 2. Process each Linear layer (including QuantizeLinear or normal Linear)
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            out_f, in_f = m.weight.shape

            # Get bit-width config from layer
            layer_idx = _get_layer_idx(name)
            blk = _which_block(name)
            layer_cfg = (
                bits_cfg["layers"].get(f"layer_{layer_idx}", bits_cfg["default"])
                if layer_idx is not None
                else bits_cfg["default"]
            )

            w_bits = layer_cfg[blk]["w_bits"]

            # 2.1 weight 
            total += out_f * in_f * (w_bits / 8.0)

            # 2.2 quantization scale
            if w_bits < 32:
                per_tensor = getattr(m, "weight_layerwise", False)
                groups = 1 if per_tensor else out_f  # per-tensor vs per-row
                total += groups * (scale_bits / 8.0)

            # 2.3 bias (if QuantizeLinear, default bias=False)
            if getattr(m, "bias", None) is not None:
                total += out_f * (ln_bits / 8.0)

    # 3) LayerNorm (typically fp16)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            n = m.weight.numel()
            total += 2 * n * (ln_bits / 8.0)  # weight + bias

    return float(total)


def compute_kv_size(model, bits_cfg, seq_len: int) -> float:
    """
    Estimate the KV cache size (seq_len related); only as an efficiency proxy, not part of "model size".
    Formula: each layer caches K and V --> 2 * n_head * L * d_head * (kv_bits/8)

    Returns: 
        float: KV cache size (in bytes)
    """
    cfg = model.transformer.config
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    d_model = cfg.n_embd
    d_head = d_model // n_head

    total = 0.0
    for l in range(n_layer):
        layer_cfg = bits_cfg["layers"].get(f"layer_{l}", bits_cfg["default"])
        kv_bits = layer_cfg["attn"].get("kv_bits", 16)
        total += 2 * n_head * seq_len * d_head * (kv_bits / 8.0)  # K+V

    return float(total)