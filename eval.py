# import evaluate
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, TypedDict
from metrics import eval_squad
from utils import update_model_bits_config as update_model_bits
from utils import (
    create_and_fill_np_array,
    format_preds,
    load_tuned_model,
    parse_bits_config_list,
)
from loralib.utils import replace_qlinear_with_loralayer, lora_state_dict
from tqdm import tqdm
import torch
import argparse
import json


class EvalScore(TypedDict):
    exact_match: float
    f1: float


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./model/fine_tuned_model_sp")  # include model and adapter
    parser.add_argument("--low_bit", type=int, default=4)
    parser.add_argument("--high_bit", type=int, default=16)
    parser.add_argument("--exp_name", type=str)  # will be added to the name of json files that contain output predicions
    # parser.add_argument("--output_path", type=str, default="./exp_outputs") # path to save the eval scores
    return parser.parse_args()


def layer_wise_group(model_name: str, high_bits: int, low_bits: int):
    """
    Devide all layers into `num_layer_group` groups, and then allocate diffenrent bit-widths to each group. Return different bits config dict for each experiment.

    """
    # define the number of layers in the model
    if model_name == "gpt2-base":
        num_layer_group = 4  # 0-2, 3-5, 6-8, 9-11

    if high_bits is not None and low_bits is not None:
        assert high_bits > low_bits

    layers_per_group = 12 // num_layer_group
    layer_groups = []
    for i in range(num_layer_group):
        start = i * layers_per_group
        end = start + layers_per_group - 1
        layer_groups.append(f"{start}-{end}")

    configs = []

    # default = high bit-width
    for i in range(num_layer_group):
        config = {
            "default": {
                "attn": {"w_bits": high_bits, "a_bits": high_bits, "kv_bits": high_bits},
                "mlp": {"w_bits": high_bits, "a_bits": high_bits},
            },
            "layers": {},
        }

        config["layers"][layer_groups[i]] = {
            "attn": {"w_bits": low_bits, "a_bits": low_bits, "kv_bits": low_bits},
            "mlp": {"w_bits": low_bits, "a_bits": low_bits},
        }

        configs.append(config)

    # default = low bit-width
    for i in range(num_layer_group):
        config = {
            "default": {
                "attn": {"w_bits": low_bits, "a_bits": low_bits, "kv_bits": low_bits},
                "mlp": {"w_bits": low_bits, "a_bits": low_bits},
            },
            "layers": {},
        }

        config["layers"][layer_groups[i]] = {
            "attn": {"w_bits": high_bits, "a_bits": high_bits, "kv_bits": high_bits},
            "mlp": {"w_bits": high_bits, "a_bits": high_bits},
        }

        configs.append(config)

    parse_bits_config_list(configs)
    return configs


def _inference(model, eval_data_loader, examples, features, device, exp_name) -> EvalScore:
    """
    Core logic for forward inference on the entire evaluation dataset (not per batch).
    """
    model.eval()

    cfg_i_start_logits = []
    cfg_i_end_logits = []

    pbar = tqdm(eval_data_loader, desc="Evaluating")
    for batch in pbar:
        with torch.no_grad():
            batch = {k: batch[k].to(device) for k in batch.keys()}
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            cfg_i_start_logits.append(start_logits.cpu().numpy())
            cfg_i_end_logits.append(end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in cfg_i_start_logits])
    st_logits_concat = create_and_fill_np_array(cfg_i_start_logits, features, max_len)
    ed_logits_concat = create_and_fill_np_array(cfg_i_end_logits, features, max_len)

    del cfg_i_start_logits
    del cfg_i_end_logits

    outputs_np = (st_logits_concat, ed_logits_concat)

    prediction = format_preds(
        examples=examples,
        features=features,
        predictions=outputs_np,
        stage=exp_name,
    )  # logits -> text (EvalPrediction)
    score = eval_squad(
        predictions=prediction.predictions, references=prediction.label_ids
    )  # {'exact_match': 0.0, 'f1': 0.0}
    return score


def inference(
    model, eval_data_loader, examples, features, device, exp_name, eval_bits_config_list=None
) -> EvalScore | List[EvalScore]:
    """
    Evaluation on the given model with support for multiple quant bits configs.

    By default, it performs evaluation once with the current model setup. If a list of per-layer quantization/Lora configs is provided, the model will be updated and evaluated separately for each configuration.

    Args:
        - model (torch.nn.Module): The tuned model to evaluate.
        - eval_data_loader (:obj:`List[datasets.Dataset]`): The evaluation dataset.
        - examples (:obj:`List[datasets.Dataset]`): The non-processed dataset.
        - features (:obj:`List[datasets.Dataset]`): The tokenized dataset.
        - device (:obj:`str`): device to run evaluation.
        - eval_bits_config (:obj:`List[Dict] | None`): A list of (quant bits and lora) configs for each layer.

    Returns:
        If there are `n` bits configs, return a list of scores which contains `n` elements.
        Each element has the shape: {"exact_match": float, "f1": float}
    """
    model.eval()
    if eval_bits_config_list is None:
        return _inference(model, eval_data_loader, examples, features, device, exp_name)
    else:
        score_list = []
        for bits_config in eval_bits_config_list:
            update_model_bits(model, bits_config)
            score_i = _inference(model, eval_data_loader, examples, features, device, exp_name)
            score_list.append(score_i)
        return score_list


def compute_score_per_batch(
    tokenizer,
    batch,
    outputs,
    offset_mapping,
    context_positions,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """Compute evaluation metrics per batch"""
    predictions = []
    references = []
    batch_size = len(batch["input_ids"])

    for i in range(batch_size):
        # Generate references
        input_ids = batch["input_ids"][i]
        start_position = batch["start_positions"][i]
        end_position = batch["end_positions"][i]
        context = tokenizer.decode(input_ids[context_positions[i] :])
        s_char, _ = offset_mapping[i][start_position]
        _, e_char = offset_mapping[i][end_position]
        references.append({"id": str(i), "answer": {"text": [context[s_char:e_char]], "answer_start": [s_char]}})

        # Generate predictions
        start_logits = outputs["start_logits"][i].detach().cpu().numpy()
        end_logits = outputs["end_logits"][i].detach().cpu().numpy()
        offsets = offset_mapping[i]
        prelim_preds = []

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
                    start_idx >= len(offsets)
                    or end_idx >= len(offsets)
                    or offsets[start_idx] is None
                    or offsets[end_idx] is None
                ):
                    continue

                # 2) answers with a length that is out of [0, max_answer_length]
                if end_idx < start_idx or (end_idx - start_idx + 1) > max_answer_length:
                    continue

                s_char, _ = offsets[start_idx]
                _, e_char = offsets[end_idx]

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
        for pred in preds_sorted:
            offset = pred.pop("offsets")
            pred["text"] = context[offset[0] : offset[1]]

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
        scores = np.array([pred.pop("score") for pred in preds_sorted])
        exp_scores = np.exp(scores - np.max(scores))  # exp(x - Logsumexp(x))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, preds_sorted):
            pred["probability"] = float(prob)

        # Pick the best prediction (only for SQuAD v1, because we omit the "no answer" option)
        predictions.append({"id": str(i), "prediction_text": preds_sorted[0]["text"]})

    return eval_squad(predictions=predictions, references=references)


if __name__ == "__main__":
    # from transformers import GPT2Tokenizer
    from config import Config
    from model.configuration_sdq import SDQConfig
    from model.modeling_sdq_gpt2 import GPT2ForQuestionAnswering
    from dataloader import create_data_loaders

    args = parse_args()
    high_bit = args.high_bit
    low_bit = args.low_bit
    exp_name = args.exp_name
    model_dir = args.tuned_ckpt_dir  # local tuned_ckpt dir

    # generate bits config list
    per_layer_configs = layer_wise_group(model_name="gpt2-base", high_bits=high_bit, low_bits=low_bit)
    # print(json.dumps(per_layer_configs, indent=4))

    # load model
    sdq_config = SDQConfig.from_pretrained("gpt2")
    sdq_config.bits_config = per_layer_configs[-1]
    model = GPT2ForQuestionAnswering(config=sdq_config)
    replace_qlinear_with_loralayer(model, Config.lora_bits_configs)
    load_tuned_model(model, model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load validation dataset
    _, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
        "rajpurkar/squad",
        batch_size=Config.batch_size,
    )

    # evaluation
    scores = inference(
        model,
        eval_data_loader=eval_data_loader,
        examples=eval_examples,
        features=eval_dataset,
        eval_bits_config_list=per_layer_configs,
        device=device,
        exp_name=exp_name,
    )
    print(scores)

"""
python eval.py \
    --tuned_ckpt_dir ./model/fine_tuned_model_sp \
    --low_bit 4 \
    --high_bit 16 \
    --exp_name h16_l4
python eval.py \
    --tuned_ckpt_dir model/tuned_gpt2_sp/2j00nb2k/ckpt_2 \
    --low_bit 4 \
    --high_bit 16 \
    --exp_name h16_l4_2j00nb2k
"""
