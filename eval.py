import os
import numpy as np
from typing import List, TypedDict
from dataloader import create_data_loaders
from metrics import eval_squad
from utils import load_tuned_qa_model, update_model_bits_config as update_model_bits
from utils import (
    create_and_fill_np_array,
    format_preds,
)
from tqdm import tqdm
import torch
import argparse
import json


class EvalScore(TypedDict):
    exact_match: float
    f1: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDQ-GPT2 Model with specific configuration")
    parser.add_argument("--model_dir", type=str, default="model/fine_tuned_model_sp")  # include model and adapter
    parser.add_argument("--config_path", type=str, default="configs/random_search_bits.json")
    parser.add_argument("--output_path", type=str, default="output/greedy_search_seq.json")
    return parser.parse_args()


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
                # 1) answers that are out-of-scope. either because the indices are out of bounds or correspond to part of the input_ids that are not in the context.
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


# TODO: add eval func and script
if __name__ == "__main__":
    args = parse_args()
    result = dict()

    output_path = args.ouput_path
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_tuned_qa_model(model_dir=args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load eval bits config list
    config = json.load(open(args.config_path))
    eval_bits_config_list = config["eval_bits_config_list"]
    if hasattr(config, "bits_config"):
        bits_config = config["bits_config"]
        update_model_bits(model, bits_config=bits_config)
    
    # Add configs to result
    result["eval_config"] = args.__dict__
    result["model_config"] = model.config.to_dict()
    result["eval_bits_config_list"] = eval_bits_config_list

    # load validation dataset
    _, eval_data_loader, eval_examples, eval_dataset = create_data_loaders(
        "rajpurkar/squad",
        batch_size=64,
    )

    # Evaluation
    scores = inference(
        model,
        eval_data_loader=eval_data_loader,
        examples=eval_examples,
        features=eval_dataset,
        device=device,
        exp_name="eval",
        eval_bits_config_list=eval_bits_config_list,
    )
    
    # Save result
    result["scores"] = scores
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    
