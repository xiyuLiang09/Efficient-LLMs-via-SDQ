from functools import partial
from typing import Dict, Tuple
from datasets import Dataset

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GPT2TokenizerFast,
)


def preprocess_for_train(examples, tokenizer, max_seq_length=512, stride=128) -> Dict:
    """ Dataset preprocess func for training."""
    eos_token_id = tokenizer.eos_token_id
    # strip whitespaces on the left of questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    for i in range(len(tokenized_examples["input_ids"])):
        tokenized_examples["input_ids"][i] = [eos_token_id] + tokenized_examples["input_ids"][i]
        tokenized_examples["attention_mask"][i] = [1] + tokenized_examples["attention_mask"][i]
        tokenized_examples["offset_mapping"][i] = [(0, 0)] + tokenized_examples["offset_mapping"][i]

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_token_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sequence_ids = [None] + sequence_ids

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_token_index)
            end_positions.append(cls_token_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = answers["answer_start"][0] + len(answers["text"][0])

            # Find the start of the context
            context_start_index = 0
            while sequence_ids[context_start_index] != 1:
                context_start_index += 1
            # Find the end of the context
            context_end_index = len(input_ids) - 1
            while sequence_ids[context_end_index] != 1:
                context_end_index -= 1

            # If the answer is not fully inside the context, label it with [CLS] index
            if offset[context_start_index][0] > end_char or offset[context_end_index][1] < start_char:
                start_positions.append(cls_token_index)
                end_positions.append(cls_token_index)
            else:
                # Otherwise find the start and end of the answer
                idx = context_start_index
                while idx <= context_end_index and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end_index
                while idx >= context_start_index and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


def preprocess_for_train2eval(examples, tokenizer, max_seq_length=512, stride=128) -> Dict:
    """Retains `offset_mapping` and `context_positions`, which are used for evaluating each batch during training."""
    eos_token_id = tokenizer.eos_token_id
    # strip whitespaces on the left of questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    for i in range(len(tokenized_examples["input_ids"])):
        tokenized_examples["input_ids"][i] = [eos_token_id] + tokenized_examples["input_ids"][i]
        tokenized_examples["attention_mask"][i] = [1] + tokenized_examples["attention_mask"][i]
        tokenized_examples["offset_mapping"][i] = [(0, 0)] + tokenized_examples["offset_mapping"][i]

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.get("offset_mapping")

    start_positions = []
    end_positions = []
    context_positions = []
    for i, offset in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_token_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sequence_ids = [None] + sequence_ids

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_token_index)
            end_positions.append(cls_token_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = answers["answer_start"][0] + len(answers["text"][0])

            # Find the start of the context
            context_start_index = 0
            while sequence_ids[context_start_index] != 1:
                context_start_index += 1
            context_positions.append(context_start_index)
            # Find the end of the context
            context_end_index = len(input_ids) - 1
            while sequence_ids[context_end_index] != 1:
                context_end_index -= 1

            # If the answer is not fully inside the context, label it with [CLS] index
            if offset[context_start_index][0] > end_char or offset[context_end_index][1] < start_char:
                start_positions.append(cls_token_index)
                end_positions.append(cls_token_index)
            else:
                # Otherwise find the start and end of the answer
                idx = context_start_index
                while idx <= context_end_index and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end_index
                while idx >= context_start_index and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["context_positions"] = context_positions

    return tokenized_examples


def preprocess_for_valid(examples, tokenizer, max_seq_length=512, stride=128) -> Dict:
    """ Dataset preprocessing for validation.
    """
    eos_token_id = tokenizer.eos_token_id
    # strip whitespaces on the left of questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    for i in range(len(tokenized_examples["input_ids"])):
        tokenized_examples["input_ids"][i] = [eos_token_id] + tokenized_examples["input_ids"][i]
        tokenized_examples["attention_mask"][i] = [1] + tokenized_examples["attention_mask"][i]
        tokenized_examples["offset_mapping"][i] = [(0, 0)] + tokenized_examples["offset_mapping"][i]

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    example_id = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sequence_ids = [None] + sequence_ids

        sample_index = sample_mapping[i]
        example_id.append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    tokenized_examples["example_id"] = example_id

    return tokenized_examples


def create_data_loaders(path, batch_size: int = 4, batch_eval: bool = False):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.cls_token = tokenizer.bos_token

    raw_dataset = load_dataset(path)
    data_collator = DataCollatorWithPadding(tokenizer)
    column_names = raw_dataset["train"].column_names

    # Process for training
    train_examples = raw_dataset["train"]
    if batch_eval:
        _preprocess_for_train = partial(preprocess_for_train2eval, tokenizer=tokenizer)
    else:
        _preprocess_for_train = partial(preprocess_for_train, tokenizer=tokenizer)
    tokenized_train_dataset = train_examples.map(
        _preprocess_for_train, batched=True, batch_size=1000, remove_columns=column_names
    )
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    # Process for validation
    eval_examples = raw_dataset["validation"]
    _preprocess_for_valid = partial(preprocess_for_valid, tokenizer=tokenizer)
    eval_dataset = eval_examples.map(
        _preprocess_for_valid, batched=True, batch_size=1000, remove_columns=column_names
    )
    eval_dataloader = DataLoader(
        eval_dataset.remove_columns(["example_id", "offset_mapping"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_dataloader, eval_dataloader, eval_examples, eval_dataset


def get_column_names(dataset: Dataset) -> Tuple[str, str, str]:
    """
    Note that this function is designed for this dataset: https://huggingface.co/datasets/rajpurkar/squad

    It can be easily modified for other data sources.

    Args:
        dataset (Dataset): The non-processed dataset

    Returns:
        [`Tuple[str, str, str]`]: column names of question, context, and answer columns
    """
    column_names = dataset.column_names

    ctx_col_name = "context" if "context" in column_names else column_names[2]
    q_col_name = "question" if "question" in column_names else column_names[3]
    a_col_name = "answers" if "answers" in column_names else column_names[4]

    return q_col_name, ctx_col_name, a_col_name
