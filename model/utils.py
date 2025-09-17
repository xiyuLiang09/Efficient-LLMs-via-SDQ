import torch
from transformers import GPT2ForQuestionAnswering


def get_linear_gpt2_weights(output_dir):
    """
    Function for transposing pre-trained GPT-2 weights to match the Linear layer,
    Have been verified to produce outputs consistent with Conv1D.
    """
    model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
    for layer in model.transformer.h:
        layer.attn.c_attn.weight = torch.nn.Parameter(
            layer.attn.c_attn.weight.transpose(0, 1).contiguous()
        )
        layer.attn.c_proj.weight = torch.nn.Parameter(layer.attn.c_proj.weight.transpose(0, 1).contiguous())
        layer.mlp.c_fc.weight = torch.nn.Parameter(layer.mlp.c_fc.weight.transpose(0, 1).contiguous())
        layer.mlp.c_proj.weight = torch.nn.Parameter(layer.mlp.c_proj.weight.transpose(0, 1).contiguous())
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args()
    if args.output_dir == "":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(current_dir, "linear_gpt2")

    get_linear_gpt2_weights(args.output_dir)
