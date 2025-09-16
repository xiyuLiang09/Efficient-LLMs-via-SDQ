import torch
import torch.nn as nn

from typing import Dict

from .layers import LoraLayer, SDQLoraLinear, LoraConfig
from quantize.quantizer import QuantizeLinear


def replace_qlinear_with_loralayer(
    model: nn.Module,
    lora_bits_configs: Dict[str, Dict[str, int | float]],
) -> torch.nn.Module:
    """
    Replace all QuantizeLinear layers in the model with SDQLoraLinear

    Args:
        model: original model (with QuantizeLinear layers)
        lora_bits_configs: different bits configs for different adapters
    """
    # Record QuantizeLinears to be replaced and their parent modules.
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeLinear):
            # e.g. if "layer.0.attn.q_proj" -> parent_name = "layer.0.attn", target_name = "q_proj"
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            target_name = name.split(".")[-1]
            modules_to_replace.append((parent_name, target_name, module))

    # Replace
    for parent_name, target_name, quant_module in modules_to_replace:
        # aim to match the corresponding Lora config based on the different per-layer quant bits.
        w_bits = quant_module.w_bits
        a_bits = quant_module.a_bits
        adapter_name = f"lora_{w_bits}_{a_bits}" 
        if adapter_name not in lora_bits_configs:
            adapter_name = "default"

        lora_layer = SDQLoraLinear(
            base_layer=quant_module,
            active_adapter=adapter_name,
            lora_bits_configs=lora_bits_configs,
            init_lora_weights=True, 
        )

        parent_module = model
        if parent_name:
            # Recursively obtain the parent_module(nn.Module) based on parent_name(str).
            # e.g. "layer.0.attn" -> model.layer.0.attn
            for part in parent_name.split("."):
                parent_module = getattr(parent_module, part)

        setattr(parent_module, target_name, lora_layer)

    return model

# ref: https://github.com/microsoft/LoRA/blob/main/loralib/utils.py
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

