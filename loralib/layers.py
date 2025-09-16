import math
from typing import Optional, TypedDict, Dict

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


class LoraConfig(TypedDict, total=False):
    r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: Optional[bool]


# Code adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
class LoraLayer:
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_adapters = []
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self._disable_adapters = False
        self.lora_bias: Dict[str, bool] = {}
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = (
                base_layer.num_embeddings,
                base_layer.embedding_dim,
            )
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape
                if hasattr(base_layer.weight, "ds_shape")
                else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(
                    f"Only same dim for query/key/value is supported as of now for {self.__class__}."
                )
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim

        self.in_features = in_features
        self.out_features = out_features

    def init_lora_modules(
        self,
        lora_bits_configs: Dict[str, LoraConfig],
        init_lora_weights,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        for _, config in lora_bits_configs.items():
            r = config["r"]
            if r <= 0:
                raise ValueError(
                    f"`r` should be a positive integer value but the value passed is {r}"
                )

        for adapter_name, config in lora_bits_configs.items():
            # w_bits, a_bits = map(int, adapter_name.split("_")[1:])
            r = config["r"]
            lora_alpha = config["lora_alpha"]
            lora_dropout = config["lora_dropout"]
            lora_bias = config.get("lora_bias", False)

            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update({adapter_name: lora_dropout_layer})
            # Actual trainable parameters
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
            self.lora_bias[adapter_name] = lora_bias

            self.scaling[adapter_name] = lora_alpha / r
            self.lora_adapters.append(adapter_name)

            if init_lora_weights:
                self.reset_lora_parameters(adapter_name, init_lora_weights)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)

        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)


class SDQLoraLinear(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        active_adapter: str,
        lora_bits_configs: Dict[str, LoraConfig],
        init_lora_weights: bool = True,
        disable_adapters: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self.active_adapter = active_adapter
        self.disable_adapters = disable_adapters
        self.lora_adapters = list(lora_bits_configs.keys())
        self.init_lora_modules(lora_bits_configs, init_lora_weights, **kwargs)

    def forward(self, x: torch.Tensor):
        result = self.base_layer(x)
        if self.disable_adapters:
            return result

        if hasattr(self.base_layer, "w_bits") and hasattr(self.base_layer, "a_bits"):
            w_bits = self.base_layer.w_bits
            a_bits = self.base_layer.a_bits
            adapter_name = f"lora_{w_bits}_{a_bits}"
            active_adapter = adapter_name if adapter_name in self.lora_adapters else "default"
        else:
            active_adapter = "default"

        torch_result_dtype = result.dtype

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        # x = self._cast_input_dtype(x, lora_A.weight.dtype)
        if x.dtype != lora_A.weight.dtype:
            x = x.to(lora_A.weight.dtype)

        result = result + lora_B(lora_A(dropout(x))) * scaling
        result = result.to(torch_result_dtype)
        return result

    def set_active_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

