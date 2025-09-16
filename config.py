from easydict import EasyDict
from utils import parse_bits_config_list

Config = EasyDict()

"""LoRA config for different quantization config"""
# r = [4, 8, 16, 32]
Config.lora_bits_configs = {
    "default": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},  # alpha = [4, 8, 16, 32]
    # "lora_4_4": {"r": 32, "lora_alpha": 64, "lora_dropout": 0.05},  # alpha = [32, 64]
    "lora_4_6": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},  # alpha = [16, 32]
    "lora_6_6": {"r": 16, "lora_alpha": 16, "lora_dropout": 0.08},  # alpha = [16, 32]
    "lora_6_8": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},  # alpha = [4, 8, 16, 32]
    "lora_8_8": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1}, # alpha = [4, 8, 16, 32]
    # "lora_32_32": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1}, # alpha = [4, 8, 16, 32]
}

"""Training bits config"""
Config.bits_config_list = [
    {
        "default": {"attn": {"w_bits": 4, "a_bits": 6, "kv_bits": 16}, "mlp": {"w_bits": 4, "a_bits": 6}},
        "layers": {},
    },
    {
        "default": {"attn": {"w_bits": 6, "a_bits": 6, "kv_bits": 16}, "mlp": {"w_bits": 6, "a_bits": 6}},
        "layers": {},
    },
    {
        "default": {"attn": {"w_bits": 6, "a_bits": 8, "kv_bits": 16}, "mlp": {"w_bits": 6, "a_bits": 8}},
        "layers": {},
    },
    {
        "default": {"attn": {"w_bits": 8, "a_bits": 8, "kv_bits": 16}, "mlp": {"w_bits": 8, "a_bits": 8}},
        "layers": {},
    },
    # {
    #     "default": {"attn": {"w_bits": 12, "a_bits": 12, "kv_bits": 16}, "mlp": {"w_bits": 12, "a_bits": 12}},
    #     "layers": {},
    # },
]

parse_bits_config_list(Config.bits_config_list)
Config.loss_scale = [1] * len(Config.bits_config_list)  # average
# Config.loss_scale = [8, 4, 8 / 3, 2, 1]

bits_config = {
    "default": {"attn": {"w_bits": 4, "a_bits": 4, "kv_bits": 16}, "mlp": {"w_bits": 4, "a_bits": 6}},
    "layers": {
        "layer_0": {"attn": {"w_bits": 4, "a_bits": 4, "kv_bits": 16}, "mlp": {"w_bits": 4, "a_bits": 6}},
        "layer_1": {"attn": {"w_bits": 4, "a_bits": 4, "kv_bits": 16}, "mlp": {"w_bits": 4, "a_bits": 6}},
    },
}
