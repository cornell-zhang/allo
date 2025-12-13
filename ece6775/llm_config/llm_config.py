# LLM model config - shared for MoE and Attention+MoE

# config modes: switch_base_8*, mixtral_8x7b*, deepseek*, custom
DEFAULT_CONFIG_MODE = "switch_base_8_scaled_1_8"


def get_moe_config(config_mode=None):
    # get MoE config (works for standalone MoE and Attention+MoE)
    if config_mode is None:
        config_mode = DEFAULT_CONFIG_MODE
    
    config = {}
    
    if config_mode == "switch_base_8":
        # Google Switch-Base-8 (original)
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 768,  # hidden_size
            "output_dim": 768,  # hidden_size
            "num_experts": 8,
            "k": 1,  # Top-1 MoE
            "hidden_dim": 2048,
        }
    elif config_mode == "switch_base_8_scaled_2_3":
        # scaled to 2/3
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 512,  # 768 * 2/3
            "output_dim": 512,
            "num_experts": 2,
            "k": 1,  # Top-1 MoE
            "hidden_dim": 1024,
        }
    elif config_mode == "switch_base_8_scaled_1_2":
        # scaled to 1/2
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 384,  # 768 * 1/2
            "output_dim": 384,
            "num_experts": 2,
            "k": 1,  # Top-1 MoE
            "hidden_dim": 1024,
        }
    elif config_mode == "switch_base_8_scaled_1_4":
        # scaled to 1/4
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 192,  # 768 * 1/4
            "output_dim": 192,
            "num_experts": 2,
            "k": 1,  # Top-1 MoE
            "hidden_dim": 512,
        }
    elif config_mode == "switch_base_8_scaled_1_8":
        # scaled to 1/8
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 96,  # 768 * 1/8
            "output_dim": 96,
            "num_experts": 2,  # changed to 2 for testing (original: 8)
            "k": 1,
            "hidden_dim": 256,
        }
    elif config_mode == "mixtral_8x7b":
        # Mixtral-8x7B (original)
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 4096,  # hidden_size
            "output_dim": 4096,  # hidden_size
            "num_experts": 8,
            "k": 2,  # Top-2 MoE
            "hidden_dim": 14336,
        }
    elif config_mode == "mixtral_8x7b_scaled_1_8":
        # scaled to 1/8 (Top-2 MoE)
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 512,  # 4096 / 8
            "output_dim": 512,
            "num_experts": 8,
            "k": 2,  # Top-2 MoE (real Mixtral uses Top-2)
            "hidden_dim": 2048,
        }
    elif config_mode == "deepseek":
        # DeepSeek MoE-16B (needs significant resources)
        config = {
            "batch_size": 1,
            "seq_len": 128,  # Can be adjusted, model supports up to 4096
            "input_dim": 2048,  # hidden_size
            "output_dim": 2048,  # hidden_size
            "num_experts": 64,
            "k": 6,
            "hidden_dim": 10944,
        }
    elif config_mode == "deepseek_scaled":
        # scaled version
        config = {
            "batch_size": 1,
            "seq_len": 128,
            "input_dim": 512,  # 2048 / 4
            "output_dim": 512,
            "num_experts": 8,  # 64 / 8
            "k": 1,  # 6 -> 1 (current implementation supports k=1)
            "hidden_dim": 2048,
        }
    else:  # custom
        # small test config
        config = {
            "batch_size": 4,
            "seq_len": 4,
            "input_dim": 64,
            "output_dim": 64,
            "num_experts": 2,
            "k": 1,  # Top-1 MoE
            "hidden_dim": 256,
        }
    
    return config


def print_config_info(config_mode, config):
    # print config info
    print("=" * 60)
    print(f"LLM Configuration (MoE)")
    print("=" * 60)
    print(f"Model: {config_mode}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['seq_len']}")
    print(f"Input dimension: {config['input_dim']}")
    print(f"Output dimension: {config['output_dim']}")
    print(f"Number of experts: {config['num_experts']}")
    print(f"Top-k: {config['k']}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print("=" * 60)

