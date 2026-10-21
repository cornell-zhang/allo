# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# check MoE model config from Hugging Face
# usage: python check_llm_config.py <model_name>

import sys
from transformers import AutoConfig


def print_moe_config(model_name):
    # print MoE model config
    try:
        print(f"Loading model configuration: {model_name}")
        print("=" * 80)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Print basic configuration
        print("\n[Basic Architecture Parameters]")
        print(
            f"Model type: {config.model_type if hasattr(config, 'model_type') else 'N/A'}"
        )
        print(
            f"Vocabulary size: {config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'}"
        )
        print(
            f"Max position embeddings: {config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 'N/A'}"
        )

        # Print hidden layer dimensions
        print("\n[Hidden Layer Dimensions]")
        if hasattr(config, "hidden_size"):
            print(f"Hidden size (hidden_size): {config.hidden_size}")
        if hasattr(config, "d_model"):
            print(f"Model dimension (d_model): {config.d_model}")
        if hasattr(config, "n_embd"):
            print(f"Embedding dimension (n_embd): {config.n_embd}")

        # Print MoE related parameters
        print("\n[MoE Expert Parameters]")
        if hasattr(config, "num_local_experts"):
            print(
                f"Number of local experts (num_local_experts): {config.num_local_experts}"
            )
        if hasattr(config, "num_experts"):
            print(f"Number of experts (num_experts): {config.num_experts}")
        if hasattr(config, "num_experts_per_tok"):
            print(
                f"Number of experts per token (num_experts_per_tok): {config.num_experts_per_tok}"
            )
        if hasattr(config, "num_experts_to_select"):
            print(
                f"Number of experts to select (num_experts_to_select): {config.num_experts_to_select}"
            )

        # Print feed-forward network parameters
        print("\n[Feed-Forward Network Parameters]")
        if hasattr(config, "intermediate_size"):
            print(f"Intermediate size (intermediate_size): {config.intermediate_size}")
        if hasattr(config, "ffn_dim"):
            print(f"FFN dimension (ffn_dim): {config.ffn_dim}")
        if hasattr(config, "n_inner"):
            print(f"Inner dimension (n_inner): {config.n_inner}")

        # Print attention parameters
        print("\n[Attention Parameters]")
        if hasattr(config, "num_attention_heads"):
            print(
                f"Number of attention heads (num_attention_heads): {config.num_attention_heads}"
            )
        if hasattr(config, "num_heads"):
            print(f"Number of heads (num_heads): {config.num_heads}")
        if hasattr(config, "n_head"):
            print(f"Number of heads (n_head): {config.n_head}")

        # Print layer count
        print("\n[Layer Parameters]")
        if hasattr(config, "num_hidden_layers"):
            print(
                f"Number of hidden layers (num_hidden_layers): {config.num_hidden_layers}"
            )
        if hasattr(config, "num_layers"):
            print(f"Number of layers (num_layers): {config.num_layers}")
        if hasattr(config, "n_layer"):
            print(f"Number of layers (n_layer): {config.n_layer}")

        # Print all configuration (for debugging)
        print("\n[Full Configuration Information]")
        print("-" * 80)
        for key, value in config.to_dict().items():
            if isinstance(value, (int, float, str, bool, type(None))):
                print(f"{key}: {value}")
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                print(f"{key}: {value}")

        print("\n" + "=" * 80)
        print("Configuration loaded successfully!")

        # Provide suggested test parameters
        print("\n[Suggested Test Parameters (for Allo Implementation)]")
        hidden_size = getattr(
            config,
            "hidden_size",
            getattr(config, "d_model", getattr(config, "n_embd", None)),
        )
        num_experts = getattr(
            config, "num_local_experts", getattr(config, "num_experts", None)
        )
        intermediate_size = getattr(
            config,
            "intermediate_size",
            getattr(config, "ffn_dim", getattr(config, "n_inner", None)),
        )
        num_experts_per_tok = getattr(
            config, "num_experts_per_tok", getattr(config, "num_experts_to_select", 1)
        )

        if hidden_size:
            print(f"input_dim = {hidden_size}  # Input dimension")
        if intermediate_size:
            print(f"hidden_dim = {intermediate_size}  # Expert hidden dimension")
        if hidden_size:
            print(f"output_dim = {hidden_size}  # Output dimension")
        if num_experts:
            print(f"num_experts = {num_experts}  # Number of experts")
        if num_experts_per_tok:
            print(
                f"k = {num_experts_per_tok}  # Top-k (number of experts activated per token)"
            )
        print(f"batch_size = 1  # Batch size (adjustable as needed)")
        print(
            f"seq_len = 128  # Sequence length (adjustable as needed, actual model may support longer)"
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print(
            "\nHint: Make sure transformers library is installed: pip install transformers"
        )
        print(
            "If the model requires trust_remote_code=True, make sure to trust the model"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n[Common MoE Model Examples]")
        print("1. DeepSeek MoE:")
        print("   python check_moe_config.py deepseek-ai/deepseek-moe-16b-base")
        print("\n2. Mixtral:")
        print("   python check_moe_config.py mistralai/Mixtral-8x7B-v0.1")
        print("\n3. Switch Transformer:")
        print("   python check_moe_config.py google/switch-base-8")
        print("\n4. GShard:")
        print("   python check_moe_config.py google/gshard-gpt")
        sys.exit(1)

    model_name = sys.argv[1]
    print_moe_config(model_name)
