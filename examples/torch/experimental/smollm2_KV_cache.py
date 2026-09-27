# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete SmolLM2 prefill and token-decode example with KV caches.

The organization mirrors ``gptneo_KV_cache.py``: a traceable PyTorch model
describes cache semantics, generic leaf modules bridge mutations to Allo, and a
host loop performs autoregressive token selection. Two fixed-shape modules are
compiled: one for prompt prefill and one for single-token decode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

import allo
from allo.frontend.pytorch import QuantizationConfig
from examples.torch.smollm2 import (
    CausalSoftmax,
    RMSNorm,
    RepeatKV,
    SmolLM2Config,
    SmolLM2MLP,
    build_rope_tables,
    load_huggingface_config,
    load_huggingface_weights,
    smollm2_leaf_modules,
)


class PositionedRotaryEmbedding(nn.Module):
    """RoPE leaf selecting rows from a maximum-length table at runtime."""

    def __init__(self, max_sequence_length: int, head_dim: int, theta: float):
        super().__init__()
        cos, sin = build_rope_tables(max_sequence_length, head_dim, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        length = x.shape[1]
        cos = self.cos[position : position + length].unsqueeze(0)
        sin = self.sin[position : position + length].unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin


class KVCacheUpdate(nn.Module):
    """Functional PyTorch reference for a mutable KV-cache update."""

    def forward(
        self,
        values: torch.Tensor,
        cache: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        updated = cache.clone()
        updated[:, position : position + values.shape[1], :] = values
        return updated


def cached_leaf_modules() -> tuple[type[nn.Module], ...]:
    return smollm2_leaf_modules() + (PositionedRotaryEmbedding, KVCacheUpdate)


class SmolLM2CachedAttention(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        query_length: int,
        max_cache_length: int,
    ):
        super().__init__()
        if batch_size != 1:
            raise ValueError("The first cached engine is specialized to batch=1")
        self.batch_size = int(batch_size)
        self.query_length = int(query_length)
        self.max_cache_length = int(max_cache_length)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = PositionedRotaryEmbedding(
            max_cache_length, self.head_dim, config.rope_theta
        )
        self.cache_update = KVCacheUpdate()
        self.repeat_kv = RepeatKV(config.num_key_value_groups)
        self.softmax = CausalSoftmax()

    def _query_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            self.batch_size,
            self.query_length,
            self.num_heads,
            self.head_dim,
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(
            self.batch_size * self.num_heads,
            self.query_length,
            self.head_dim,
        )

    def _key_value_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            self.batch_size,
            self.query_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(
            self.batch_size * self.num_key_value_heads,
            self.query_length,
            self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.rotary(
            self._query_heads(self.q_proj(hidden_states)), cache_position
        )
        key = self.rotary(
            self._key_value_heads(self.k_proj(hidden_states)), cache_position
        )
        value = self._key_value_heads(self.v_proj(hidden_states))

        key_cache = self.cache_update(key, key_cache, cache_position)
        value_cache = self.cache_update(value, value_cache, cache_position)
        expanded_key = self.repeat_kv(key_cache)
        expanded_value = self.repeat_kv(value_cache)

        scores = torch.matmul(query, expanded_key.transpose(1, 2)) * self.scaling
        probabilities = self.softmax(scores, cache_position)
        attention = torch.matmul(probabilities, expanded_value)
        attention = attention.reshape(
            self.batch_size,
            self.num_heads,
            self.query_length,
            self.head_dim,
        )
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(
            self.batch_size, self.query_length, self.hidden_size
        )
        return self.o_proj(attention), key_cache, value_cache


class SmolLM2CachedDecoderLayer(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        query_length: int,
        max_cache_length: int,
    ):
        super().__init__()
        self.self_attn = SmolLM2CachedAttention(
            config,
            batch_size=batch_size,
            query_length=query_length,
            max_cache_length=max_cache_length,
        )
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention, key_cache, value_cache = self.self_attn(
            hidden_states, key_cache, value_cache, cache_position
        )
        hidden_states = residual + attention

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, key_cache, value_cache


class SmolLM2CachedBackbone(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        query_length: int,
        max_cache_length: int,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                SmolLM2CachedDecoderLayer(
                    config,
                    batch_size=batch_size,
                    query_length=query_length,
                    max_cache_length=max_cache_length,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        cache_position: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        updated_keys = []
        updated_values = []
        for index, layer in enumerate(self.layers):
            hidden_states, key_cache, value_cache = layer(
                hidden_states,
                key_caches[index],
                value_caches[index],
                cache_position,
            )
            updated_keys.append(key_cache)
            updated_values.append(value_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(updated_keys), tuple(updated_values)


class SmolLM2CachedForCausalLM(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int = 1,
        query_length: int = 1,
        max_cache_length: int = 64,
    ):
        super().__init__()
        self.config = config
        self.model = SmolLM2CachedBackbone(
            config,
            batch_size=batch_size,
            query_length=query_length,
            max_cache_length=max_cache_length,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        cache_position: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        hidden_states, key_caches, value_caches = self.model(
            input_ids, key_caches, value_caches, cache_position
        )
        return self.lm_head(hidden_states), key_caches, value_caches


def empty_caches(
    config: SmolLM2Config,
    max_cache_length: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    shape = (config.num_key_value_heads, max_cache_length, config.head_dim)
    keys = tuple(torch.zeros(shape, dtype=torch.float32) for _ in range(config.num_hidden_layers))
    values = tuple(
        torch.zeros(shape, dtype=torch.float32) for _ in range(config.num_hidden_layers)
    )
    return keys, values


def compile_cached_smollm2(
    model: SmolLM2CachedForCausalLM,
    example_inputs,
    *,
    target: str = "llvm",
    project: str = "smollm2_cache.prj",
):
    return allo.frontend.from_pytorch(
        model.eval(),
        example_inputs=example_inputs,
        leaf_modules=cached_leaf_modules(),
        quant_config=QuantizationConfig(),
        qdq_lowering_mode="fused",
        weights_as_args=True,
        target=target,
        project=project,
    )


def split_cached_outputs(
    outputs,
    num_layers: int,
):
    if not isinstance(outputs, (tuple, list)):
        raise TypeError("Cached SmolLM2 must return logits and two cache tuples")
    logits = outputs[0]
    keys = tuple(outputs[1 : 1 + num_layers])
    values = tuple(outputs[1 + num_layers : 1 + 2 * num_layers])
    return logits, keys, values


def generate(
    model_path: str | os.PathLike[str],
    prompt: str,
    *,
    max_new_tokens: int = 4,
    max_cache_length: int = 64,
    target: str = "llvm",
) -> str:
    """Compile prefill/decode modules and run greedy cached generation."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(torch.int32)
    prompt_length = int(token_ids.shape[1])
    if prompt_length + max_new_tokens > max_cache_length:
        raise ValueError("Prompt and generated tokens exceed max_cache_length")

    config = load_huggingface_config(model_path)
    prefill = SmolLM2CachedForCausalLM(
        config,
        query_length=prompt_length,
        max_cache_length=max_cache_length,
    ).eval()
    decode = SmolLM2CachedForCausalLM(
        config,
        query_length=1,
        max_cache_length=max_cache_length,
    ).eval()
    reference = load_huggingface_weights(prefill, model_path)
    decode.load_state_dict(prefill.state_dict(), strict=False)
    del reference

    key_caches, value_caches = empty_caches(config, max_cache_length)
    # Calibrate cache tensors with representative checkpoint activations. The
    # runtime prefill still receives empty caches, but using all-zero cache
    # examples here would force a scale of 1.0 and discard most K/V precision.
    with torch.no_grad():
        eager_logits, calibration_keys, calibration_values = prefill(
            token_ids, key_caches, value_caches, 0
        )
    decode_example = torch.tensor(
        [[int(torch.argmax(eager_logits[0, -1]))]], dtype=torch.int32
    )
    prefill_module = compile_cached_smollm2(
        prefill,
        (token_ids, calibration_keys, calibration_values, 0),
        target=target,
        project="smollm2_prefill.prj",
    )
    decode_module = compile_cached_smollm2(
        decode,
        (decode_example, calibration_keys, calibration_values, prompt_length),
        target=target,
        project="smollm2_decode.prj",
    )
    if target == "mlir":
        return str(prefill_module.module) + "\n" + str(decode_module.module)

    outputs = prefill_module(
        token_ids.numpy(),
        tuple(cache.numpy() for cache in key_caches),
        tuple(cache.numpy() for cache in value_caches),
        0,
    )
    logits, key_caches_np, value_caches_np = split_cached_outputs(
        outputs, config.num_hidden_layers
    )
    generated = token_ids.reshape(-1).tolist()
    next_token = int(np.argmax(logits[0, -1]))

    for step in range(max_new_tokens):
        generated.append(next_token)
        if step + 1 == max_new_tokens:
            break
        position = prompt_length + step
        outputs = decode_module(
            np.asarray([[next_token]], dtype=np.int32),
            key_caches_np,
            value_caches_np,
            position,
        )
        logits, key_caches_np, value_caches_np = split_cached_outputs(
            outputs, config.num_hidden_layers
        )
        next_token = int(np.argmax(logits[0, -1]))
    return tokenizer.decode(generated)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.getenv("ALLO_SMOLLM2_MODEL_DIR"),
    )
    parser.add_argument("--prompt", default="The future of hardware is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-cache-length", type=int, default=64)
    parser.add_argument("--target", choices=("mlir", "llvm"), default="mlir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model_dir is None:
        raise SystemExit("Pass --model-dir or set ALLO_SMOLLM2_MODEL_DIR")
    print(
        generate(
            args.model_dir,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_cache_length=args.max_cache_length,
            target=args.target,
        )
    )


if __name__ == "__main__":
    main()
