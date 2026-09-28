# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete fixed-shape SmolLM2-135M PyTorch frontend example.

This follows ``examples/torch/gpt2.py``: the model architecture is expressed as
traceable PyTorch modules, while Allo's frontend only knows generic operators.
The Hugging Face model supplies configuration, checkpoint weights, and the
floating-point reference. No SmolLM2-specific lowering is added to TorchBuilder.

The fixed-shape path covers token embedding, all decoder layers, final RMSNorm,
the tied LM head, and full logits. KV-cache generation is kept in
``examples/torch/experimental/smollm2_KV_cache.py``.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import allo
from allo.frontend.pytorch import QuantizationConfig


@dataclass(frozen=True)
class SmolLM2Config:
    """Static architectural fields required by the traceable model."""

    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1.0e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_attention_heads < 1 or self.num_key_value_heads < 1:
            raise ValueError("SmolLM2 attention-head counts must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if (
            min(
                self.num_hidden_layers,
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.max_position_embeddings,
            )
            < 1
        ):
            raise ValueError("SmolLM2 dimensions must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_huggingface(cls, config) -> "SmolLM2Config":
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        else:
            rope_theta = config.rope_theta
        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            max_position_embeddings=int(config.max_position_embeddings),
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=float(rope_theta),
            tie_word_embeddings=bool(config.tie_word_embeddings),
        )


def build_rope_tables(
    sequence_length: int,
    head_dim: int,
    theta: float,
    *,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Llama-style full-width cosine and sine tables."""

    positions = torch.arange(
        position_offset, position_offset + sequence_length, dtype=torch.float32
    )
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (theta ** (dimensions / head_dim))
    frequencies = torch.outer(positions, inverse_frequency)
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos(), embeddings.sin()


class RMSNorm(nn.Module):
    """Trace leaf matching the Llama/SmolLM2 RMSNorm equation."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


class RotaryEmbedding(nn.Module):
    """Fixed-position RoPE leaf operating on ``[heads, tokens, head_dim]``."""

    def __init__(
        self,
        sequence_length: int,
        head_dim: int,
        theta: float,
        *,
        position_offset: int = 0,
    ):
        super().__init__()
        cos, sin = build_rope_tables(
            sequence_length,
            head_dim,
            theta,
            position_offset=position_offset,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cos.unsqueeze(0) + self.rotate_half(x) * self.sin.unsqueeze(0)


class RepeatKV(nn.Module):
    """Grouped-query head expansion with repeat-interleave ordering."""

    def __init__(self, repeat_factor: int):
        super().__init__()
        self.repeat_factor = int(repeat_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(x, self.repeat_factor, dim=0)


class CausalSoftmax(nn.Module):
    """Causal softmax leaf shared by prefill and cached decode."""

    def forward(self, scores: torch.Tensor, causal_offset: int = 0) -> torch.Tensor:
        query_length, key_length = scores.shape[-2:]
        query_positions = torch.arange(
            causal_offset,
            causal_offset + query_length,
            device=scores.device,
        )
        key_positions = torch.arange(key_length, device=scores.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        masked = scores.masked_fill(
            ~allowed.unsqueeze(0), torch.finfo(scores.dtype).min
        )
        return F.softmax(masked.float(), dim=-1).to(scores.dtype)


def smollm2_leaf_modules() -> tuple[type[nn.Module], ...]:
    """Custom PyTorch leaves with generic TorchBuilder implementations."""

    return (RMSNorm, RotaryEmbedding, RepeatKV, CausalSoftmax)


class SmolLM2Attention(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        sequence_length: int,
        position_offset: int = 0,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
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
        self.rotary = RotaryEmbedding(
            sequence_length,
            self.head_dim,
            config.rope_theta,
            position_offset=position_offset,
        )
        self.repeat_kv = RepeatKV(config.num_key_value_groups)
        self.softmax = CausalSoftmax()

    def _query_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            self.batch_size,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(
            self.batch_size * self.num_heads,
            self.sequence_length,
            self.head_dim,
        )

    def _key_value_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(
            self.batch_size,
            self.sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(
            self.batch_size * self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.rotary(self._query_heads(self.q_proj(hidden_states)))
        key = self.rotary(self._key_value_heads(self.k_proj(hidden_states)))
        value = self._key_value_heads(self.v_proj(hidden_states))

        key = self.repeat_kv(key)
        value = self.repeat_kv(value)
        scores = torch.matmul(query, key.transpose(1, 2)) * self.scaling
        probabilities = self.softmax(scores, 0)
        attention = torch.matmul(probabilities, value)

        attention = attention.reshape(
            self.batch_size,
            self.num_heads,
            self.sequence_length,
            self.head_dim,
        )
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(
            self.batch_size, self.sequence_length, self.hidden_size
        )
        return self.o_proj(attention)


class SmolLM2MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class SmolLM2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        sequence_length: int,
        position_offset: int = 0,
    ):
        super().__init__()
        self.self_attn = SmolLM2Attention(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            position_offset=position_offset,
        )
        self.mlp = SmolLM2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class SmolLM2Backbone(nn.Module):
    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int,
        sequence_length: int,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                SmolLM2DecoderLayer(
                    config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class SmolLM2ForCausalLM(nn.Module):
    """Full fixed-length model with tied token embedding and LM-head weights."""

    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int = 1,
        sequence_length: int = 8,
    ):
        super().__init__()
        self.config = config
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.model = SmolLM2Backbone(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids))


class SmolLM2DecoderLayerHarness(nn.Module):
    """One-layer entry point used by the progressive tests."""

    def __init__(
        self,
        config: SmolLM2Config,
        *,
        batch_size: int = 1,
        sequence_length: int = 8,
        layer_index: int = 0,
    ):
        super().__init__()
        self.layer_index = int(layer_index)
        if not 0 <= self.layer_index < config.num_hidden_layers:
            raise ValueError("layer_index is outside the SmolLM2 decoder stack")
        self.layer = SmolLM2DecoderLayer(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden_states)


def load_huggingface_config(model_path: str | os.PathLike[str]) -> SmolLM2Config:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return SmolLM2Config.from_huggingface(hf_config)


def load_huggingface_weights(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    layer_index: int | None = None,
) -> nn.Module:
    """Load the full checkpoint or one selected decoder layer."""

    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if isinstance(model, SmolLM2DecoderLayerHarness):
        index = model.layer_index if layer_index is None else int(layer_index)
        source = reference.model.layers[index].state_dict()
        missing, unexpected = model.layer.load_state_dict(source, strict=False)
    else:
        missing, unexpected = model.load_state_dict(
            reference.state_dict(), strict=False
        )
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors: {unexpected}")
    trainable_missing = [name for name in missing if not name.endswith(("cos", "sin"))]
    if trainable_missing:
        raise RuntimeError(f"Missing checkpoint tensors: {trainable_missing}")
    return reference


def compile_fixed_smollm2(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    target: str = "llvm",
    project: str = "smollm2.prj",
    weights_as_args: bool = True,
    calibration_inputs: tuple[tuple[torch.Tensor, ...], ...] | None = None,
):
    """Compile a layer or the complete no-cache model through TorchBuilder."""

    return allo.frontend.from_pytorch(
        model.eval(),
        example_inputs=example_inputs,
        leaf_modules=smollm2_leaf_modules(),
        quant_config=QuantizationConfig(
            weight_granularity="per_channel",
        ),
        qdq_lowering_mode="fused",
        calibration_inputs=(
            (example_inputs,) if calibration_inputs is None else calibration_inputs
        ),
        weights_as_args=weights_as_args,
        target=target,
        project=project,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.getenv("ALLO_SMOLLM2_MODEL_DIR"),
        help="Local HuggingFaceTB/SmolLM2-135M checkpoint directory",
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--target", choices=("mlir", "llvm", "vitis_hls"), default="mlir"
    )
    parser.add_argument("--project", default="smollm2.prj")
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument("--layer-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model_dir is None:
        raise SystemExit("Pass --model-dir or set ALLO_SMOLLM2_MODEL_DIR")
    config = load_huggingface_config(args.model_dir)
    if args.layer_only:
        model = SmolLM2DecoderLayerHarness(
            config,
            sequence_length=args.sequence_length,
            layer_index=args.layer_index,
        ).eval()
        reference = load_huggingface_weights(model, args.model_dir)
        del reference
        example = torch.randn(1, args.sequence_length, config.hidden_size)
    else:
        model = SmolLM2ForCausalLM(config, sequence_length=args.sequence_length).eval()
        reference = load_huggingface_weights(model, args.model_dir)
        del reference
        example = torch.arange(args.sequence_length, dtype=torch.int32).reshape(1, -1)

    compiled = compile_fixed_smollm2(
        model,
        (example,),
        target=args.target,
        project=args.project,
    )
    if args.target == "mlir":
        print(compiled.module)
        return
    actual = np.asarray(compiled(example.detach().cpu().numpy()), dtype=np.float32)
    width = config.hidden_size if args.layer_only else config.vocab_size
    expected_shape = (1, args.sequence_length, width)
    if actual.shape != expected_shape or not np.isfinite(actual).all():
        raise AssertionError(
            "SmolLM2 must return the expected shape and finite outputs"
        )
    print("SmolLM2 fixed-shape execution passed")


if __name__ == "__main__":
    main()
