# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolLM2 KV-cache prefill, decode, and generation tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

from allo.frontend.pytorch import QuantizationConfig, TorchBuilder
from allo.frontend.tracer import AlloTracer
from examples.torch.experimental.smollm2_KV_cache import (
    SmolLM2CachedForCausalLM,
    cached_leaf_modules,
    compile_cached_smollm2,
    empty_caches,
    generate,
    split_cached_outputs,
)
from examples.torch.smollm2 import (
    SmolLM2Config,
    SmolLM2ForCausalLM,
    load_huggingface_config,
    load_huggingface_weights,
)


def tiny_config() -> SmolLM2Config:
    return SmolLM2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=30,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )


def checkpoint_directory() -> Path:
    value = os.getenv("ALLO_SMOLLM2_MODEL_DIR")
    required = os.getenv("ALLO_REQUIRE_SMOLLM2") == "1"
    if value is None or not Path(value).is_dir():
        message = "Set ALLO_SMOLLM2_MODEL_DIR to a local SmolLM2-135M checkpoint"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    return Path(value)


def require_gate(name: str):
    if os.getenv(name) != "1":
        pytest.skip(f"Set {name}=1 to run this expensive cached-model gate")


def copy_matching_weights(source: torch.nn.Module, destination: torch.nn.Module):
    missing, unexpected = destination.load_state_dict(source.state_dict(), strict=False)
    trainable_missing = [name for name in missing if not name.endswith(("cos", "sin"))]
    assert not trainable_missing
    assert not unexpected


def build_cached_source(model, example_inputs):
    leaves = cached_leaf_modules()
    tracer = AlloTracer(model, concrete_args={}, leaf_modules=leaves)
    graph = tracer.trace()
    gm = GraphModule(tracer.root, graph, model.__class__.__name__)
    ShapeProp(gm).propagate(*example_inputs)
    builder = TorchBuilder(
        gm,
        example_inputs,
        leaf_modules=leaves,
        weights_as_args=True,
        quant_config=QuantizationConfig(),
        qdq_lowering_mode="fused",
    )
    return gm, builder, builder.build()


def test_cached_prefill_matches_no_cache_for_complete_30_layer_model():
    torch.manual_seed(12)
    config = tiny_config()
    prompt = torch.tensor([[2, 5, 9]], dtype=torch.int32)
    fixed = SmolLM2ForCausalLM(config, sequence_length=3).eval()
    cached = SmolLM2CachedForCausalLM(config, query_length=3, max_cache_length=8).eval()
    copy_matching_weights(fixed, cached)
    keys, values = empty_caches(config, 8)

    with torch.no_grad():
        expected = fixed(prompt)
        actual, updated_keys, updated_values = cached(prompt, keys, values, 0)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    assert len(updated_keys) == config.num_hidden_layers
    assert len(updated_values) == config.num_hidden_layers
    assert tuple(updated_keys[0].shape) == (1, 8, config.head_dim)
    assert torch.count_nonzero(updated_keys[0][:, :3]).item() > 0
    assert torch.count_nonzero(updated_keys[0][:, 3:]).item() == 0


def test_single_token_cached_decode_matches_full_sequence_logits():
    torch.manual_seed(21)
    config = tiny_config()
    prompt = torch.tensor([[2, 5, 9]], dtype=torch.int32)
    next_token = torch.tensor([[4]], dtype=torch.int32)
    complete = torch.cat((prompt, next_token), dim=1)

    full_model = SmolLM2ForCausalLM(config, sequence_length=4).eval()
    prefill = SmolLM2CachedForCausalLM(
        config, query_length=3, max_cache_length=8
    ).eval()
    decode = SmolLM2CachedForCausalLM(config, query_length=1, max_cache_length=8).eval()
    copy_matching_weights(full_model, prefill)
    copy_matching_weights(full_model, decode)
    keys, values = empty_caches(config, 8)

    with torch.no_grad():
        expected = full_model(complete)[:, -1]
        _, keys, values = prefill(prompt, keys, values, 0)
        actual, keys, values = decode(next_token, keys, values, 3)
    torch.testing.assert_close(actual[:, -1], expected, rtol=3.0e-5, atol=3.0e-5)
    assert torch.count_nonzero(keys[0][:, :4]).item() > 0
    assert torch.count_nonzero(values[0][:, 4:]).item() == 0


def test_complete_cached_model_emits_positioned_rope_and_60_cache_updates():
    torch.manual_seed(7)
    config = tiny_config()
    model = SmolLM2CachedForCausalLM(config, query_length=1, max_cache_length=8).eval()
    input_ids = torch.tensor([[3]], dtype=torch.int32)
    keys, values = empty_caches(config, 8)
    _, builder, source = build_cached_source(model, (input_ids, keys, values, 3))

    assert source.count("nn.qpositioned_rope3d[") == config.num_hidden_layers * 2
    assert source.count("nn.qkv_cache_update3d[") == config.num_hidden_layers * 2
    assert source.count("nn.qcausal_softmax3d[") == config.num_hidden_layers
    assert source.count("nn.qmatmul3d[") == config.num_hidden_layers * 2
    assert source.count("_dequantized = nn.dequantize3d[") >= (
        config.num_hidden_layers * 2 + 1
    )
    assert len(builder.input_args) == 2 * config.num_hidden_layers + 2


def test_real_checkpoint_cached_prefill_and_decode_match_huggingface():
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    prompt = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    next_token = torch.tensor([[11]], dtype=torch.int32)
    complete = torch.cat((prompt, next_token), dim=1)

    prefill = SmolLM2CachedForCausalLM(
        config, query_length=8, max_cache_length=16
    ).eval()
    decode = SmolLM2CachedForCausalLM(
        config, query_length=1, max_cache_length=16
    ).eval()
    reference = load_huggingface_weights(prefill, model_dir)
    copy_matching_weights(prefill, decode)
    keys, values = empty_caches(config, 16)

    with torch.no_grad():
        expected_prefill = reference(
            prompt.to(torch.int64), use_cache=False
        ).logits.float()
        expected_decode = (
            reference(complete.to(torch.int64), use_cache=False).logits[:, -1].float()
        )
        actual_prefill, keys, values = prefill(prompt, keys, values, 0)
        actual_decode, _, _ = decode(next_token, keys, values, 8)
    torch.testing.assert_close(
        actual_prefill.float(), expected_prefill, rtol=2.0e-4, atol=2.0e-4
    )
    torch.testing.assert_close(
        actual_decode[:, -1].float(),
        expected_decode,
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_real_checkpoint_cached_model_lowers_to_native_allo_mlir():
    require_gate("ALLO_RUN_SMOLLM2_CACHE_MLIR")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2CachedForCausalLM(config, query_length=1, max_cache_length=16).eval()
    reference = load_huggingface_weights(model, model_dir)
    del reference
    ids = torch.tensor([[1]], dtype=torch.int32)
    keys, values = empty_caches(config, 16)

    schedule = compile_cached_smollm2(
        model,
        (ids, keys, values, 8),
        target="mlir",
    )
    mlir = str(schedule.module)
    assert "func.func @forward" in mlir
    assert mlir.count("qkv_cache_update3d") >= 60
    assert mlir.count("qpositioned_rope3d") >= 60
    assert "memref<1x1x49152xf32>" in mlir


def test_real_checkpoint_cached_decode_executes_in_llvm():
    require_gate("ALLO_RUN_SMOLLM2_CACHE_LLVM")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2CachedForCausalLM(config, query_length=1, max_cache_length=16).eval()
    reference = load_huggingface_weights(model, model_dir)
    del reference
    ids = torch.tensor([[1]], dtype=torch.int32)
    keys, values = empty_caches(config, 16)
    module = compile_cached_smollm2(
        model,
        (ids, keys, values, 0),
        target="llvm",
    )
    outputs = module(
        ids.numpy(),
        tuple(cache.numpy() for cache in keys),
        tuple(cache.numpy() for cache in values),
        0,
    )
    logits, updated_keys, updated_values = split_cached_outputs(
        outputs, config.num_hidden_layers
    )
    assert np.asarray(logits).shape == (1, 1, config.vocab_size)
    assert len(updated_keys) == 30
    assert len(updated_values) == 30
    assert np.isfinite(logits).all()


def test_real_checkpoint_token_by_token_generation():
    require_gate("ALLO_RUN_SMOLLM2_GENERATION")
    model_dir = checkpoint_directory()
    text = generate(
        model_dir,
        "The future of hardware is",
        max_new_tokens=2,
        max_cache_length=32,
        target="llvm",
    )
    assert isinstance(text, str)
    assert len(text) > 0
