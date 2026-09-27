"""End-to-end fixed-length SmolLM2 tests without a KV cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

from allo.frontend.pytorch import (
    QuantizationConfig,
    TorchBuilder,
)
from allo.frontend.tracer import AlloTracer
from examples.torch.smollm2 import (
    SmolLM2Config,
    SmolLM2DecoderLayerHarness,
    SmolLM2ForCausalLM,
    compile_fixed_smollm2,
    load_huggingface_config,
    load_huggingface_weights,
    smollm2_leaf_modules,
)


def tiny_complete_config() -> SmolLM2Config:
    return SmolLM2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=30,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )


def build_full_model_source(model: torch.nn.Module, input_ids: torch.Tensor):
    leaves = smollm2_leaf_modules()
    tracer = AlloTracer(model, concrete_args={}, leaf_modules=leaves)
    graph = tracer.trace()
    gm = GraphModule(tracer.root, graph, model.__class__.__name__)
    ShapeProp(gm).propagate(input_ids)
    builder = TorchBuilder(
        gm,
        (input_ids,),
        leaf_modules=leaves,
        weights_as_args=True,
        quant_config=QuantizationConfig(),
        qdq_lowering_mode="fused",
    )
    return gm, builder, builder.build()


def checkpoint_directory() -> Path:
    value = os.getenv("ALLO_SMOLLM2_MODEL_DIR")
    required = os.getenv("ALLO_REQUIRE_SMOLLM2") == "1"
    if value is None or not Path(value).is_dir():
        message = "Set ALLO_SMOLLM2_MODEL_DIR to a local SmolLM2-135M checkpoint"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    return Path(value)


def require_heavy_gate(name: str):
    if os.getenv(name) != "1":
        pytest.skip(f"Set {name}=1 to run this expensive full-model gate")


def test_complete_synthetic_model_has_all_30_layers_and_tied_head():
    config = tiny_complete_config()
    model = SmolLM2ForCausalLM(config, sequence_length=3).eval()
    assert len(model.model.layers) == 30
    assert model.lm_head.weight is model.model.embed_tokens.weight

    input_ids = torch.tensor([[1, 7, 11]], dtype=torch.int32)
    with torch.no_grad():
        logits = model(input_ids)
    assert tuple(logits.shape) == (1, 3, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_complete_synthetic_model_fx_graph_contains_every_layer():
    config = tiny_complete_config()
    model = SmolLM2ForCausalLM(config, sequence_length=3).eval()
    input_ids = torch.tensor([[1, 7, 11]], dtype=torch.int32)
    graph, _, _ = build_full_model_source(model, input_ids)

    module_targets = {
        str(node.target) for node in graph.graph.nodes if node.op == "call_module"
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        assert f"{prefix}.self_attn.q_proj" in module_targets
        assert f"{prefix}.mlp.down_proj" in module_targets
        assert f"{prefix}.input_layernorm" in module_targets
    output_meta = next(node for node in graph.graph.nodes if node.op == "output").meta[
        "tensor_meta"
    ]
    assert tuple(output_meta.shape) == (1, 3, config.vocab_size)


def test_complete_synthetic_model_emits_all_integer_regions_and_runtime_weights():
    config = tiny_complete_config()
    model = SmolLM2ForCausalLM(config, sequence_length=3).eval()
    input_ids = torch.tensor([[1, 7, 11]], dtype=torch.int32)
    _, builder, source = build_full_model_source(model, input_ids)

    assert source.count("nn.linear3d[") == config.num_hidden_layers * 7 + 1
    assert source.count("nn.qrms_norm3d[") == config.num_hidden_layers * 2 + 1
    assert source.count("nn.qrope3d[") == config.num_hidden_layers * 2
    assert source.count("nn.qmatmul3d[") == config.num_hidden_layers * 2
    assert source.count("nn.qcausal_softmax3d[") == config.num_hidden_layers
    assert source.count("nn.qsilu3d[") == config.num_hidden_layers
    assert source.count("nn.qmul3d[") == config.num_hidden_layers
    assert source.count("nn.qadd3d[") == config.num_hidden_layers * 2
    assert "nn.qembedding2d[" in source

    assert "model_embed_tokens_weight: int8" in source
    assert "lm_head_weight: int8" not in source
    assert "lm_head_weight" not in builder.runtime_param_data
    assert "model_layers_0_self_attn_q_proj_zero_bias: int32" in source
    assert "= g_model_layers_0_self_attn_q_proj_weight" not in source
    tied_codes = builder.runtime_param_data["model_embed_tokens_weight"]
    assert tied_codes.shape == (config.vocab_size, config.hidden_size)
    assert source.count("model_embed_tokens_weight: int8") == 1


# BEGIN NATIVE INT8 QUANTIZATION: ADDED SmoothQuant semantic validation
def test_smoothquant_preserves_fp32_and_uses_shared_projection_scales():
    torch.manual_seed(17)
    config = tiny_complete_config()
    model = SmolLM2DecoderLayerHarness(config, sequence_length=3).eval()
    hidden_states = torch.randn(1, 3, config.hidden_size)

    norm_before = model.layer.input_layernorm.weight.detach().clone()
    post_norm_before = model.layer.post_attention_layernorm.weight.detach().clone()
    q_before = model.layer.self_attn.q_proj.weight.detach().clone()
    k_before = model.layer.self_attn.k_proj.weight.detach().clone()
    v_before = model.layer.self_attn.v_proj.weight.detach().clone()
    gate_before = model.layer.mlp.gate_proj.weight.detach().clone()
    up_before = model.layer.mlp.up_proj.weight.detach().clone()
    with torch.no_grad():
        expected = model(hidden_states)

    # NATIVE INT8 QUANTIZATION: CHANGED exercise the public Allo frontend path;
    # no decoder-layer names or Q/K/V projection names are supplied here.
    schedule = compile_fixed_smollm2(
        model,
        (hidden_states,),
        target="mlir",
        weights_as_args=True,
        smoothquant_alpha=0.5,
        smoothquant_calibration_inputs=((hidden_states,),),
    )
    scales = model._allo_smoothquant_scales
    attention_scale = scales["layer.input_layernorm"]
    mlp_scale = scales["layer.post_attention_layernorm"]
    with torch.no_grad():
        actual = model(hidden_states)

    input_norm = model.layer.input_layernorm
    post_norm = model.layer.post_attention_layernorm
    q_proj = model.layer.self_attn.q_proj
    k_proj = model.layer.self_attn.k_proj
    v_proj = model.layer.self_attn.v_proj
    gate_proj = model.layer.mlp.gate_proj
    up_proj = model.layer.mlp.up_proj

    torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-4)
    assert "requantize_per_channel3d" in str(schedule.module)
    torch.testing.assert_close(
        input_norm.weight,
        norm_before / attention_scale,
    )
    for actual_weight, original_weight in (
        (q_proj.weight, q_before),
        (k_proj.weight, k_before),
        (v_proj.weight, v_before),
    ):
        torch.testing.assert_close(
            actual_weight,
            original_weight * attention_scale.reshape(1, -1),
        )
    torch.testing.assert_close(
        post_norm.weight,
        post_norm_before / mlp_scale,
    )
    for actual_weight, original_weight in (
        (gate_proj.weight, gate_before),
        (up_proj.weight, up_before),
    ):
        torch.testing.assert_close(
            actual_weight,
            original_weight * mlp_scale.reshape(1, -1),
        )


# END NATIVE INT8 QUANTIZATION: ADDED SmoothQuant semantic validation


def test_real_checkpoint_configuration_is_exact_smollm2_135m():
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    assert config.hidden_size == 576
    assert config.intermediate_size == 1536
    assert config.num_hidden_layers == 30
    assert config.num_attention_heads == 9
    assert config.num_key_value_heads == 3
    assert config.head_dim == 64
    assert config.vocab_size == 49152
    assert config.tie_word_embeddings


def test_real_checkpoint_one_decoder_layer_matches_huggingface():
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2DecoderLayerHarness(
        config,
        sequence_length=8,
    ).eval()
    reference = load_huggingface_weights(
        model,
        model_dir,
        layer_index=0,
    )
    input_ids = torch.arange(
        8,
        dtype=torch.int64,
    ).reshape(1, 8)
    captured = {}

    def capture_input(_module, args):
        captured["input"] = args[0].detach()

    def capture_output(_module, _args, output):
        captured["output"] = (
            output[0]
            if isinstance(output, tuple)
            else output
        ).detach()

    input_handle = reference.model.layers[
        0
    ].register_forward_pre_hook(capture_input)
    output_handle = reference.model.layers[
        0
    ].register_forward_hook(capture_output)

    with torch.no_grad():
        reference(input_ids, use_cache=False)

    input_handle.remove()
    output_handle.remove()

    with torch.no_grad():
        actual = model(captured["input"])

    torch.testing.assert_close(
        actual.float(),
        captured["output"].float(),
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_real_checkpoint_decoder_layer_lowers_to_native_allo_mlir():
    require_heavy_gate("ALLO_RUN_SMOLLM2_LAYER_MLIR")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2DecoderLayerHarness(config, sequence_length=8).eval()
    reference = load_huggingface_weights(model, model_dir, layer_index=0)
    del reference
    hidden_states = torch.randn(1, 8, config.hidden_size)

    schedule = compile_fixed_smollm2(
        model,
        (hidden_states,),
        target="mlir",
        weights_as_args=True,
    )
    mlir = str(schedule.module)
    assert "func.func @forward" in mlir
    assert "memref<1x8x576xf32>" in mlir
    assert mlir.count("qrms_norm3d") >= 2
    assert mlir.count("qmatmul3d") >= 2
    assert mlir.count("qrope3d") >= 2


def test_real_checkpoint_decoder_layer_executes_in_llvm():
    require_heavy_gate("ALLO_RUN_SMOLLM2_LAYER_LLVM")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2DecoderLayerHarness(config, sequence_length=8).eval()
    reference = load_huggingface_weights(model, model_dir, layer_index=0)
    del reference
    torch.manual_seed(31)
    hidden_states = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        expected = model(hidden_states).numpy()

    module = compile_fixed_smollm2(
        model,
        (hidden_states,),
        target="llvm",
        weights_as_args=True,
    )
    assert "layer_self_attn_q_proj_weight" in module.weight_names
    actual = np.asarray(
        module(hidden_states.numpy(), *module.weight_data), dtype=np.float32
    )
    relative_l2 = np.linalg.norm(actual - expected) / max(
        np.linalg.norm(expected), 1.0e-8
    )
    assert relative_l2 < 0.35


# BEGIN NATIVE INT8 QUANTIZATION: ADDED real-activation SmoothQuant LLVM gate
def test_real_checkpoint_decoder_layer_smoothquant_executes_in_llvm():
    require_heavy_gate("ALLO_RUN_SMOLLM2_SMOOTHQUANT_LLVM")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED cross-layer validation
    layer_index = int(
        os.getenv("ALLO_SMOLLM2_LAYER_INDEX", "0")
    )
    if not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError(
            f"Invalid decoder layer index: {layer_index}"
        )

    model = SmolLM2DecoderLayerHarness(
        config,
        sequence_length=8,
        layer_index=layer_index,
    ).eval()
    reference = load_huggingface_weights(
        model,
        model_dir,
        layer_index=layer_index,
    )
    # END NATIVE INT8 QUANTIZATION: ADDED cross-layer validation

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED representative multi-sample calibration
    # Collect real layer inputs from natural-text prompts. The last prompt is
    # held out so calibration is not evaluated on its own input samples.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    calibration_prompts = (
        "Machine learning models predict the next token in a sequence.",
        "Quantization reduces memory and computational requirements.",
        "Processors execute arithmetic instructions at very high speeds.",
        "Scientific experiments require careful repeatable measurements.",
        "Matrix multiplication is important for neural network inference.",
        "Hardware accelerators improve numerical computing performance.",
        "The weather became colder as evening approached the city.",
        "Small language models can generate coherent pieces of text.",
    )
    evaluation_prompt = (
        "Researchers designed an efficient system for processing information."
    )
    prompts = calibration_prompts + (evaluation_prompt,)

    input_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=8,
    )["input_ids"]

    if tuple(input_ids.shape) != (len(prompts), 8):
        raise RuntimeError(
            f"Unexpected token shape: {tuple(input_ids.shape)}"
        )

    captured = {}

    def capture_input(_module, args):
        captured["input"] = args[0].detach().float()

    handle = reference.model.layers[
        layer_index
    ].register_forward_pre_hook(capture_input)

    with torch.no_grad():
        reference(input_ids, use_cache=False)

    handle.remove()
    del reference

    all_hidden_states = captured["input"]
    calibration_inputs = tuple(
        (all_hidden_states[index : index + 1].contiguous(),)
        for index in range(len(calibration_prompts))
    )
    hidden_states = all_hidden_states[-1:].contiguous()
    # END NATIVE INT8 QUANTIZATION: ADDED representative multi-sample calibration

    with torch.no_grad():
        expected = model(hidden_states).numpy()

    # Build the unsmoothed path on the identical real activation.
    baseline_module = compile_fixed_smollm2(
        model,
        (hidden_states,),
        target="llvm",
        weights_as_args=True,
        # NATIVE INT8 QUANTIZATION: CHANGED use representative calibration.
        smoothquant_calibration_inputs=calibration_inputs,
    )
    baseline_actual = np.asarray(
        baseline_module(
            hidden_states.numpy(),
            *baseline_module.weight_data,
        ),
        dtype=np.float32,
    )
    baseline_relative_l2 = (
        np.linalg.norm(baseline_actual - expected)
        / max(np.linalg.norm(expected), 1.0e-8)
    )

    alpha = float(
        os.getenv(
            "ALLO_SMOLLM2_SMOOTHQUANT_ALPHA",
            "0.5",
        )
    )
    module = compile_fixed_smollm2(
        model,
        (hidden_states,),
        target="llvm",
        weights_as_args=True,
        smoothquant_alpha=alpha,
        # NATIVE INT8 QUANTIZATION: CHANGED use identical calibration data.
        smoothquant_calibration_inputs=calibration_inputs,
    )
    actual = np.asarray(
        module(
            hidden_states.numpy(),
            *module.weight_data,
        ),
        dtype=np.float32,
    )
    relative_l2 = (
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), 1.0e-8)
    )

    # BEGIN NATIVE INT8 QUANTIZATION: ADDED SmoothQuant efficacy diagnostics
    # Verify that generic SmoothQuant applied nontrivial scales and expose
    # output differences that six-decimal error printing can hide.
    smoothquant_scales = model._allo_smoothquant_scales
    assert smoothquant_scales
    smoothquant_scale_values = torch.cat(
        [
            scale.detach().float().reshape(-1)
            for scale in smoothquant_scales.values()
        ]
    )
    output_relative_delta = np.linalg.norm(
        actual - baseline_actual
    ) / max(np.linalg.norm(baseline_actual), 1.0e-8)
    improvement = baseline_relative_l2 - relative_l2
    # END NATIVE INT8 QUANTIZATION: ADDED SmoothQuant efficacy diagnostics

    print(
        f"layer={layer_index} "
        f"baseline relative L2={baseline_relative_l2:.9f}; "
        f"SmoothQuant alpha={alpha:.2f} "
        f"relative L2={relative_l2:.9f}; "
        f"improvement={improvement:+.9f}; "
        f"output delta={output_relative_delta:.9f}; "
        f"scale range=[{float(smoothquant_scale_values.min()):.6g}, "
        f"{float(smoothquant_scale_values.max()):.6g}]"
    )
    assert relative_l2 <= baseline_relative_l2
    assert relative_l2 < 0.35


# END NATIVE INT8 QUANTIZATION: ADDED real-activation SmoothQuant LLVM gate


def test_real_checkpoint_complete_pytorch_model_matches_huggingface_logits():
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2ForCausalLM(config, sequence_length=8).eval()
    reference = load_huggingface_weights(model, model_dir)
    input_ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)

    with torch.no_grad():
        expected = reference(input_ids.to(torch.int64), use_cache=False).logits.float()
        actual = model(input_ids).float()
    torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-4)


def test_real_checkpoint_complete_model_lowers_to_native_allo_mlir():
    require_heavy_gate("ALLO_RUN_SMOLLM2_MLIR")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2ForCausalLM(config, sequence_length=8).eval()
    reference = load_huggingface_weights(model, model_dir)
    del reference
    input_ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)

    schedule = compile_fixed_smollm2(
        model,
        (input_ids,),
        target="mlir",
        weights_as_args=True,
    )
    mlir = str(schedule.module)
    assert "func.func @forward" in mlir
    assert "memref<1x8xi32>" in mlir
    assert "memref<1x8x49152xf32>" in mlir
    assert mlir.count("qrms_norm3d") >= 61
    assert mlir.count("qmatmul3d") >= 60
    assert "qembedding2d" in mlir


def test_real_checkpoint_fixed_length_full_logits_execute_in_llvm():
    require_heavy_gate("ALLO_RUN_SMOLLM2_LLVM")
    model_dir = checkpoint_directory()
    config = load_huggingface_config(model_dir)
    model = SmolLM2ForCausalLM(config, sequence_length=8).eval()
    reference = load_huggingface_weights(model, model_dir)
    input_ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    with torch.no_grad():
        expected = reference(input_ids.to(torch.int64), use_cache=False).logits.float()
    del reference

    module = compile_fixed_smollm2(
        model,
        (input_ids,),
        target="llvm",
        weights_as_args=True,
    )
    actual = np.asarray(module(input_ids.numpy()), dtype=np.float32)
    assert actual.shape == (1, 8, config.vocab_size)
    assert np.isfinite(actual).all()

    expected_np = expected.cpu().numpy()
    error = np.linalg.norm(actual - expected_np)
    baseline = max(np.linalg.norm(expected_np), 1.0e-8)
    assert error / baseline < 0.35
    assert int(actual[0, -1].argmax()) == int(expected_np[0, -1].argmax())
