"""Match kernels with respective schedules."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .systolic import (
    systolic,
    packed_systolic,
    packed_int8xint8_systolic,
    schedule_systolic,
)

from .gemv import (
    int8xint8_mat_vec,
    schedule_int8xint8_mat_vec,
)

from .nn import (
    linear2d,
    linear3d,
    schedule_linear2d,
    schedule_linear3d,
    relu2d,
    relu4d,
    schedule_relu2d,
    schedule_relu4d,
    softmax,
    schedule_softmax,
    layer_norm,
    schedule_layernorm,
    GeLU,
    schedule_gelu,
    conv2d,
    schedule_conv2d,
    maxpool2d,
    schedule_maxpool2d,
    avgpool2d,
    schedule_avgpool2d,
    batchnorm2d,
    schedule_batchnorm2d,
    relu3d,
    schedule_relu3d,
    repeat_batch3d,
    schedule_repeat_batch3d,
    batchnorm1d_2d,
    schedule_batchnorm1d_2d,
    batchnorm1d_3d,
    schedule_batchnorm1d_3d,
    log_softmax,
    schedule_log_softmax,
    concat,
    schedule_concat,
    # BEGIN NATIVE INT8 QUANTIZATION: ADDED integer kernel exports
    roundeven,
    quantize2d,
    quantize3d,
    dequantize2d,
    dequantize3d,
    requantize2d,
    requantize3d,
    requantize_per_channel2d,
    requantize_per_channel3d,
    qadd2d,
    qadd3d,
    # BEGIN NATIVE INT8 QUANTIZATION: ADDED transformer kernel exports
    rms_norm3d,
    silu3d,
    rope3d,
    positioned_rope3d,
    repeat_interleave3d,
    causal_softmax3d,
    embedding2d,
    kv_cache_update3d,
    qmul3d,
    qmatmul3d,
    qsilu3d,
    qrope3d,
    qpositioned_rope3d,
    qcausal_softmax3d,
    qrms_norm3d,
    qembedding2d,
    qkv_cache_update3d,
    # END NATIVE INT8 QUANTIZATION: ADDED transformer kernel exports
    schedule_native_quantized,
    # END NATIVE INT8 QUANTIZATION: ADDED integer kernel exports
)

KERNEL2SCHEDULE = {}
# BEGIN NATIVE INT8 QUANTIZATION: ADDED integer schedule registration

KERNEL2SCHEDULE.update(
    {
        roundeven: schedule_native_quantized,
        quantize2d: schedule_native_quantized,
        quantize3d: schedule_native_quantized,
        dequantize2d: schedule_native_quantized,
        dequantize3d: schedule_native_quantized,
        requantize2d: schedule_native_quantized,
        requantize3d: schedule_native_quantized,
        requantize_per_channel2d: schedule_native_quantized,
        requantize_per_channel3d: schedule_native_quantized,
        qadd2d: schedule_native_quantized,
        qadd3d: schedule_native_quantized,
        # BEGIN NATIVE INT8 QUANTIZATION: ADDED transformer schedule registration
        rms_norm3d: schedule_native_quantized,
        silu3d: schedule_native_quantized,
        rope3d: schedule_native_quantized,
        positioned_rope3d: schedule_native_quantized,
        repeat_interleave3d: schedule_native_quantized,
        causal_softmax3d: schedule_native_quantized,
        embedding2d: schedule_native_quantized,
        kv_cache_update3d: schedule_native_quantized,
        qmul3d: schedule_native_quantized,
        qmatmul3d: schedule_native_quantized,
        qsilu3d: schedule_native_quantized,
        qrope3d: schedule_native_quantized,
        qpositioned_rope3d: schedule_native_quantized,
        qcausal_softmax3d: schedule_native_quantized,
        qrms_norm3d: schedule_native_quantized,
        qembedding2d: schedule_native_quantized,
        qkv_cache_update3d: schedule_native_quantized,
        # END NATIVE INT8 QUANTIZATION: ADDED transformer schedule registration
    }
)
# END NATIVE INT8 QUANTIZATION: ADDED integer schedule registration

KERNEL2SCHEDULE.update(
    {
        systolic: schedule_systolic,
        packed_systolic: schedule_systolic,
        packed_int8xint8_systolic: schedule_systolic,
    }
)

KERNEL2SCHEDULE[int8xint8_mat_vec] = schedule_int8xint8_mat_vec

KERNEL2SCHEDULE.update(
    {
        linear2d: schedule_linear2d,
        linear3d: schedule_linear3d,
        relu2d: schedule_relu2d,
        relu4d: schedule_relu4d,
        softmax: schedule_softmax,
        layer_norm: schedule_layernorm,
        GeLU: schedule_gelu,
        conv2d: schedule_conv2d,
        maxpool2d: schedule_maxpool2d,
        avgpool2d: schedule_avgpool2d,
        batchnorm2d: schedule_batchnorm2d,
        relu3d: schedule_relu3d,
        repeat_batch3d: schedule_repeat_batch3d,
        batchnorm1d_2d: schedule_batchnorm1d_2d,
        batchnorm1d_3d: schedule_batchnorm1d_3d,
        log_softmax: schedule_log_softmax,
        concat: schedule_concat,
    }
)
