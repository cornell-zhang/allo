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
)

KERNEL2SCHEDULE = {}

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
    }
)
