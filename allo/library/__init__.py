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

KERNEL2SCHEDULE = {}

KERNEL2SCHEDULE.update(
    {
        systolic: schedule_systolic,
        packed_systolic: schedule_systolic,
        packed_int8xint8_systolic: schedule_systolic,
    }
)

KERNEL2SCHEDULE[int8xint8_mat_vec] = schedule_int8xint8_mat_vec
