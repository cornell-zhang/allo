# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .systolic import (
    systolic,
    packed_systolic,
    packed_int8xint8_systolic,
    schedule_systolic,
)

KERNEL2SCHEDULE = {
    systolic: schedule_systolic,
    packed_systolic: schedule_systolic,
    packed_int8xint8_systolic: schedule_systolic,
}
