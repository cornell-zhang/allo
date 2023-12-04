# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .systolic import systolic, schedule_systolic

KERNEL2SCHEDULE = {
    systolic: schedule_systolic,
}
