# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def np_type_to_str(dtype):
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    if dtype == np.int32:
        return "i32"
    if dtype == np.int64:
        return "i64"
    raise RuntimeError("Unsupported dtype")
