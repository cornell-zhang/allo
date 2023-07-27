# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def np_type_to_str(dtype):
    if dtype == np.float32:
        return "f32"
    elif dtype == np.float64:
        return "f64"
    elif dtype == np.int32:
        return "i32"
    elif dtype == np.int64:
        return "i64"
    else:
        raise RuntimeError("Unsupported dtype")
