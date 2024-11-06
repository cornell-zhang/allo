# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M, N, K = 16, 16, 16


@df.region()
def top():
    pipe = df.pipe(dtype=Ty, shape=(), depth=2)

if __name__ == "__main__":
    s = allo.customize(top, verbose=True)
    print(s.module)