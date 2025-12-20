# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import allo
from allo.ir.types import int1, int8, int32, float32, stateful


def parametric_kernel_rng[SEED: int32]() -> int32:
    seed: stateful(int32) = SEED
    seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
    return seed & 0xFFFF


def kernel_rng():
    return parametric_kernel_rng[0]()


s = allo.customize(kernel_rng)
print(s.module)
code = s.build(target="vhls")
print(code)
