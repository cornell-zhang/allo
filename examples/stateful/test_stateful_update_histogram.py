# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Int, int8, int32, float32, stateful


def update_histogram(number: int8) -> int32[256]:
    histogram: stateful(int32[256]) = 0

    histogram[number] += 1
    return histogram


s = allo.customize(update_histogram)
print(s.module)
code = s.build(target="vhls")
print(code)
