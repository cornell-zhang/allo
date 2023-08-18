# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32


def test_int32_float32():
    def kernel(a: int32) -> float32:
        return float(int(float(a)))

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(1) == kernel(1)


if __name__ == "__main__":
    test_int32_float32()
