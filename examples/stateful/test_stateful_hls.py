# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Int, int32, float32, stateful


def test_stateful_scalar(x: Int(4)) -> Int(4):
    acc: stateful(Int(4)) = 0
    acc = acc + x
    return acc


def test_moving_average(new_value: float32) -> float32:
    window: stateful(float32[4]) = 0.0
    i: stateful(int32) = 0
    count: stateful(int32) = 0
    total: stateful(float32) = 0.0

    # Subtract the old value that's being replaced
    total = total - window[i]

    # Add new value to window
    window[i] = new_value
    total = total + new_value

    # Move to next position (circular buffer)
    i = (i + 1) % 4

    # Track how many values we've seen (max 4)
    if count < 4:
        count = count + 1

    # Return average of values in window
    return total / count


s = allo.customize(test_stateful_scalar)
print(s.module)

code = s.build(target="vhls")
print(code)
