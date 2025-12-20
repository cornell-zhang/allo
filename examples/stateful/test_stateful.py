# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, float32, stateful


# Scalar stateless
def test_stateless_scalar(x: int32) -> int32:
    acc: int32 = 0
    acc = acc + x
    return acc


# Scalar stateful
def test_stateful_scalar(x: int32) -> int32:
    acc: stateful(int32) = 0
    acc = acc + x
    return acc


# Array stateless
def test_stateless_array(x: float32) -> float32:
    buffer: float32[10] = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]


# Array stateful
def test_stateful_array(x: float32) -> float32:
    buffer: stateful(float32[10]) = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]


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


s1 = allo.customize(test_stateless_scalar)
print(s1.module)  # Should show alloc

s2 = allo.customize(test_stateful_scalar)
print(s2.module)  # Should show memref.global

s3 = allo.customize(test_stateless_array)
print(s3.module)  # Should show alloc with shape

s4 = allo.customize(test_stateful_array)
print(s4.module)  # Should show memref.global with shape

s5 = allo.customize(test_moving_average)
print(s5.module)
