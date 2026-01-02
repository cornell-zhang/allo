# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import allo
from allo.ir.types import Int, int32, float32, stateful


def test_stateful_scalar_hls():
    """Test HLS code generation for stateful scalar"""

    def test_stateful_scalar(x: Int(4)) -> Int(4):
        acc: Int(4) @ stateful = 0
        acc = acc + x
        return acc

    s = allo.customize(test_stateful_scalar)
    mod = s.build(target="vhls")
    code = mod.hls_code

    # Static qualifier must be present for stateful variables
    static_count = len(re.findall(r"\bstatic\b", code))
    assert (
        static_count >= 1
    ), f"Expected at least one static variable for stateful state, found {static_count}"

    # Static variable should be of correct type (ap_int<4>)
    assert re.search(
        r"static\s+ap_int<4>", code
    ), "Expected static variable with ap_int<4> type"

    # Static variable should be initialized to 0
    static_init_pattern = r"static\s+ap_int<4>\s+\w+\s*=\s*\{?0\}?"
    assert re.search(
        static_init_pattern, code
    ), "Expected static variable to be initialized to 0"

    print("test_stateful_scalar_hls passed!")


def test_moving_average_hls():
    """Test HLS code generation for moving average with multiple stateful variables"""

    def test_moving_average(new_value: float32) -> float32:
        window: float32[4] @ stateful = 0.0
        i: int32 @ stateful = 0
        count: int32 @ stateful = 0
        total: float32 @ stateful = 0.0

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

    s = allo.customize(test_moving_average)
    mod = s.build(target="vhls")
    code = mod.hls_code

    # Should have multiple static variables (4 total: window, i, count, total)
    static_count = len(re.findall(r"\bstatic\b", code))
    assert (
        static_count >= 4
    ), f"Expected at least 4 static variables for stateful states, found {static_count}"

    # Should have static array (for window)
    assert re.search(
        r"static\s+float\s+\w+\s*\[\s*4\s*\]", code
    ), "Expected static float array of size 4 for window"

    # Should have static int32_t variables (for i and count)
    int_static_count = len(re.findall(r"static\s+int32_t", code))
    assert (
        int_static_count >= 2
    ), f"Expected at least 2 static int32_t variables, found {int_static_count}"

    # Should have static float scalar (for total)
    float_static_count = len(re.findall(r"static\s+float\s+\w+\s*=", code))
    assert (
        float_static_count >= 1
    ), f"Expected at least 1 static float scalar variable, found {float_static_count}"

    print("test_moving_average_hls passed!")


def test_stateful_isolate_hls():
    """Test HLS code generation for isolated stateful functions"""

    def acc[T_in](x: "T_in") -> "T_in":
        state: T_in @ stateful = 0
        state = state + x
        return state

    def top(x: int32, y: int32) -> int32:
        res1 = acc[int32, "A1"](x)
        res2 = acc[int32, "A2"](y)
        return res1 + res2

    s1 = allo.customize(acc, instantiate=[int32])
    s2 = allo.customize(acc, instantiate=[int32])
    s = allo.customize(top)
    s.compose(s1, id="A1")
    s.compose(s2, id="A2")

    mod = s.build(target="vhls")
    code = mod.hls_code

    # Should have at least 2 static variables (one for each isolated instance)
    static_count = len(re.findall(r"\bstatic\b", code))
    assert (
        static_count >= 2
    ), f"Expected at least 2 static variables for isolated instances, found {static_count}"

    # Should have at least 2 static int32_t variables (for A1 and A2 states)
    int_static_count = len(re.findall(r"static\s+int32_t", code))
    assert (
        int_static_count >= 2
    ), f"Expected at least 2 static int32_t variables for isolated states, found {int_static_count}"

    # Verify isolation: should have distinct static variables for A1 and A2
    has_a1_static = re.search(r"static\s+int32_t\s+\w*A1\w*\s*=", code)
    has_a2_static = re.search(r"static\s+int32_t\s+\w*A2\w*\s*=", code)
    assert has_a1_static, "Expected static variable containing 'A1' identifier"
    assert has_a2_static, "Expected static variable containing 'A2' identifier"

    # Both static variables should be initialized to 0
    a1_init = re.search(r"static\s+int32_t\s+\w*A1\w*\s*=\s*\{?0\}?", code)
    a2_init = re.search(r"static\s+int32_t\s+\w*A2\w*\s*=\s*\{?0\}?", code)
    assert a1_init, "Expected A1 static variable to be initialized to 0"
    assert a2_init, "Expected A2 static variable to be initialized to 0"

    print("test_stateful_isolate_hls passed!")
