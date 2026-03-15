# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, float32, bool
from allo.spmw import kernel


def test_basic_call():
    @kernel
    def helper_func(x: int32) -> int32:
        return x * 2

    @kernel
    def helper_func2(x: int32) -> int32:
        return helper_func(x) * 2

    @kernel
    def helper_func3(x: int32) -> int32:
        return helper_func(x) * helper_func(x)

    @kernel
    def kernel1(x: int32) -> int32:
        ret: int32 = helper_func(x)
        return ret

    s = process(kernel1)
    assert s(2) == 4
    assert s(3) == 6
    assert s(-4) == -8

    @kernel
    def kernel2(x: int32) -> int32:
        ret: int32 = helper_func2(x)
        return ret

    s = process(kernel2)
    assert s(2) == 8
    assert s(3) == 12
    assert s(-4) == -16

    @kernel
    def kernel3(x: int32) -> int32:
        ret: int32 = helper_func3(x)
        return ret

    s = process(kernel3)
    assert s(2) == 16
    assert s(3) == 36
    assert s(-4) == 64

    @kernel
    def kernel4(x: int32) -> int32[4]:
        ret: int32[4]
        ret[0], ret[1], ret[2], ret[3] = (
            helper_func(x),
            helper_func(x),
            helper_func(x),
            helper_func(x),
        )
        return ret

    s = process(kernel4)
    assert np.array_equal(s(2), np.array([4, 4, 4, 4]))
    assert np.array_equal(s(3), np.array([6, 6, 6, 6]))
    assert np.array_equal(s(-4), np.array([-8, -8, -8, -8]))

    print("test_basic_call passed")


def test_call_with_tensor_args():
    @kernel
    def helper_func(x: int32) -> int32[4]:
        return x * 2

    @kernel
    def helper_func2(x: int32) -> int32[4]:
        return helper_func(x) * 2

    @kernel
    def helper_func3(x: int32[4], y: int32[4]) -> int32[4]:
        ret: int32[4] = 0
        for i in range(4):
            ret[i] = x[i] * y[i]
        return ret

    @kernel
    def helper_func4(x: int32[4], y: int32[4]):
        for i in range(4):
            y[i] = 1 + helper_func(x[i])[0]

    @kernel
    def kernel1(x: int32) -> int32[4]:
        ret: int32[4] = helper_func(x)
        return ret

    s = process(kernel1)
    assert np.array_equal(s(2), np.array([4, 4, 4, 4]))
    assert np.array_equal(s(3), np.array([6, 6, 6, 6]))
    assert np.array_equal(s(-4), np.array([-8, -8, -8, -8]))

    @kernel
    def kernel2(x: int32) -> int32[4]:
        ret: int32[4] = helper_func2(x)
        return ret

    s = process(kernel2)
    assert np.array_equal(s(2), np.array([8, 8, 8, 8]))
    assert np.array_equal(s(3), np.array([12, 12, 12, 12]))
    assert np.array_equal(s(-4), np.array([-16, -16, -16, -16]))

    @kernel
    def kernel3(x: int32[4], y: int32[4]) -> int32[4]:
        ret: int32[4] = helper_func3(x, y)
        return ret

    s = process(kernel3)
    np_A = np.random.randint(0, 10, size=(4,)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4,)).astype(np.int32)
    assert np.array_equal(s(np_A, np_B), np_A * np_B)

    @kernel
    def kernel4(x: int32[4], y: int32[4]):
        helper_func4(x, y)

    s = process(kernel4)
    np_A = np.random.randint(0, 10, size=(4,)).astype(np.int32)
    np_B = np.zeros(
        4,
    ).astype(np.int32)
    s(np_A, np_B)
    assert np.array_equal(np_B, np_A * 2 + 1)

    print("test_call_with_tensor_args passed")


def test_call_recursive():
    @kernel
    def helper_func(x: int32) -> int32:
        ret: int32
        if x == 0:
            ret = 1
        else:
            ret = x * helper_func(x - 1)
        return ret

    @kernel
    def kernel1(x: int32) -> int32:
        ret: int32 = helper_func(x)
        return ret

    s = process(kernel1)
    assert s(2) == 2
    assert s(3) == 6
    assert s(4) == 24

    @kernel
    def is_even(x: int32) -> bool:
        ret: bool
        if x == 0:
            ret = True
        else:
            ret = is_odd(x - 1)
        return ret

    @kernel
    def is_odd(x: int32) -> bool:
        ret: bool
        if x == 0:
            ret = False
        else:
            ret = is_even(x - 1)
        return ret

    @kernel
    def kernel2(x: int32) -> bool:
        ret: bool = is_even(x)
        return ret

    s = process(kernel2)
    assert s(2) == True
    assert s(3) == False
    assert s(4) == True

    print("test_call_recursive passed")


def test_call_casting():
    @kernel
    def helper_func(x: int32) -> float32:
        return x

    @kernel
    def helper_func2(x: int32[4]) -> float32[4]:
        return x + 1

    @kernel
    def kernel1(x: int32) -> float32:
        return helper_func(x)

    s = process(kernel1)
    assert s(2) == 2.0
    assert s(3) == 3.0
    assert s(-4) == -4.0

    @kernel
    def kernel2(x: int32[4]) -> float32[4]:
        return helper_func2(x)

    s = process(kernel2)
    assert np.array_equal(
        s(np.array([2, 3, -4, 5]).astype(np.int32)), np.array([3.0, 4.0, -3.0, 6.0])
    )

    @kernel
    def kernel3(x: float32[4]) -> int32[4]:
        return helper_func2(x)

    s = process(kernel3)
    assert np.array_equal(
        s(np.array([2, 3, -4, 5]).astype(np.float32)), np.array([3, 4, -3, 6])
    )

    print("test_call_casting passed")


if __name__ == "__main__":
    test_basic_call()
    test_call_with_tensor_args()
    test_call_recursive()
    test_call_casting()
