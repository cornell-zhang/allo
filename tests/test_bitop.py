# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import uint1, uint2, int32, uint8, uint32, UInt, float16, float32


def test_scalar():
    def kernel(a: int32) -> int32:
        return a[28:32]

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    assert mod(0xABCD0123) == 0xA


def test_get_bit():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0]
        return B

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 1, rtol=1e-5, atol=1e-5)


def test_get_bit_slice():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            B[i] = (A[i] + 1)[0:2]
        return B

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.randint(10, size=(10,))
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), (np_A + 1) & 0b11, rtol=1e-5, atol=1e-5)


def test_reverse():
    def kernel(A: uint8[10]) -> uint8[10]:
        B: uint8[10] = 0
        for i in range(10):
            B[i] = (A[i][0:8]).reverse()
        return B

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    np_A = np.random.randint(10, size=(10,)).astype(np.uint8)
    golden = (np_A & 0xFF).astype(np.uint8)
    mod = s.build()
    ret = mod(np_A)
    for i in range(0, 10):
        x = np.unpackbits(golden[i])
        x = np.flip(x)
        y = np.unpackbits(ret[i])
        assert np.array_equal(x, y)


def test_set_bit_tensor():
    def kernel1(A: uint1[10], B: int32[10]):
        for i in range(10):
            b: int32 = B[i]
            b[0] = A[i]
            B[i] = b

    def kernel2(A: uint1[10], B: int32[10]):
        for i in range(10):
            B[i][0] = A[i]

    for kernel in [kernel1, kernel2]:
        s = allo.customize(kernel)
        print(s.module)
        np_A = np.random.randint(2, size=(10,))
        np_B = np.random.randint(10, size=(10,))
        golden = np_B & 0b1110 | np_A
        mod = s.build()
        mod(np_A, np_B)
        assert np.array_equal(golden, np_B)


def test_set_slice():
    def kernel(A: uint2[10], B: int32[10]):
        for i in range(10):
            B[i][0:2] = A[i]

    s = allo.customize(kernel)
    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(7, size=(10,))
    golden = (np_B & 0b1100) | np_A
    mod = s.build()
    mod(np_A, np_B)
    assert np.array_equal(golden, np_B)


def test_dynamic_index():
    def kernel(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1]

    s = allo.customize(kernel)
    np_B = np.zeros((11,), dtype=np.int32)
    mod = s.build()
    mod(1234, np_B)
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])


def test_dynamic_slice():
    def kernel(A: int32, B: int32[11]):
        for i in range(1, 12):
            B[i - 1] = A[i - 1 : i]

    s = allo.customize(kernel)
    np_B = np.zeros((11,), dtype=np.int32)
    mod = s.build()
    mod(1234, np_B)
    assert bin(1234) == "0b" + "".join([str(np_B[i]) for i in range(10, -1, -1)])


def test_bitcast_uint2float32():
    def kernel(A: uint32[10, 10]) -> float32[10, 10]:
        B: float32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j].bitcast()
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()

    A_np = np.random.randint(100, size=(10, 10)).astype(np.uint32)
    B_np = mod(A_np)
    answer = np.frombuffer(A_np.tobytes(), np.float32).reshape((10, 10))
    assert np.array_equal(B_np, answer)

    code = str(s.build(target="vhls"))
    assert "union" in code and "uint32" in code
    print("Passed!")


def test_bitcast_uint2float16():
    def kernel(A: int32[10, 10]) -> float16[10, 10]:
        B: float16[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j][0:16].bitcast()
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()

    A_np = np.random.randint(100, size=(10, 10)).astype(np.int32)
    B_np = mod(A_np)
    answer = np.frombuffer(A_np.astype(np.int16).tobytes(), np.float16).reshape(
        (10, 10)
    )
    assert np.array_equal(B_np, answer)

    code = str(s.build(target="vhls"))
    print(code)
    assert "union" in code and "half" in code
    print("Passed!")


def test_bitcast_float2uint():
    def kernel(A: float32[10, 10]) -> uint32[10, 10]:
        B: uint32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j].bitcast()
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()

    A_np = np.random.rand(10, 10).astype(np.float32)
    B_np = mod(A_np)
    answer = np.frombuffer(A_np.tobytes(), np.uint32).reshape((10, 10))
    assert np.array_equal(B_np, answer)

    code = str(s.build(target="vhls"))
    assert "union" in code and "uint32" in code
    print("Passed!")


def test_bitcast_float2int():
    def kernel(A: float32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j].bitcast()
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()

    A_np = np.random.rand(10, 10).astype(np.float32)
    B_np = mod(A_np)
    answer = np.frombuffer(A_np.tobytes(), np.int32).reshape((10, 10))
    assert np.array_equal(B_np, answer)

    code = str(s.build(target="vhls"))
    assert "union" in code and "int32" in code
    print(code)
    print("Passed!")


def test_packed_bconv2D_nchw():
    bs = 4
    ic, oc = 16, 32
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1
    L = ic * kh * kw
    packing_factor = 16
    kc = ic // packing_factor

    def bconv(
        A: UInt(packing_factor)[bs, kc, ih, iw], F: UInt(packing_factor)[oc, kc, kh, kw]
    ) -> int32[bs, oc, oh, ow]:
        B: int32[bs, oc, oh, ow] = 0
        for n, c, h, w in allo.grid(bs, oc, oh, ow):
            # popcount
            v: int32 = 0
            for rc, rh, rw, rb in allo.reduction(kc, kh, kw, packing_factor):
                v += (A[n, rc, h + rh, w + rw] ^ F[c, rc, rh, rw])[rb]
            B[n, c, h, w] = L - (v << 1)
        return B

    s = allo.customize(bconv)
    LB = s.reuse_at(s.A, axis="h")
    s.reuse_at(LB, axis="w")
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 2, size=(bs, ic, ih, iw))
    np_B = np.random.randint(0, 2, size=(oc, ic, kh, kw))
    np_C = np.zeros((bs, oc, oh, ow), np.int32)

    for n in range(0, bs):
        for c in range(0, oc):
            for y in range(0, oh):
                for x in range(0, ow):
                    for rc in range(0, ic):
                        for rh in range(0, kh):
                            for rw in range(0, kw):
                                np_C[n][c][y][x] += 1 - 2 * (
                                    np_A[n][rc][y + rh][x + rw] ^ np_B[c][rc][rh][rw]
                                )

    packed_A = np.ascontiguousarray(
        np.packbits(
            np_A.transpose((0, 2, 3, 1)).astype(np.bool_), axis=3, bitorder="little"
        )
        .view(np.uint16)
        .transpose((0, 3, 1, 2))
    )
    packed_B = np.ascontiguousarray(
        np.packbits(
            np_B.transpose((0, 2, 3, 1)).astype(np.bool_), axis=3, bitorder="little"
        )
        .view(np.uint16)
        .transpose((0, 3, 1, 2))
    )
    allo_C = mod(packed_A, packed_B)
    assert np.array_equal(np_C, allo_C)
    print("Passed!")


if __name__ == "__main__":
    pytest.main([__file__])
