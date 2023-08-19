# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32, index


def test_gemm_grid_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_range_for():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_float():
    def gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0.0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_gemm_reduction_var():
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            v: int32 = 0
            for k in range(32):
                v += A[i, k] * B[k, j]
            C[i, j] = v
        return C

    s = allo.customize(gemm)
    # transformations are applied immediately
    s.split("i", 8)
    s.split("j", 8)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)


def test_nested_if():
    def kernel(a: int32, b: int32) -> int32:
        r: int32 = 0
        if a == 0:
            r = 1
        elif a == 1:
            r = 2
            if b == 2:
                r = 3
        else:
            r = 4
        return r

    s = allo.customize(kernel)
    print(s.module)


def test_rhs_binaryop():
    def kernel() -> int32[11]:
        v: int32 = 5
        res: int32[11] = 0
        res[0] = 1 + v
        res[1] = 1 - v
        res[2] = v * 3
        res[3] = 52 / v
        res[4] = 6 // v
        res[5] = 6 % v
        res[6] = 1 << v
        res[7] = 64 >> v
        res[8] = 1 & v
        res[9] = 1 | v
        res[10] = res[9]
        return res

    s = allo.customize(kernel, verbose=True)
    print(s.module)


def test_fcompute_function_wrapper():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0

        def foo(x: int32) -> int32:
            return x + 1

        for i in range(10):
            B[i] = foo(A[i])
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(10,)).astype(np.int32)
    np_C = np_A + 1
    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


def test_llvm_arg():
    def kernel(A: float32[10], B: int32, C: float32) -> float32:
        v: float32 = 0.0
        v = A[0] + float(B) + C
        return v

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.random((10,)).astype(np.float32)
    B = 1
    C = 2.0
    allo_B = mod(np_A, B, C)
    np.testing.assert_allclose(allo_B, np_A[0] + 3.0)


def test_fcompute_wrap_more():
    def kernel(A: int32[10]) -> int32[10]:
        def foo(x: index, A: int32[10]) -> int32:
            y: int32 = 0
            y = A[x] + 1
            return y

        B: int32[10] = 0
        for i in range(10):
            B[i] = foo(i, A)
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(10,)).astype(np.int32)
    np_C = np_A + 1
    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


def test_index_arg():
    def kernel(A: int32[10]) -> int32[10]:
        B: int32[10] = 0

        def foo(A: int32[10], x: index) -> int32:
            y: int32 = 0
            C: int32[10] = 0
            for i in range(10):
                C[i] = A[i] + 1
            y = C[x]
            return y

        for i in range(10):
            B[i] = foo(A, i)
        return B

    s = allo.customize(kernel)
    print(s.module)


@pytest.mark.skip(reason="Cannot pass type checking")
def test_polymorphism():
    T = float32
    M, N, K = 32, 32, 32

    def gemm(A: T[M, K], B: T[K, N]) -> T[M, N]:
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    print(s.module)
    mod = s.build()
    np_A = np.random.random((M, K)).astype(np.float32)
    np_B = np.random.random((K, N)).astype(np.float32)
    allo_C = mod(np_A, np_B)
    np.testing.assert_allclose(np_A @ np_B, allo_C, rtol=1e-5)

    T = int32
    M, N, K = 4, 4, 4
    s1 = allo.customize(gemm)
    print(s1.module)
    mod1 = s1.build()
    np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)
    allo_C = mod1(np_A, np_B)
    np.testing.assert_allclose(np_A @ np_B, allo_C, rtol=1e-5)


def test_math_scalar():
    M = 10
    K = 15
    N = 20
    A = np.float32(np.random.uniform(size=(M, K)))
    B = np.float32(np.random.uniform(size=(K, N)))

    def kernel(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
        C: float32[M, N] = 0.0
        D: float32[M, N] = 0.0
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]
        for i, j in allo.grid(M, N):
            D[i, j] = (allo.exp(C[i, j]) + allo.log(C[i, j])) / C[i, j]
        return D

    s = allo.customize(kernel)
    f = s.build()
    print(s.module)
    outs = np.zeros((M, N), dtype="float32")
    outs = f(A, B)
    np1 = np.matmul(A, B)
    np_outs = (np.exp(np1) + np.log(np1)) / np1
    np.testing.assert_allclose(outs, np_outs, atol=1e-3)


def test_no_init_scalar():
    def kernel() -> int32:
        v: int32
        return v

    s = allo.customize(kernel)
    print(s.module)


def test_copy_memref():
    M, N = 2, 2

    def kernel() -> int32[M, N]:
        temp: int32[M, N] = 0
        outp: int32[M, N] = temp
        return outp

    s = allo.customize(kernel)
    print(s.module)
    f = s.build(target="vhls")
    print(f)


def test_copy_scalar():
    def kernel() -> int32:
        temp: int32 = 0
        outp: int32 = temp
        return outp

    s = allo.customize(kernel)
    print(s.module)
    f = s.build(target="vhls")
    print(f)


def test_copy_arg():
    M, N = 2, 2

    def kernel(inp: int32[M, N]) -> int32[M, N]:
        outp: int32[M, N] = inp
        return outp

    s = allo.customize(kernel)
    print(s.module)
    f = s.build(target="vhls")
    print(f)

    mod = s.build()
    np_inp = np.random.randint(0, 10, size=(M, N)).astype(np.int32)
    np_outp = mod(np_inp)
    assert np.array_equal(np_inp, np_outp)


def test_copy_arg_scalar():
    def kernel(inp: int32) -> int32:
        temp: int32 = inp
        outp: int32
        outp = temp * temp
        return outp

    s = allo.customize(kernel)
    print(s.module)
    f = s.build(target="vhls")
    print(f)

    mod = s.build()
    inp = 5
    outp = mod(inp)
    assert np.array_equal(inp * inp, outp)


if __name__ == "__main__":
    pytest.main([__file__])
