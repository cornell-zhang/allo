# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32


def test_nested_functions():
    M, K, N = 32, 32, 32

    def matrix_add(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N] = 0
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] + 1
        return B

    def gemm(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]
        return C

    def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C = gemm(A, B)
        D = matrix_add(C)
        return D

    # Separate compilation (just for testing)
    s_gemm = allo.customize(gemm)
    mod_gemm = s_gemm.build()

    # Top-level
    s = allo.customize(top)
    print(s.module)
    mod = s.build()

    # Testing
    np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)
    np_D = np.matmul(np_A, np_B)
    np_C = mod_gemm(np_A, np_B)
    assert np.array_equal(np_C, np_D)

    np_A = np.random.randint(0, 10, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(K, N)).astype(np.int32)
    np_D = np_A @ np_B + 1
    np_C = mod(np_A, np_B)
    assert np.array_equal(np_C, np_D)


def test_nested_functions_2():
    M, K, N = 32, 32, 32

    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]) -> None:
        for i, j in allo.grid(M, N):
            for k in allo.reduction(K):
                C[i, j] += A[i, k] * B[k, j]

    def top(A: int32[M, K], B: int32[K, N]) -> int32[M, N]:
        C: int32[M, N] = 0
        gemm(A, B, C)
        return C

    s1 = allo.customize(gemm)
    s1.reorder("k", "j")
    s1.partition(s1.C, dim=2)
    s1.buffer_at(s1.C, axis="i")
    s1.pipeline("j")
    # Top-level
    s = allo.customize(top)
    s.compose(s1)
    print(s.module)
    mod = s.build()

    # Testing
    np_A = np.random.randint(0, 100, size=(M, K)).astype(np.int32)
    np_B = np.random.randint(0, 100, size=(K, N)).astype(np.int32)
    np_C = mod(np_A, np_B)
    np_D = np.matmul(np_A, np_B)
    assert np.array_equal(np_C, np_D)
    print("Success!")


def test_nested_functions_3():
    M = 1024
    N = 1024
    K = 1024

    def gemm1(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i in range(M):
            for j in range(N):
                v: float32 = 0.0
                for k in allo.reduction(K):
                    v += inp[i, k] * W[k, j]
                outp[i, j] = v + B[j]
        return outp

    def gemm2(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i in range(M):
            for j in range(N):
                for k in allo.reduction(K):
                    outp[i, j] += inp[i, k] * W[k, j]
                outp[i, j] += B[j]
        return outp

    def gemm3(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i in range(M):
            for j in range(N):
                outp[i, j] = B[j]
        for i in range(M):
            for j in range(N):
                for k in allo.reduction(K):
                    outp[i, j] += inp[i, k] * W[k, j]
        return outp

    s1 = allo.customize(gemm1)
    print(s1.module)
    s2 = allo.customize(gemm2)
    print(s2.module)
    s3 = allo.customize(gemm3)
    print(s3.module)
    f = s1.build(target="vhls")
    print(f)


def test_triple_call():
    M = 12
    N = 768
    K = 768
    H = 64

    def Linear_layer(
        inp: float32[M, K], W: float32[K, N], B: float32[N]
    ) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="gemm"):
            for k in allo.reduction(K):
                outp[i, j] += inp[i, k] * W[k, j]
        for i, j in allo.grid(M, N, name="bias"):
            outp[i, j] += B[j]
        return outp

    def Add1(inp: float32[M, N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, H, name="add"):
            outp[i, j] = inp[i, j] + 1.0
        return outp

    def Add2(inp: float32[M, N]) -> float32[M, N]:
        outp = Add1(inp)
        return outp

    def Top(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp = Linear_layer(inp, W, B)
        outp = Add2(outp)
        return outp

    s = allo.customize(Top)
    print(s.module)


def test_compose_nested():
    M, N, K = 4, 4, 4

    def Linear_layer(
        inp: float32[M, K], W: float32[K, N], B: float32[N]
    ) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="gemm"):
            for k in allo.reduction(K):
                outp[i, j] += inp[i, k] * W[k, j]
        for i, j in allo.grid(M, N, name="bias"):
            outp[i, j] += B[j]
        return outp

    def Add1(inp: float32[M, N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="add"):
            outp[i, j] = inp[i, j] + 1.0
        return outp

    def Add2(inp: float32[M, N]) -> float32[M, N]:
        outp = Add1(inp)
        return outp

    def Top(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        outp = Linear_layer(inp, W, B)
        outp = Add2(outp)
        return outp

    s_add2 = allo.customize(Add2)
    s_add2.partition(s_add2.inp)
    print(s_add2.module)
    s = allo.customize(Top)
    s.compose(s_add2)
    print(s.module)

    f = s.build(target="vhls")
    print(f)


def test_double_partition():
    M = 4
    N = 4
    K = 4

    def Linear_layer1(
        inp: float32[M, K], W: float32[K, N], B: float32[N]
    ) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="gemm"):
            for k in allo.reduction(K):
                outp[i, j] += inp[i, k] * W[k, j]
        for i, j in allo.grid(M, N, name="bias"):
            outp[i, j] += B[j]
        return outp

    def Linear_layer2(
        inp: float32[M, K], W: float32[K, N], B: float32[N]
    ) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="gemm"):
            for k in allo.reduction(K):
                outp[i, j] += inp[i, k] * W[k, j]
        for i, j in allo.grid(M, N, name="bias"):
            outp[i, j] += B[j]
        return outp

    def Add(inp1: float32[M, N], inp2: float32[M, N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="add"):
            outp[i, j] = inp1[i, j] + inp2[i, j]
        return outp

    def Top(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        add1 = Linear_layer1(inp, W, B)
        add2 = Linear_layer2(inp, W, B)
        outp1 = Add(add1, add2)
        return outp1

    s_add = allo.customize(Add)
    s_add.partition(s_add.inp1, partition_type=1, dim=2, factor=2)
    s_add.partition(s_add.inp2, partition_type=1, dim=2, factor=2)
    print(s_add.module)
    s = allo.customize(Top)
    s.compose(s_add)
    f = s.build(target="vhls")
    print(f)


def test_output_partition_compose():
    M, N, K = 4, 4, 4

    def Linear_layer(
        inp: float32[M, K], W: float32[K, N], B: float32[N]
    ) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="gemm"):
            for k in allo.reduction(K):
                outp[i, j] += inp[i, k] * W[k, j]
        for i, j in allo.grid(M, N, name="bias"):
            outp[i, j] += B[j]
        return outp

    def Add(inp1: float32[M, N], inp2: float32[M, N]) -> float32[M, N]:
        outp: float32[M, N] = 0.0
        for i, j in allo.grid(M, N, name="add"):
            outp[i, j] = inp1[i, j] + inp2[i, j]
        return outp

    def Top(inp: float32[M, K], W: float32[K, N], B: float32[N]) -> float32[M, N]:
        add1 = Linear_layer(inp, W, B)
        add2 = Linear_layer(inp, W, B)
        outp1 = Add(add1, add2)
        return outp1

    s_ll = allo.customize(Linear_layer)
    s_ll.partition(s_ll.outp, partition_type=1, dim=2, factor=2)
    s = allo.customize(Top)
    s.compose(s_ll)
    print(s.module)

    f = s.build(target="vhls")
    print(f)


def test_partition_and_compose():
    inp_num = 12
    inp_len = 768
    Max_size = 12

    def Linear_layer_q(
        inp: float32[inp_num, inp_len],
        W: float32[inp_len, inp_len],
        B: float32[inp_len],
    ) -> float32[inp_num, inp_len]:
        outp: float32[inp_num, inp_len]
        for i, j in allo.grid(inp_num, inp_len, name="bias"):
            outp[i, j] = B[j]
        for i, j, k in allo.grid(inp_num, inp_len, inp_len, name="gemm"):
            outp[i, j] += inp[i, k] * W[j, k]
        return outp

    def top(
        inp: float32[inp_num, inp_len],
        W: float32[inp_len, inp_len],
        B: float32[inp_len],
    ) -> float32[inp_num, inp_len]:
        outp: float32[inp_num, inp_len]
        outp = Linear_layer_q(inp, W, B)
        return outp

    s_q = allo.customize(Linear_layer_q)
    s_q.partition(s_q.inp, partition_type=2, dim=1, factor=Max_size)
    s_q.partition(s_q.W, partition_type=2, dim=1, factor=Max_size)
    print(s_q.module)
    s = allo.customize(top)
    s.compose(s_q)
    print(s.module)
    print(s.build(target="vhls"))


if __name__ == "__main__":
    pytest.main([__file__])
