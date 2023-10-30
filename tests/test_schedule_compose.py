# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int32, float32


def get_user_names(users):
    return set([user.path + ":" + user.name for user in users])


def test_use_def_chain():
    def foo2(A: int32) -> int32:
        B: int32 = A + 1
        return B

    def foo(A: int32) -> int32:
        B: int32 = (A - 1) / (A + 1)
        C: int32 = foo2(A) + B
        return C

    def kernel(A: int32) -> int32:
        B: int32 = A + 1
        C: int32 = A * B
        D: int32 = (C + 1) - (B * A)
        E: int32 = foo(D)
        return E

    s = allo.customize(kernel, verbose=True)
    assert get_user_names(s.use_def_chain.get_equivalent_tensors("kernel:D")) == set(
        ["foo:A", "foo2:A"]
    )
    assert get_user_names(s.use_def_chain["kernel:A"].users) == set(
        [
            "kernel:D",
            "kernel:C",
            "kernel:B",
        ]
    )
    assert get_user_names(s.use_def_chain["kernel:B"].users) == set(
        ["kernel:D", "kernel:C"]
    )
    assert get_user_names(s.use_def_chain["kernel:D"].users) == set(
        ["foo:A", "kernel:E"]
    )
    assert get_user_names(s.use_def_chain["foo:A"].users) == set(
        [
            "foo2:A",
            "foo:C",
            "foo:B",
        ]
    )
    assert get_user_names(s.use_def_chain["foo:C"].users) == set(["kernel:E"])


def test_use_def_chain_array():
    def kernel(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            for k in allo.reduction(32):
                C[i, j] += A[i, k] * B[k, j]
        return C

    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        ret = kernel(A, B)
        return ret

    s = allo.customize(gemm, verbose=True)
    print(s.module)
    assert get_user_names(s.use_def_chain["gemm:A"].users) == set(
        ["gemm:ret", "kernel:A"]
    )
    assert get_user_names(s.use_def_chain["gemm:B"].users) == set(
        ["gemm:ret", "kernel:B"]
    )
    assert get_user_names(s.use_def_chain["kernel:A"].users) == set(["kernel:C"])
    assert get_user_names(s.use_def_chain["kernel:B"].users) == set(["kernel:C"])
    assert get_user_names(s.use_def_chain["kernel:C"].users) == set(["gemm:ret"])


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
        out0 = Linear_layer(inp, W, B)
        out1 = Add2(out0)
        return out1

    s_add2 = allo.customize(Add2)
    s_add2.partition(s_add2.inp)
    print(s_add2.module)
    s = allo.customize(Top)
    s.compose(s_add2)
    print(s.module)

    f = s.build(target="vhls")
    print(f)


def test_double_partition():
    M, N, K = 4, 4, 4

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


def test_nested_compose_partition():
    M, N = 2, 2

    def matrix_addi(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] + 1
        return B

    s_addi = allo.customize(matrix_addi)
    s_addi.partition(s_addi.A)

    def matrix_addi_top(A: int32[M, N]) -> int32[M, N]:
        B = matrix_addi(A)
        return B

    s_addi_top = allo.customize(matrix_addi_top)
    s_addi_top.compose(s_addi)
    print(s_addi_top.module)

    def top(inp: int32[M, N]) -> int32[M, N]:
        outp = matrix_addi_top(inp)
        return outp

    s = allo.customize(top)
    # s.partition(s.inp)
    s.compose(s_addi_top)
    print(s.module)


def test_reuse_function_1():
    M, N = 2, 2

    def matrix_addi(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] + 1
        return B

    s_addi = allo.customize(matrix_addi)
    s_addi.partition(s_addi.A)

    def matrix_subi(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] - 1
        return B

    s_subi = allo.customize(matrix_subi)

    def top(inp: int32[M, N]) -> int32[M, N]:
        temp1 = matrix_addi(inp)
        temp2 = matrix_subi(temp1)
        outp = matrix_addi(temp2)
        return outp

    s = allo.customize(top)
    s.compose(s_addi)
    s.compose(s_subi)
    print(s.module)


def test_reuse_function_2():
    M, N = 2, 2

    def matrix_addi(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] + 1
        return B

    s_addi = allo.customize(matrix_addi)
    # s_addi.partition(s_addi.A)

    def matrix_subi(A: int32[M, N]) -> int32[M, N]:
        B: int32[M, N]
        for i, j in allo.grid(M, N):
            B[i, j] = A[i, j] - 1
        return B

    s_subi = allo.customize(matrix_subi)
    # s_subi.partition(s_subi.B)

    def top(inp: int32[M, N]) -> int32[M, N]:
        temp1 = matrix_addi(inp)
        temp2 = matrix_subi(temp1)
        temp3 = matrix_addi(temp2)
        outp = matrix_subi(temp3)
        return outp

    s = allo.customize(top)
    s.partition(s.outp)
    s.compose(s_addi)
    s.compose(s_subi)
    print(s.module)


if __name__ == "__main__":
    pytest.main([__file__])
