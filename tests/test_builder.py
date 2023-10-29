# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import int4, int8, int16, int32, float32, index, Int, UInt


def test_grid_for_gemm():
    # This test is to make sure the whole flow works properly.
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        # Use grid_for with name annotation
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    # 1. Create customization
    s = allo.customize(gemm)
    print(s.module)

    # 2. Apply transformations and make sure each step the module can be printed
    s.split("i", 8)
    print(s.module)
    s.split("j", 8)
    print(s.module)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)
    # Make sure the generated loops are correct and ordered
    loops = s.get_loops()
    expected = ["i.outer", "j.outer", "i.inner", "j.inner", "k"]
    assert expected == list(loops.C.loops.keys())

    # 3. Build and run
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    np_C_allo = mod(np_A, np_B)
    np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-5)

    # 4. Generate HLS module
    mod = s.build(target="vhls")
    hls_code = mod.hls_code
    loop_labels = ["l_C_i_outer", "l_j_outer", "l_i_inner", "l_j_inner", "l_k"]
    for label in loop_labels:
        assert label in hls_code


def test_all_gemm():
    def range_for_gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(range_for_gemm)
    print(s.module)

    def float_gemm(A: float32[32, 32], B: float32[32, 32]) -> float32[32, 32]:
        C: float32[32, 32] = 0.0
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(float_gemm)
    print(s.module)

    def reduction_gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j in allo.grid(32, 32):
            v: int32 = 0
            for k in range(32):
                v += A[i, k] * B[k, j]
            C[i, j] = v
        return C

    s = allo.customize(reduction_gemm)
    print(s.module)


def test_range_for():
    def kernel(A: int32[20]):
        for i in range(10):
            A[i] = i
        for i in range(10, 20):
            A[i] = i
        for i in range(0, 20, 2):
            A[i] = i * 2

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.zeros((20,), dtype=np.int32)
    kernel(np_A)
    np_B = np.zeros((20,), dtype=np.int32)
    mod(np_B)
    np.testing.assert_allclose(np_A, np_B)


def test_variable_bound_for():
    def kernel(A: int32[10]):
        for i in range(10):
            for j in range(i + 1, 10):
                for k in range(j * 2, 10):
                    A[k] += i - j

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.zeros((10,), dtype=np.int32)
    kernel(np_A)
    np_B = np.zeros((10,), dtype=np.int32)
    mod(np_B)
    np.testing.assert_allclose(np_A, np_B)


def test_variable_bound_for_2():
    def kernel() -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            for j in range(i, i + 1):
                B[i] += j
        return B

    s = allo.customize(kernel)
    print(s.module)


def test_scf_for():
    def kernel(A: int32[10], B: int32[10]):
        for i in range(10):
            for j in range(A[i], 10, A[i]):
                for k in range(A[i] - 1, A[i] + 2):
                    B[k] += i - j

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    mod = s.build()
    np_A = np.zeros((10,), dtype=np.int32) + 1
    np_B = np.zeros((10,), dtype=np.int32)
    kernel(np_A, np_B)
    np_C = np.zeros((10,), dtype=np.int32) + 1
    np_D = np.zeros((10,), dtype=np.int32)
    mod(np_C, np_D)
    np.testing.assert_allclose(np_B, np_D)


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
    mod = s.build()
    assert mod(0, 0) == kernel(0, 0)
    assert mod(1, 1) == kernel(1, 1)
    assert mod(1, 2) == kernel(1, 2)


def test_logic_and_or():
    def kernel(A: int32[3], b: int32) -> int32:
        r: int32 = 0
        if A[0] > 0 and b < 0:
            r = 1
        elif A[1] * 2 <= 1 or b + 1 >= 1:
            r = 2
        elif A[2] != 3:
            r = 3
        return r

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    np_A = np.array([0, 1, 2], dtype=np.int32)
    mod = s.build()
    assert mod(np_A, 0) == kernel(np_A, 0)
    assert mod(np_A, 1) == kernel(np_A, 1)
    assert mod(np_A, 2) == kernel(np_A, 2)


def test_assign_logic():
    def kernel(A: int32) -> int32:
        B: int32 = 0
        if A > B:
            B = A
        return B

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    mod = s.build()
    assert mod(2) == kernel(2)


def test_while_basic():
    def kernel(A: int32[10]):
        i: index = 0
        while i < 10:
            A[i] = i
            i += 1

    s = allo.customize(kernel, verbose=True)
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(10, size=(10,))
    np_A_copy = np_A.copy()
    kernel(np_A)
    mod(np_A_copy)
    assert np.array_equal(np_A, np_A_copy)


def test_select():
    def kernel(A: int32[32]) -> int32[32]:
        B: int32[32] = 0
        for i in range(32):
            B[i] = 1 if A[i] % 2 == 0 else 0
        return B

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(32,)).astype(np.int32)
    np_B = mod(np_A)
    np.testing.assert_allclose(np_B, np_A % 2 == 0)


def test_select_cast():
    def kernel(A: int32[32], B: int32[32]):
        for i in range(32):
            B[i] = (i * 2) if A[i] % 2 == 0 else 0

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(32,)).astype(np.int32)
    np_B = np.zeros((32,), dtype=np.int32)
    kernel(np_A, np_B)
    np_C = np.zeros((32,), dtype=np.int32)
    mod(np_A, np_C)
    np.testing.assert_allclose(np_B, np_C)


def test_unary():
    def kernel() -> int32:
        v: int32 = 5
        vi: int32 = -(v + 1)
        vf: float32 = -(v + 1.0)
        return +(vi + vf)

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np.testing.assert_allclose(mod(), kernel())


def test_rhs_binaryop():
    def kernel() -> int32[11]:
        v: int32 = 5
        res: int32[11] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        res[0] = 1 + v
        res[1] = 1 - v
        res[2] = v * 3
        # One tricky thing is that Python does not require all
        # the elements in the list to be the same type,
        # so the following result becomes 10.4 (float);
        # while in Allo, the array can only have one type,
        # so the result is 10 (int).
        # res[3] = 52 / v
        res[4] = 6 // v
        res[5] = 6 % v
        res[6] = 1 << v
        res[7] = 64 >> v
        res[8] = 1 & v
        res[9] = 1 | v
        res[10] = res[9]
        return res

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np.testing.assert_allclose(mod(), kernel())


def test_nested_func_def():
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
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(10,)).astype(np.int32)
    np_C = np_A + 1
    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


def test_llvm_scalar_arg():
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
    np.testing.assert_allclose(allo_B, kernel(np_A, B, C))


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
    assert np.array_equal(kernel(5), mod(5))


def test_constexpr():
    M = 10

    def kernel(A: int32[((M + 1) * 2) // 2]) -> float32[M + 1]:
        res: float32[M + 1] = 0
        for i in range(M + 1):
            res[i] = A[i] + 1
        return res

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=((M + 1) * 2) // 2).astype(np.int32)
    np_res = mod(np_A)
    np.testing.assert_allclose(np_res, np_A + 1)


def test_multiple_returns_1D():
    M = 10

    def kernel(A: int32[M], B: int32[M]) -> (int32[M], int32[M]):
        res0: int32[M] = 0
        res1: int32[M] = 0
        for i in range(M):
            res0[i] = A[i] + 1
            res1[i] = B[i] + 1
        return res0, res1

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.random.randint(0, 10, size=(M,)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(M,)).astype(np.int32)
    np_res0, np_res1 = mod(np_A, np_B)
    np.testing.assert_allclose(np_res0, np_A + 1)
    np.testing.assert_allclose(np_res1, np_B + 1)


def test_multiple_returns_4D():
    M = 10

    def kernel(
        A: float32[M, M, M, M], B: float32[M, M, M, M], C: float32[M, M]
    ) -> (float32[M, M, M, M], float32[M, M, M, M], float32[M, M]):
        res0: float32[M, M, M, M] = 0
        res1: float32[M, M, M, M] = 0
        for i, j, k, l in allo.grid(M, M, M, M):
            res0[i, j, k, l] = A[i, j, k, l] + 1
            res1[i, j, k, l] = B[i, j, k, l] + 1
        res2: float32[M, M] = 0
        for i, j in allo.grid(M, M):
            res2[i, j] = C[i, j] + 1
        return res0, res1, res2

    s = allo.customize(kernel)
    mod = s.build()
    np_A = np.random.random((M, M, M, M)).astype(np.float32)
    np_B = np.random.random((M, M, M, M)).astype(np.float32)
    np_C = np.random.random((M, M)).astype(np.float32)
    np_res0, np_res1, np_res2 = mod(np_A, np_B, np_C)
    np.testing.assert_allclose(np_res0, np_A + 1)
    np.testing.assert_allclose(np_res1, np_B + 1)
    np.testing.assert_allclose(np_res2, np_C + 1)


def test_subview():
    def kernel(A: int32[10, 10]) -> int32[10]:
        return A[5]

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    mod = s.build()
    assert np.array_equal(mod(np_A), kernel(np_A))

    def kernel(A: float32[5, 10, 15]) -> float32[15]:
        return A[3, 2]

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.random((5, 10, 15)).astype(np.float32)
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), kernel(np_A))

    def kernel(A: float32[5, 10, 15]) -> float32[10, 15]:
        return A[3]

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.random((5, 10, 15)).astype(np.float32)
    mod = s.build()
    np.testing.assert_allclose(mod(np_A), kernel(np_A))


def test_dynamic_subview():
    def kernel(A: float32[5, 10, 15], i: index, j: index) -> float32[15]:
        return A[i, j]

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.random((5, 10, 15)).astype(np.float32)
    mod = s.build()
    np.testing.assert_allclose(mod(np_A, 3, 3), kernel(np_A, 3, 3))


def test_subview_systolic():
    M, N, K = 2, 2, 2

    def kernel(
        A_in: int8[K],
        B_in: int8[K],
        A_out: int8[K],
        B_out: int8[K],
        C: int16[M, N],
        i: index,
        j: index,
    ):
        for k in range(K):
            a: int8 = A_in[k]
            b: int8 = B_in[k]
            C[i, j] += a * b
            A_out[k] = a
            B_out[k] = b

    def systolic_array(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
        A_fifo: int8[M, N + 1, K]
        B_fifo: int8[N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(N):
                B_fifo[n, 0, k] = B[k, n]
        for i, j in allo.grid(M, N, name="PE"):
            kernel(
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
        A_drain: int8[M]
        B_drain: int8[N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(N):
                B_drain[n] = B_fifo[n, M, k]

    s = allo.customize(systolic_array)
    print(s.module)

    mod = s.build()
    A = np.random.randint(-8, 8, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 8, size=(K, N)).astype(np.int8)
    allo_C = np.zeros((M, N), dtype=np.int16)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


def test_subview_systolic_dsp_packed_int4xint4():
    M, N, K = 2, 2, 2

    def kernel(
        A_in: int4[K],
        B_in: int4[K],
        A_out: int4[K],
        B_out: int4[K],
        C: int8[M, N],
        i: index,
        j: index,
    ):
        for k in range(0, K, 2):
            a0: int4 = A_in[k]
            a1: int4 = A_in[k + 1]
            b0: int4 = B_in[k]
            b1: int4 = B_in[k + 1]
            a0u: UInt(4) = 0
            a1u: UInt(4) = 0
            b0u: UInt(4) = 0
            b1u: UInt(4) = 0
            s0: UInt(1) = a0[3] ^ b0[3]
            s1: UInt(1) = a1[3] ^ b1[3]
            if a0 < 0: a0u = -a0
            else: a0u = a0
            if a1 < 0: a1u = -a1
            else: a1u = a1
            if b0 < 0: b0u = -b0
            else: b0u = b0
            if b1 < 0: b1u = -b1
            else: b1u = b1
            op0: UInt(27) = 0
            op1: UInt(18) = 0
            op0[0:4] = a0u
            op0[22:26] = a1u
            op1[0:4] = b0u
            op1[11:15] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(8) = res[0:8]
            res1u: UInt(8) = res[33:41]
            res0: int8 = 0
            res1: int8 = 0
            if s0: res0 = -res0u
            else: res0 = res0u
            if s1: res1 = -res1u
            else: res1 = res1u
            C[i, j] += res0
            C[i, j] += res1
            A_out[k] = a0
            A_out[k + 1] = a1
            B_out[k] =  b0
            B_out[k + 1] =  b1

    def systolic_array(A: int4[M, K], B: int4[K, N], C: int8[M, N]):
        A_fifo: int4[M, N + 1, K]
        B_fifo: int4[N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(N):
                B_fifo[n, 0, k] = B[k, n]
        for i, j in allo.grid(M, N, name="PE"):
            kernel(
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
            )
        A_drain: int4[M]
        B_drain: int4[N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(N):
                B_drain[n] = B_fifo[n, M, k]

    s = allo.customize(systolic_array)
    # print(s.module)

    mod = s.build()
    A = np.random.randint(-8, 7, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 7, size=(K, N)).astype(np.int8)
    allo_C = np.zeros((M, N), dtype=np.int8)
    mod(A, B, allo_C)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


def test_subview_systolic_dsp_packed_int4xint8():
    M, N, K = 4, 4, 4
    half_N = 2

    def kernel(
        A_in: int8[K], # not bit-packed
        B_in: int8[K], # bit-packed, each element is 4 bits
        A_out: int8[K],
        B_out: int8[K],
        C: int32[M, N], # bit-packed, each element is 16 bits
        i: index,
        j: index,
    ):
        for k in range(K):
            a: int8 = A_in[k]
            b_packed: int8 = B_in[k]
            b0: int4 = b_packed[0:4]
            b1: int4 = b_packed[4:8]
            au : UInt(8) = 0
            b0u: UInt(4) = 0
            b1u: UInt(4) = 0
            s0: UInt(1) = a[7] ^ b0[3]
            s1: UInt(1) = a[7] ^ b1[3]
            if a < 0: au = 0-a
            else: au = a
            if b0 < 0: b0u = 0-b0
            else: b0u = b0
            if b1 < 0: b1u = 0-b1
            else: b1u = b1
            op0: UInt(18) = 0
            op1: UInt(27) = 0
            op0[0:8] = au
            op1[0:4] = b0u
            op1[13:17] = b1u
            res: UInt(48) = op0 * op1
            res0u: UInt(12) = res[0:12]
            res1u: UInt(12) = res[13:25]
            res0: int16 = 0
            res1: int16 = 0
            if s0: res0 = 0-res0u
            else: res0 = res0u
            if s1: res1 = 0-res1u
            else: res1 = res1u
            c_packed : int32 = C[i, j]
            c0: int16 = c_packed[0:16]
            c1: int16 = c_packed[16:32]
            c_packed[0:16] = c0 + res0
            c_packed[16:32] = c1 + res1
            C[i, j] = c_packed
            A_out[k] = a
            B_out[k] = b_packed

    def systolic_array(A: int8[M, K], B: int4[K, N], C: int16[M, N]):
        # bitpack B
        B_packed: int8[K, half_N] = 0
        for k in range(K):
            for n in range(half_N):
                B_packed[k, n][0:4] = B[k, n * 2]
                B_packed[k, n][4:8] = B[k, n * 2 + 1]
        
        A_fifo: int8[M, half_N + 1, K]
        B_fifo: int8[half_N, M + 1, K]

        for k in range(K, name="data_load"):
            for m in range(M):
                A_fifo[m, 0, k] = A[m, k]
            for n in range(half_N):
                B_fifo[n, 0, k] = B_packed[k, n]
        C_packed: int32[M, half_N] = 0
        for i, j in allo.grid(M, half_N, name="PE"):
            kernel(
                A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C_packed, i, j
            )
        A_drain: int8[M]
        B_drain: int8[half_N]
        for k in range(K, name="data_drain"):
            for m in range(M):
                A_drain[m] = A_fifo[m, N, k]
            for n in range(half_N):
                B_drain[n] = B_fifo[n, M, k]
        # unpack C
        for i in range(M):
            for j in range(half_N):
                C[i, j * 2] = C_packed[i, j][0: 16]
                C[i, j * 2 + 1] = C_packed[i, j][16: 32]

    s = allo.customize(systolic_array)
    # print(s.module)

    mod = s.build()
    A = np.random.randint(-128, 127, size=(M, K)).astype(np.int8)
    B = np.random.randint(-8, 7, size=(K, N)).astype(np.int8)
    np_C = A.astype(np.int16) @ B.astype(np.int16)
    allo_C = np.zeros((M, N), dtype=np.int16)
    mod(A, B, allo_C)
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
