# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
from allo.ir.types import bool, int8, int32, float32, index
import allo.backend.hls as hls


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

    # 5. HLS CSIM
    if not hls.is_available("vitis_hls"):
        print("Vitis HLS not found, skipping...")
        return
    hls_mod = s.build(
        target="vitis_hls",
        mode="csim",
        project=f"test_gemm.prj",
    )
    csim_out = np.zeros((32, 32), dtype=np.int32)
    hls_mod(np_A, np_B, csim_out)
    np.testing.assert_allclose(csim_out, np_C, atol=1e-3)
    print("Passed HLS csim test!")


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


def test_negative_step_for():
    N = 256

    def kernel(x: int32[N], y: int32[N]):
        for i in range(N - 1, -1, -1):
            y[i] = x[i]

    with pytest.raises(RuntimeError):
        s = allo.customize(kernel)


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


def test_not():
    def kernel[Ty](flag: bool) -> "Ty":
        X: Ty
        if not flag:
            X = 1
        else:
            X = 0
        return X

    s = allo.customize(kernel, instantiate=[int8])
    print(s.module)
    mod = s.build()
    assert mod(True) == 0
    assert mod(False) == 1


def test_complex_not():
    def kernel[Ty](inp: Ty) -> "Ty":
        return 3 if not (inp + 1 > 5) else 4

    s = allo.customize(kernel, instantiate=[int8])
    print(s.module)
    mod = s.build()
    assert mod(4) == 3
    assert mod(5) == 4


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


def test_dynamic_shape():
    def kernel(A: float32[...], B: float32[...], size: int32):
        for i in range(size):
            B[i] = A[i]

    s = allo.customize(kernel)
    print(s.module)
    np_A = np.random.random((256,)).astype(np.float32)
    allo_A = np.zeros((256,)).astype(np.float32)
    mod = s.build()
    mod(np_A, allo_A, 256)
    np.testing.assert_allclose(np_A, allo_A)
    code = s.build(target="vhls")
    print(code)


def test_build_none_return():
    def kernel0(A: int32[32]):
        return None

    s0 = allo.customize(kernel0)

    def kernel1(A: int32[32]) -> None:
        return None

    s1 = allo.customize(kernel1)

    def kernel2(A: int32[32]):
        return

    s2 = allo.customize(kernel2)

    def kernel3(A: int32[32]) -> None:
        return

    s3 = allo.customize(kernel3)

    def kernel4(A: int32[32]):
        pass

    s4 = allo.customize(kernel4)

    def kernel5(A: int32[32]) -> None:
        pass

    s5 = allo.customize(kernel5)

    def kernel6(A: int32[32]) -> int32:
        return

    with pytest.raises(RuntimeError):
        s6 = allo.customize(kernel6)

    def kernel7(A: int32[32]) -> int32:
        pass

    with pytest.raises(RuntimeError):
        s7 = allo.customize(kernel7)


def test_comments():
    def top(x_in: "int8[1]") -> "int8":
        """Test text"""
        return x_in[0]

    print(allo.customize(top, verbose=True).build()(np.array([5], dtype=np.int8)))


def test_size1_array():
    def kernel[Ty](X: "Ty[1]"):
        a: Ty

    def top[Ty](X_buf: "Ty[2, 2, 1]"):
        kernel[Ty](X_buf[0, 0])

    s = allo.customize(top, instantiate=[int8], verbose=True)
    print(s.module)


def test_tuple():
    def callee(a: float32, b: float32) -> (float32, float32):
        c: float32 = a + b
        d: float32 = a - b
        return c, d

    def kernel(A: float32[10], B: float32[10]) -> (float32[10], float32[10]):
        C: float32[10] = 0
        D: float32[10] = 0
        for i in range(10):
            C[i], D[i] = callee(A[i], B[i])
        return C, D

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.random((10,)).astype(np.float32)
    np_B = np.random.random((10,)).astype(np.float32)
    np_C, np_D = mod(np_A, np_B)
    np_C_ref = np.zeros((10,), dtype=np.float32)
    np_D_ref = np.zeros((10,), dtype=np.float32)
    for i in range(10):
        np_C_ref[i], np_D_ref[i] = callee(np_A[i], np_B[i])
    np.testing.assert_allclose(np_C, np_C_ref)
    np.testing.assert_allclose(np_D, np_D_ref)


@pytest.mark.parametrize("T", [int8, int32, float32])
def test_minmax(T):
    def kernel(A: T[10]) -> (T[2], T[2]):
        min_val: T[2] = 0x3F3F3F3F
        max_val: T[2] = -0x3F3F3F3F
        for i in range(10):
            min_val[0] = min(min_val[0], A[i])
            max_val[0] = max(max_val[0], A[i])
        return min_val, max_val

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    if T == int8:
        np_A = np.random.randint(-64, 64, size=(10,)).astype(np.int8)
    elif T == int32:
        np_A = np.random.randint(-1000, 1000, size=(10,)).astype(np.int32)
    elif T == float32:
        np_A = np.random.random((10,)).astype(np.float32)
    allo_min, allo_max = mod(np_A)
    assert allo_min[0] == np.min(np_A)
    assert allo_max[0] == np.max(np_A)
    mod = s.build(target="vhls")
    assert "min" in mod.hls_code
    assert "max" in mod.hls_code


def test_minmax_cast():
    def kernel(A: int8[2]) -> int32[2]:
        res: int32[2] = 0
        res[0] = min(A[0], 0)
        res[1] = max(A[1], 0.0)
        return res

    s = allo.customize(kernel)
    print(s.module)
    mod = s.build()
    np_A = np.random.randint(-64, 64, size=(2,)).astype(np.int8)
    allo_B = mod(np_A)
    assert allo_B[0] == min(np_A[0], 0)
    assert allo_B[1] == max(np_A[1], 0.0)
    mod = s.build(target="vhls")
    print(mod)
    assert "min" in mod.hls_code
    assert "max" in mod.hls_code
    assert "(float)" in mod.hls_code


def test_scalar():
    def kernel() -> int32:
        a: int32 = 0
        b: int32 = a + 1
        return b

    s = allo.customize(kernel)
    print(s.module)
    assert "%alloc[]" in str(s.module)
    mod = s.build()
    assert mod() == 1
    mod = s.build(target="vhls")
    assert "," not in mod.hls_code


if __name__ == "__main__":
    pytest.main([__file__])
