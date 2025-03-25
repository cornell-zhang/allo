# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import int32, float32
import allo.ir.types as T
import allo
from allo.customize import Schedule
import numpy as np
import os
import json
from typing import Dict, Any, Callable, Tuple, List


polybench_registry: dict[str, tuple[Callable, Callable, Callable]] = {}


def add_benchmark(name: str, sch_fn: Callable, np_fn: Callable, gen_fn: Callable):
    """Add a benchmark to the registry"""
    polybench_registry[name] = (sch_fn, np_fn, gen_fn)


def get_polybench(
    test_name: str, size: str = "small", concrete_type=float32
) -> Tuple[Schedule, Tuple[np.ndarray, ...], np.ndarray]:
    """Get a polybench test case with inputs and reference output

    Args:
        test_name: Name of the polybench test (e.g. 'three_mm', 'gemm', etc)
        size: Problem size ('mini', 'small', 'medium')
        dtype: Data type to use

    Returns:
        Tuple of:
        - Schedule object
        - Tuple of input matrices
        - Reference output from numpy implementation
    """
    if test_name not in polybench_registry:
        raise ValueError(f"Unknown benchmark {test_name}")

    # Load problem sizes from json
    setting_path = os.path.join(
        os.path.dirname(__file__), "../../examples/polybench/psize.json"
    )
    with open(setting_path, "r") as fp:
        psize = json.load(fp)

    if test_name not in psize:
        raise ValueError(f"No problem size defined for {test_name}")
    if size not in psize[test_name]:
        valid_sizes = list(psize[test_name].keys())
        raise ValueError(
            f"Size {size} not defined for {test_name}. Valid sizes are: {valid_sizes}"
        )

    # Get the size parameters
    params = psize[test_name][size]
    param_values = list(params.values())

    # Get the schedule, numpy, and generator implementations
    sch_fn, np_fn, gen_fn = polybench_registry[test_name]
    # Create schedule
    sch = sch_fn(concrete_type, *param_values)

    # Generate inputs using the provided generator function
    inputs = gen_fn(concrete_type, *param_values)

    # Get reference output
    ref_out = np_fn(*[np.copy(input) for input in inputs])

    return sch, inputs, ref_out


# Three-matrix multiplication (three_mm)
def three_mm_np(A, B, C, D):
    out_AB = np.dot(A, B)
    out_CD = np.dot(C, D)
    out_ABC = np.dot(out_AB, out_CD)
    return out_ABC


def three_mm(concrete_type, p, r, q, t, s):
    def mm1[
        DType: (float32, int32), P: int32, Q: int32, R: int32
    ](A: "DType[P, Q]", B: "DType[Q, R]", out_AB: "DType[P, R]"):
        for i0, j0 in allo.grid(P, R, name="mm1"):
            for k0 in allo.reduction(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    def mm2[
        DType: (float32, int32), R: int32, S: int32, T: int32
    ](C: "DType[R, S]", D: "DType[S, T]", out_CD: "DType[R, T]"):
        for i1, j1 in allo.grid(R, T, name="mm2"):
            for k1 in allo.reduction(S):
                out_CD[i1, j1] += C[i1, k1] * D[k1, j1]

    def mm3[
        DType: (float32, int32), P: int32, R: int32, T: int32
    ](out_AB: "DType[P, R]", out_CD: "DType[R, T]", out_ABC: "DType[P, T]"):
        for i2, j2 in allo.grid(P, T, name="mm3"):
            for k2 in allo.reduction(R):
                out_ABC[i2, j2] += out_AB[i2, k2] * out_CD[k2, j2]

    def kernel_3mm[
        DType: (float32, int32), P: int32, Q: int32, R: int32, S: int32, T: int32
    ](
        A: "DType[P, Q]", B: "DType[Q, R]", C: "DType[R, S]", D: "DType[S, T]"
    ) -> "DType[P, T]":
        out_AB: DType[P, R] = 0
        out_CD: DType[R, T] = 0
        output: DType[P, T] = 0
        mm1[DType, P, Q, R](A, B, out_AB)
        mm2[DType, R, S, T](C, D, out_CD)
        mm3[DType, P, R, T](out_AB, out_CD, output)
        return output

    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    sch1 = allo.customize(mm2, instantiate=[concrete_type, r, s, t])
    sch2 = allo.customize(mm3, instantiate=[concrete_type, p, r, t])
    sch = allo.customize(kernel_3mm, instantiate=[concrete_type, p, q, r, s, t])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.compose(sch2)

    return sch


# Two-matrix multiplication (two_mm)
def two_mm_np(A, B, C, D, alpha=0.1, beta=0.5):
    out_AB = np.dot(A, B)
    out_ABC = np.dot(out_AB, C)
    output = out_ABC * beta + D * alpha
    return output


def two_mm(concrete_type, p, r, q, s, alpha=0.1, beta=0.5):
    def mm1[
        T: (float32, int32), P: int32, Q: int32, R: int32
    ](A: "T[P, Q]", B: "T[Q, R]", out_AB: "T[P, R]"):
        for i0, j0 in allo.grid(P, R, name="mm1"):
            for k0 in allo.reduction(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    def mm2[
        T: (float32, int32), P: int32, R: int32, S: int32
    ](out_AB: "T[P, R]", C: "T[R, S]", out_ABC: "T[P, S]"):
        for i1, j1 in allo.grid(P, S, name="mm2"):
            for k1 in allo.reduction(R):
                out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]

    def ele_add[
        T: (float32, int32), P: int32, S: int32
    ](out_ABC: "T[P, S]", D: "T[P, S]", output: "T[P, S]"):
        for i2, j2 in allo.grid(P, S):
            output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha

    def kernel_2mm[
        T: (float32, int32), P: int32, R: int32, Q: int32, S: int32
    ](A: "T[P, Q]", B: "T[Q, R]", C: "T[R, S]", D: "T[P, S]") -> "T[P, S]":
        out_AB: T[P, R] = 0
        out_ABC: T[P, S] = 0
        output: T[P, S] = 0
        mm1[T, P, Q, R](A, B, out_AB)
        mm2[T, P, R, S](out_AB, C, out_ABC)
        ele_add[T, P, S](out_ABC, D, output)
        return output

    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    sch1 = allo.customize(mm2, instantiate=[concrete_type, p, r, s])
    sch2 = allo.customize(ele_add, instantiate=[concrete_type, p, s])
    sch = allo.customize(kernel_2mm, instantiate=[concrete_type, p, r, q, s])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.compose(sch2)

    return sch


# ATAX: Matrix Transpose and Vector Multiplication
def atax_np(A, x, y):
    temp = np.dot(A.T, np.dot(A, x))
    y[:] = temp
    return y


def atax(concrete_type, m, n):
    def vec_mul[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", tmp: "T[M]"):
        for i0 in allo.grid(M, name="matmul"):
            for j0 in allo.reduction(N):
                tmp[i0] += A[i0, j0] * x[j0]

    def vec_mul_t[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", tmp: "T[M]", y: "T[N]"):
        for i1 in allo.grid(N, name="matmul_t"):
            for j1 in allo.reduction(M):
                y[i1] += A[j1, i1] * tmp[j1]

    def kernel_atax[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", x: "T[N]", y: "T[N]"):
        tmp: T[M] = 0
        vec_mul[T, M, N](A, x, tmp)
        vec_mul_t[T, M, N](A, tmp, y)

    sch0 = allo.customize(vec_mul, instantiate=[concrete_type, m, n])
    sch1 = allo.customize(vec_mul_t, instantiate=[concrete_type, m, n])
    sch = allo.customize(kernel_atax, instantiate=[concrete_type, m, n])
    sch.compose(sch0)
    sch.compose(sch1)
    return sch


# BICG: BiCGStab Linear Solver Kernel
def bicg_np(A, s, q, p, r):
    q[:] = np.dot(A, p)
    s[:] = np.dot(A.T, r)
    return q, s


def bicg(concrete_type, m, n):
    def vec_mul[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", p: "T[N]", q: "T[M]"):
        for i0 in allo.grid(M, name="matmul"):
            for j0 in allo.reduction(N):
                q[i0] += A[i0, j0] * p[j0]

    def vec_mul_t[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", r: "T[M]", s: "T[N]"):
        for i1 in allo.grid(N, name="matmul_t"):
            for j1 in allo.reduction(M):
                s[i1] += A[j1, i1] * r[j1]

    def kernel_bicg[
        T: (float32, int32), M: int32, N: int32
    ](A: "T[M, N]", p: "T[N]", r: "T[M]", q: "T[M]", s: "T[N]"):
        vec_mul[T, M, N](A, p, q)
        vec_mul_t[T, M, N](A, r, s)

    sch0 = allo.customize(vec_mul, instantiate=[concrete_type, m, n])
    sch1 = allo.customize(vec_mul_t, instantiate=[concrete_type, m, n])
    sch = allo.customize(kernel_bicg, instantiate=[concrete_type, m, n])
    sch.compose(sch0)
    sch.compose(sch1)
    return sch


# GEMM: General Matrix Multiplication
def gemm_np(A, B, C, beta=0.1):
    out_AB = np.dot(A, B)
    out_ABC = beta * C + out_AB
    return out_ABC


def gemm(concrete_type, p, r, q, beta=0.1):
    def mm1[
        T: (float32, int32), P: int32, Q: int32, R: int32
    ](A: "T[P, Q]", B: "T[Q, R]", out_AB: "T[P, R]"):
        for i0, j0 in allo.grid(P, R, name="mm1"):
            for k0 in allo.reduction(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    def ele_add[
        T: (float32, int32), P: int32, R: int32
    ](out_AB: "T[P, R]", C: "T[P, R]", output: "T[P, R]"):
        for i2, j2 in allo.grid(P, R):
            output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]

    def kernel_gemm[
        T: (float32, int32), P: int32, Q: int32, R: int32
    ](A: "T[P, Q]", B: "T[Q, R]", C: "T[P, R]", output: "T[P, R]"):
        out_AB: T[P, R] = 0
        mm1[T, P, Q, R](A, B, out_AB)
        ele_add[T, P, R](out_AB, C, output)

    sch0 = allo.customize(mm1, instantiate=[concrete_type, p, q, r])
    sch1 = allo.customize(ele_add, instantiate=[concrete_type, p, r])
    sch = allo.customize(kernel_gemm, instantiate=[concrete_type, p, q, r])
    sch.compose(sch0)
    sch.compose(sch1)

    return sch


# GESUMMV: Scalar, Vector, Matrix Multiplication
def gesummv_np(A, B, x, alpha=1.0, beta=1.0):
    y = alpha * np.dot(A, x) + beta * np.dot(B, x)
    return y


def gesummv(concrete_type, n, alpha=1.0, beta=1.0):
    def kernel_gesummv[
        T: (float32, int32), N: int32
    ](A: "T[N, N]", B: "T[N, N]", x: "T[N]", y: "T[N]"):
        tmp: T[N]
        for i, j in allo.grid(N, N):
            tmp[i] += A[i, j] * x[j]
            y[i] += B[i, j] * x[j]
        for i in allo.grid(N):
            y[i] = alpha * tmp[i] + beta * y[i]

    sch = allo.customize(kernel_gesummv, instantiate=[concrete_type, n])
    return sch


# MVT: Matrix Vector Product and Transpose
def mvt_np(A, x1, x2, y1, y2):
    x1 += np.dot(A, y1)
    x2 += np.dot(A.T, y2)
    return x1, x2


def mvt(concrete_type, n):
    def vec_dot1[
        T: (float32, int32), N: int32
    ](A: "T[N, N]", y1: "T[N]", x1: "T[N]", out_x1: "T[N]"):
        for i0 in allo.grid(N, name="init"):
            out_x1[i0] = x1[i0]
        for i0 in allo.grid(N, name="vec_dot1"):
            for j0 in allo.reduction(N):
                out_x1[i0] += A[i0, j0] * y1[j0]

    def vec_dot2[
        T: (float32, int32), N: int32
    ](A: "T[N, N]", y2: "T[N]", x2: "T[N]", out_x2: "T[N]"):
        for i1 in allo.grid(N, name="init"):
            out_x2[i1] = x2[i1]
        for i1 in allo.grid(N, name="vec_dot2"):
            for j1 in allo.reduction(N):
                out_x2[i1] += A[j1, i1] * y2[j1]

    def kernel_mvt[
        T: (float32, int32), N: int32
    ](
        A: "T[N, N]",
        y1: "T[N]",
        y2: "T[N]",
        x1: "T[N]",
        x2: "T[N]",
        out_x1: "T[N]",
        out_x2: "T[N]",
    ):
        vec_dot1[T, N](A, y1, x1, out_x1)
        vec_dot2[T, N](A, y2, x2, out_x2)

    sch0 = allo.customize(vec_dot1, instantiate=[concrete_type, n])
    sch1 = allo.customize(vec_dot2, instantiate=[concrete_type, n])
    sch = allo.customize(kernel_mvt, instantiate=[concrete_type, n])
    sch.compose(sch0)
    sch.compose(sch1)
    return sch


def gen_three_mm_inputs(concrete_type, p, r, q, t, s):
    """Generate input matrices for three_mm benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        p, r, q, t, s: Matrix dimensions

    Returns:
        Tuple of input matrices (A, B, C, D)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (r, s)).astype(np_dtype)
    D = np.random.randint(-10, 10, (s, t)).astype(np_dtype)

    return A, B, C, D


def gen_two_mm_inputs(concrete_type, p, r, q, s, alpha=0.1, beta=0.5):
    """Generate input matrices for two_mm benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        p, r, q, s: Matrix dimensions
        alpha, beta: Scaling factors

    Returns:
        Tuple of input matrices (A, B, C, D)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (r, s)).astype(np_dtype)
    D = np.random.randint(-10, 10, (p, s)).astype(np_dtype)

    return A, B, C, D


def gen_atax_inputs(concrete_type, m, n):
    """Generate input matrices for atax benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        m, n: Matrix dimensions

    Returns:
        Tuple of input matrices (A, x, y)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (m, n)).astype(np_dtype)
    x = np.random.randint(-10, 10, n).astype(np_dtype)
    y = np.zeros(n, dtype=np_dtype)

    return A, x, y


def gen_bicg_inputs(concrete_type, m, n):
    """Generate input matrices for bicg benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        m, n: Matrix dimensions

    Returns:
        Tuple of input vectors (A, s, q, p, r)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (m, n)).astype(np_dtype)
    s = np.zeros(n, dtype=np_dtype)
    q = np.zeros(m, dtype=np_dtype)
    p = np.random.randint(-10, 10, n).astype(np_dtype)
    r = np.random.randint(-10, 10, m).astype(np_dtype)

    return A, s, q, p, r


def gen_gemm_inputs(concrete_type, p, r, q, beta=0.1):
    """Generate input matrices for gemm benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        p, r, q: Matrix dimensions
        beta: Scaling factor

    Returns:
        Tuple of input matrices (A, B, C)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (p, r)).astype(np_dtype)

    return A, B, C


def gen_gesummv_inputs(concrete_type, n, alpha=1.0, beta=1.0):
    """Generate input matrices for gesummv benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        n: Matrix dimension
        alpha, beta: Scaling factors

    Returns:
        Tuple of input matrices (A, B, x)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    B = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    x = np.random.randint(-10, 10, n).astype(np_dtype)

    return A, B, x


def gen_mvt_inputs(concrete_type, n):
    """Generate input matrices for mvt benchmark

    Args:
        concrete_type: Data type (float32 or int32)
        n: Matrix dimension

    Returns:
        Tuple of input vectors (A, x1, x2, y1, y2)
    """
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    # Create random matrices with appropriate shapes
    A = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    x1 = np.random.randint(-10, 10, n).astype(np_dtype)
    x2 = np.random.randint(-10, 10, n).astype(np_dtype)
    y1 = np.random.randint(-10, 10, n).astype(np_dtype)
    y2 = np.random.randint(-10, 10, n).astype(np_dtype)

    return A, x1, x2, y1, y2


# Register all benchmarks using the helper function
add_benchmark("three_mm", three_mm, three_mm_np, gen_three_mm_inputs)
add_benchmark("two_mm", two_mm, two_mm_np, gen_two_mm_inputs)
add_benchmark("gemm", gemm, gemm_np, gen_gemm_inputs)
add_benchmark("mvt", mvt, mvt_np, gen_mvt_inputs)
add_benchmark("atax", atax, atax_np, gen_atax_inputs)
add_benchmark("bicg", bicg, bicg_np, gen_bicg_inputs)
add_benchmark("gesummv", gesummv, gesummv_np, gen_gesummv_inputs)
