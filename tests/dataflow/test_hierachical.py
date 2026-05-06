# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from allo.ir.types import float32, int32, ConstExpr, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

M, N, K = 32, 32, 32


@df.region()
def inner[P0, P1](A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        pi, pj = df.get_pid()
        Mt: ConstExpr[int32] = M // P0
        Nt: ConstExpr[int32] = N // P1
        for i in range(pi * Mt, (pi + 1) * Mt):
            for j in range(pj * Nt, (pj + 1) * Nt):
                for k in range(K):
                    local_C[i, j] += local_A[i, k] * local_B[k, j]


@df.region()
def top(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
    @df.kernel(mapping=[2], args=[A, B, C1, C2])
    def wrapper(
        local_A: float32[M, K],
        local_B: float32[K, N],
        local_C1: float32[M, N],
        local_C2: float32[M, N],
    ):
        i = df.get_pid()
        with allo.meta_if(i == 0):
            inner[2, 2](local_A, local_B, local_C1)
        with allo.meta_if(i == 1):
            inner[4, 4](local_A, local_B, local_C2)


def test_hierachical_function():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C1 = np.zeros((M, N), dtype=np.float32)
    C2 = np.zeros((M, N), dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C1, C2)
    np.testing.assert_allclose(C1, np.dot(A, B), rtol=1e-5)
    np.testing.assert_allclose(C2, np.dot(A, B), rtol=1e-5)
    print("Dataflow Simulator Passed!")

    mod = df.build(top)
    print(mod.module)
    assert "scf.for" not in str(mod.module), "SCF ops are not expected in the module"
    if hls.is_available("vitis_hls"):
        C1 = np.zeros((M, N), dtype=np.float32)
        C2 = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C1, C2)
        np.testing.assert_allclose(C1, np.dot(A, B), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(C2, np.dot(A, B), rtol=1e-5, atol=1e-5)
        print("Success!")


# ---------------------------------------------------------------------------
# Regression: recursive OMP injection for hierarchical streaming kernels
#
# The simulator's build_dataflow_simulator originally only wrapped the
# top-level function's PE calls in omp.parallel sections. Inner-region
# kernel calls (producer/consumer communicating via a stream) ran
# sequentially: producer filled the stream buffer and blocked forever
# because consumer hadn't started. The fix extracts _inject_omp_parallel_sections
# and applies it recursively to every function that has PE kernel calls.
# ---------------------------------------------------------------------------

_N = 4
_CAP = 2  # stream capacity < _N → sequential execution deadlocks


@df.region()
def _inner_streaming(result: int32[1]):
    s: Stream[int32, _CAP]

    @df.kernel(mapping=[1])
    def producer():
        for i in range(_N):
            v: int32 = i  # loop var is index type; annotate to get i32 for stream put
            s.put(v)

    @df.kernel(mapping=[1], args=[result])
    def consumer(out: int32[1]):
        acc: int32 = 0
        for i in range(_N):
            acc += s.get()
        out[0] = acc


@df.region()
def _outer_streaming(result: int32[1]):
    @df.kernel(mapping=[1], args=[result])
    def driver(out: int32[1]):
        _inner_streaming(out)


def test_hierarchical_omp_deadlock_fix():
    result = np.zeros(1, dtype=np.int32)
    sim_mod = df.build(_outer_streaming, target="simulator")
    sim_mod(result)
    assert result[0] == sum(range(_N)), f"Expected {sum(range(_N))}, got {result[0]}"


# ---------------------------------------------------------------------------
# Regression: scalar in args=[...] must be rejected at type-inference time
# ---------------------------------------------------------------------------


def test_scalar_in_kernel_args_raises():
    # Allo's customize() converts all type-inference errors to sys.exit(1).
    # Verify the build fails; the error message is printed to stdout.
    with pytest.raises(SystemExit):

        @df.region()
        def _bad(n: int32, out: int32[1]):
            @df.kernel(mapping=[1], args=[n, out])
            def kernel(scalar: int32, o: int32[1]):
                o[0] = scalar

        df.build(_bad, target="simulator")


if __name__ == "__main__":
    test_hierachical_function()
    test_hierarchical_omp_deadlock_fix()
    test_scalar_in_kernel_args_raises()
