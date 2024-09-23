# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M, N = 16, 16
P0 = 2
Mt = M // P0


@df.kernel(mapping=[P0])
def gemv0(A: Ty[M, N], x: Ty[N]):
    pi = df.get_pid()
    pipe: Stream[Ty[Mt]] = df.pipe(src="gemv0", dst="gemv1")
    y_out: Ty[Mt] = 0
    for m in range(pi * Mt, (pi + 1) * Mt):
        y_acc: Ty = 0
        for n in range(N // 2):
            y_acc += A[m, n] * x[n]
        y_out[m - pi * Mt] = y_acc
    pipe.put(y_out)


@df.kernel(mapping=[P0])
def gemv1(A: Ty[M, N], x: Ty[N], y: Ty[M]):
    pi = df.get_pid()
    pipe: Stream[Ty[Mt]] = df.pipe(src="gemv0", dst="gemv1")
    y_out: Ty[Mt] = 0
    for m in range(pi * Mt, (pi + 1) * Mt):
        y_acc: Ty = 0
        for n in range(N // 2, N):
            y_acc += A[m, n] * x[n]
        y_out[m - pi * Mt] = y_acc
    y_prev: Ty[Mt] = pipe.get()
    for m in range(pi * Mt, (pi + 1) * Mt):
        y[m] = y_out[m - pi * Mt] + y_prev[m - pi * Mt]


def test_cooperative_gemv():
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(
        N,
    ).astype(np.float32)
    C = np.zeros((M,), dtype=np.float32)
    top = df.build([gemv0, gemv1])
    if hls.is_available("vitis_hls"):
        top(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_cooperative_gemv()
