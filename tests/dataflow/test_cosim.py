# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Test for Vitis HLS C/RTL co-simulation (cosim) mode.
# Related to: https://github.com/cornell-zhang/allo/issues/572

import tempfile

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

# ---------------------------------------------------------------------------
# Producer-consumer dataflow with a single stream channel.
# Each stream is written by exactly one producer and read by exactly one
# consumer, which satisfies the Vitis HLS dataflow constraint.
# ---------------------------------------------------------------------------
N = 8


@df.region()
def top(A: float32[N], B: float32[N], C: float32[N]):
    pipe: Stream[float32, 16]

    @df.kernel(mapping=[1], args=[A, B])
    def producer(local_A: float32[N], local_B: float32[N]):
        for i in range(N):
            val: float32 = local_A[i] + local_B[i]
            pipe.put(val)

    @df.kernel(mapping=[1], args=[C])
    def consumer(local_C: float32[N]):
        for i in range(N):
            local_C[i] = pipe.get()


def test_producer_consumer_cosim():
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    C_golden = A + B

    # Verify simulator
    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, C_golden, atol=1e-5)
    print("Simulator Passed!")

    if hls.is_available("vitis_hls"):
        # Test csyn to confirm synthesis works
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = df.build(
                top,
                target="vitis_hls",
                mode="csyn",
                project=tmpdir,
                wrap_io=False,
            )
            mod()
            print("C Synthesis Passed!")

        # Test cosim (C synthesis + C/RTL co-simulation)
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = df.build(
                top,
                target="vitis_hls",
                mode="cosim",
                project=tmpdir,
                wrap_io=False,
            )
            C = np.zeros(N, dtype=np.float32)
            mod(A, B, C)
            np.testing.assert_allclose(C, C_golden, atol=1e-5)
            print("Cosim Passed!")


if __name__ == "__main__":
    test_producer_consumer_cosim()
