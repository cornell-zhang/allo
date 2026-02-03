# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import allo
from allo.ir.types import int16, float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

# Test 1: Simple stream of blocks with 2D blocks
M, N = 4, 4
NUM_BLOCKS = 2


@df.region()
def top_stream_2d_blocks(A: int16[M * NUM_BLOCKS, N], B: int16[M * NUM_BLOCKS, N]):
    # Stream where each element is a 4x4 block of int16
    pipe: Stream[int16[M, N], 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: int16[M * NUM_BLOCKS, N]):
        for i in range(NUM_BLOCKS):
            block: int16[M, N] = 0
            for m in range(M):
                for n in range(N):
                    block[m, n] = local_A[i * M + m, n]
            pipe.put(block)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: int16[M * NUM_BLOCKS, N]):
        for i in range(NUM_BLOCKS):
            block: int16[M, N] = pipe.get()
            for m in range(M):
                for n in range(N):
                    local_B[i * M + m, n] = block[m, n]


def test_2d_blocks():
    A = np.random.randint(0, 100, (M * NUM_BLOCKS, N), dtype=np.int16)
    B = np.zeros((M * NUM_BLOCKS, N), dtype=np.int16)

    # Test with simulator
    sim_mod = df.build(top_stream_2d_blocks, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A, atol=1e-5)
    print("Dataflow Simulator Passed for 2D blocks!")

    # Test with HLS backend
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = df.build(
                top_stream_2d_blocks,
                target="vitis_hls",
                mode="csim",
                project=tmpdir,
            )
            mod(A, B)
            np.testing.assert_allclose(B, A, atol=1e-5)
            print("HLS CSim Passed for 2D blocks!")

    # Test generated HLS code
    mod = df.build(top_stream_2d_blocks, target="vhls")
    code = mod.hls_code
    print(code)
    assert (
        "producer" in code or "Producer" in code
    ), "Producer function should be in HLS code"
    assert (
        "consumer" in code or "Consumer" in code
    ), "Consumer function should be in HLS code"
    assert "hls::vector" in code, "hls::vector should be in HLS code"
    assert "hls::stream" in code, "hls::stream should be in HLS code"
    print("HLS Code Generation Passed for 2D blocks!")


# Test 2: Stream of blocks with computation
@df.region()
def top_stream_block_compute(A: float32[M * 2], B: float32[M * 2]):
    # Stream where each element is a 1D block of float32
    pipe: Stream[float32[M], 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: float32[M * 2]):
        for i in range(2):
            block: float32[M] = 0
            for m in range(M):
                block[m] = local_A[i * M + m]
            pipe.put(block)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: float32[M * 2]):
        for i in range(2):
            block: float32[M] = pipe.get()
            # Apply some computation
            for m in range(M):
                block[m] = block[m] * 2.0 + 1.0
            for m in range(M):
                local_B[i * M + m] = block[m]


def test_blocks_compute():
    A = np.random.rand(M * 2).astype(np.float32)
    B = np.zeros(M * 2, dtype=np.float32)
    expected = (A * 2.0 + 1.0).reshape(2, M).flatten()

    # Test with simulator
    sim_mod = df.build(top_stream_block_compute, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, expected, atol=1e-5)
    print("Dataflow Simulator Passed for blocks with computation!")

    # Test with HLS backend
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = df.build(
                top_stream_block_compute,
                target="vitis_hls",
                mode="csim",
                project=tmpdir,
            )
            B.fill(0)
            mod(A, B)
            np.testing.assert_allclose(B, expected, atol=1e-5)
            print("HLS CSim Passed for blocks with computation!")
    mod = df.build(top_stream_block_compute, target="vhls")
    code = mod.hls_code
    print(code)
    assert (
        "producer" in code or "Producer" in code
    ), "Producer function should be in HLS code"
    assert (
        "consumer" in code or "Consumer" in code
    ), "Consumer function should be in HLS code"
    assert "hls::vector" in code, "hls::vector should be in HLS code"
    assert "hls::stream" in code, "hls::stream should be in HLS code"
    print("HLS Code Generation Passed for blocks with computation!")


# Test 3: Multiple streams of blocks
@df.region()
def top_multiple_stream_blocks(A: int16[M, N], B: int16[M, N], C: int16[M, N]):
    pipe_A: Stream[int16[M, N], 4]
    pipe_B: Stream[int16[M, N], 4]

    @df.kernel(mapping=[1], args=[A, B])
    def producer(local_A: int16[M, N], local_B: int16[M, N]):
        pipe_A.put(local_A)
        pipe_B.put(local_B)

    @df.kernel(mapping=[1], args=[C])
    def consumer(local_C: int16[M, N]):
        block_A: int16[M, N] = pipe_A.get()
        block_B: int16[M, N] = pipe_B.get()
        for m in range(M):
            for n in range(N):
                local_C[m, n] = block_A[m, n] + block_B[m, n]


def test_multiple_blocks():
    A = np.random.randint(0, 50, (M, N), dtype=np.int16)
    B = np.random.randint(0, 50, (M, N), dtype=np.int16)
    C = np.zeros((M, N), dtype=np.int16)
    expected = A + B

    # Test with simulator
    sim_mod = df.build(top_multiple_stream_blocks, target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, expected, atol=1e-5)
    print("Dataflow Simulator Passed for multiple streams of blocks!")

    # Test with HLS backend
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = df.build(
                top_multiple_stream_blocks,
                target="vitis_hls",
                mode="csim",
                project=tmpdir,
            )
            C.fill(0)
            mod(A, B, C)
            np.testing.assert_allclose(C, expected, atol=1e-5)
            print("HLS CSim Passed for multiple streams of blocks!")
    mod = df.build(top_multiple_stream_blocks, target="vhls")
    code = mod.hls_code
    print(code)
    assert (
        "producer" in code or "Producer" in code
    ), "Producer function should be in HLS code"
    assert (
        "consumer" in code or "Consumer" in code
    ), "Consumer function should be in HLS code"
    assert "hls::vector" in code, "hls::vector should be in HLS code"
    assert "hls::stream" in code, "hls::stream should be in HLS code"
    print("HLS Code Generation Passed for multiple streams of blocks!")


if __name__ == "__main__":
    print("Testing stream of blocks...")
    print("=" * 50)
    test_2d_blocks()
    print("=" * 50)
    test_blocks_compute()
    print("=" * 50)
    test_multiple_blocks()
    print("=" * 50)
    print("All tests passed!")
