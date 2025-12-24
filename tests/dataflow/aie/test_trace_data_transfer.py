# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from allo.ir.types import int16, Stream
import allo.dataflow as df
import numpy as np
from allo.backend.aie import is_available

Ty = int16
M, N = 32, 32


@df.region()
def top():
    pipe: Stream[Ty[M, N], 2]

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        pipe.put(A)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        B[:, :] = pipe.get()


@pytest.mark.skipif(
    os.environ.get("NPU2") == "1",
    reason="[FIXME]: seems that this test may crash the device",
)
def test_trace_data_transfer():
    if is_available():
        mod = df.build(
            top,
            project="transfer.prj",
            target="aie",
            trace=[("producer", (0,)), ("consumer", (0,))],
            trace_size=32768,
        )
        A = np.random.randint(0, 64, (M, N)).astype(np.int16)
        B = np.zeros((M, N)).astype(np.int16)
        mod(A, B)
        np.testing.assert_allclose(B, A, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_trace_data_transfer()
