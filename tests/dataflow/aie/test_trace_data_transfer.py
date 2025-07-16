# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import int16
import allo.dataflow as df
import numpy as np

Ty = int16
M, N = 32, 32


@df.region()
def top():
    pipe = df.pipe(dtype=Ty, shape=(M, N), depth=2)

    @df.kernel(mapping=[1])
    def producer(A: Ty[M, N]):
        pipe.put(A)

    @df.kernel(mapping=[1])
    def consumer(B: Ty[M, N]):
        B[:, :] = pipe.get()


def trace_data_transfer():
    mod = df.build(
        top,
        target="aie-mlir",
        use_default_codegen=True,
        trace=[("producer", (0,)), ("consumer", (0,))],
        trace_size=65536,
    )
    A = np.random.randint(0, 64, (M, N)).astype(np.int16)
    B = np.zeros((M, N)).astype(np.int16)
    mod(A, B)
    np.testing.assert_allclose(B, A, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    trace_data_transfer()
