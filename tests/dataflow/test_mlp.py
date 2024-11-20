# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
BS = 2
M0, M1, M2 = 256, 128, 64
NUM_CLASSES = 2

np_W0 = np.random.rand(M0, M1).astype(np.float32)
np_W1 = np.random.rand(M1, M2).astype(np.float32)
np_W2 = np.random.rand(M2, NUM_CLASSES).astype(np.float32)


@df.region()
def top():
    Z0 = df.pipe(dtype=Ty, shape=(), depth=4)
    Z1 = df.pipe(dtype=Ty, shape=(), depth=4)

    @df.kernel(mapping=[1])
    def linear1(X: Ty[BS, M0]):
        # BS*M0 * M0*M1 = BS*M1
        W0: Ty[M0, M1] = np_W0
        for i in range(BS):
            buf: Ty[M1] = 0
            for k in range(M0):
                # reorder reduction loop outside, and pipeline
                x: Ty = X[i, k]
                for j in range(M1):
                    buf[j] += x * W0[k, j]
            for j_back in range(M1):
                # relu
                Z0.put(max(buf[j_back], 0))

    @df.kernel(mapping=[1])
    def linear2():
        # BS*M1 * M1*M2 = BS*M2
        W1: Ty[M1, M2] = np_W1
        for i in range(BS):
            buf: Ty[M2] = 0
            for k in range(M1):
                # reorder reduction loop outside, and pipeline
                x: Ty = Z0.get()
                for j in range(M2):
                    buf[j] += x * W1[k, j]
            for j_back in range(M2):
                Z1.put(max(buf[j_back], 0))

    @df.kernel(mapping=[1])
    def linear3(Z2: Ty[BS, NUM_CLASSES]):
        # BS*M2 * M2*NUM_CLASSES = BS*NUM_CLASSES
        W2: Ty[M2, NUM_CLASSES] = np_W2
        for i in range(BS):
            buf: Ty[NUM_CLASSES] = 0
            for k in range(M2):
                # reorder reduction loop outside, and pipeline
                x: Ty = Z1.get()
                for j in range(NUM_CLASSES):
                    buf[j] += x * W2[k, j]
            for j_back in range(NUM_CLASSES):
                Z2[i, j_back] = max(buf[j_back], 0)


def test_mlp():
    X = np.random.rand(BS, M0).astype(np.float32)
    Y = np.maximum(
        np.dot(np.maximum(np.dot(np.maximum(np.dot(X, np_W0), 0), np_W1), 0), np_W2), 0
    )
    mod = df.build(top)
    allo_final_Y = np.zeros((BS, NUM_CLASSES), dtype=np.float32)
    mod(X, allo_final_Y)
    np.testing.assert_allclose(Y, allo_final_Y, rtol=1e-5)
    print("PASSED!")
    # hls
    if hls.is_available("vitis_hls"):
        s = df.customize(top)
        s.pipeline("linear1_0:j")
        s.pipeline("linear2_0:j")
        s.pipeline("linear3_0:j")
        print(s.module)
        mod = s.build(target="vitis_hls", mode="hw", project="df-mlp3-relu-on-chip.prj")
        mod(X, allo_final_Y)
        np.testing.assert_allclose(Y, allo_final_Y, rtol=1e-5)


if __name__ == "__main__":
    test_mlp()
