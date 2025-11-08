# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from allo.ir.types import float32, Stream
from allo.ir.utils import MockBuffer
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


Ty = float32
BS = 2
M0, M1, M2 = 256, 128, 64
NUM_CLASSES = 2

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NeuralNetwork, self).__init__()
#         self.layer1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(128, 64)
#         self.output_layer = nn.Linear(64, num_classes)

#     def forward(self, x):
#         out = self.relu(self.layer1(x))
#         out = self.relu(self.layer2(out))
#         out = self.output_layer(out)  # No softmax here as it's included in nn.CrossEntropyLoss
#         return out

if os.path.exists("np_W0.txt"):
    np_W0 = np.loadtxt("np_W0.txt", dtype=np.float32)
    np_W1 = np.loadtxt("np_W1.txt", dtype=np.float32)
    np_W2 = np.loadtxt("np_W2.txt", dtype=np.float32)
else:
    np_W0 = np.random.rand(M0, M1).astype(np.float32)
    np_W1 = np.random.rand(M1, M2).astype(np.float32)
    np_W2 = np.random.rand(M2, NUM_CLASSES).astype(np.float32)
    np.savetxt("np_W0.txt", np_W0, fmt="%f")
    np.savetxt("np_W1.txt", np_W1, fmt="%f")
    np.savetxt("np_W2.txt", np_W2, fmt="%f")


@df.region()
def top():
    Z0: Stream[Ty, BS * M1]
    Z1: Stream[Ty, BS * M2]

    @df.kernel(mapping=[1])
    def linear1(X: Ty[BS, M0]):
        # BS*M0 * M0*M1 = BS*M1
        W0: Ty[M0, M1] = np_W0
        buf: Ty[M1]
        for i in range(BS):
            for j_init in range(M1):
                buf[j_init] = 0
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
        buf: Ty[M2]
        for i in range(BS):
            for j_init in range(M2):
                buf[j_init] = 0
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
        buf: Ty[NUM_CLASSES]
        for i in range(BS):
            for j_init in range(NUM_CLASSES):
                buf[j_init] = 0
            for k in range(M2):
                # reorder reduction loop outside, and pipeline
                x: Ty = Z1.get()
                for j in range(NUM_CLASSES):
                    buf[j] += x * W2[k, j]
            for j_back in range(NUM_CLASSES):
                Z2[i, j_back] = max(buf[j_back], 0)


def schedule_linear(s, lid, factor=4):
    s.pipeline(f"linear{lid}_0:j")
    s.pipeline(f"linear{lid}_0:j_init")
    s.pipeline(f"linear{lid}_0:j_back")
    s.unroll(f"linear{lid}_0:j", factor=factor)
    s.unroll(f"linear{lid}_0:j_init", factor=factor)
    # s.unroll(f"linear{lid}_0:j_back", factor=factor)
    s.partition(
        MockBuffer(f"linear{lid}_0", "buf"), dim=0, partition_type=2, factor=factor
    )


def test_mlp():
    X = np.random.rand(BS, M0).astype(np.float32)
    Y = np.maximum(
        np.dot(np.maximum(np.dot(np.maximum(np.dot(X, np_W0), 0), np_W1), 0), np_W2), 0
    )
    s = df.customize(top)
    schedule_linear(s, 1, factor=8)
    schedule_linear(s, 2, factor=8)
    schedule_linear(s, 3, factor=1)
    print(s.module)

    sim_final_Y = np.zeros((BS, NUM_CLASSES), dtype=np.float32)
    sim_mod = df.build(top, target="simulator")
    sim_mod(X, sim_final_Y)
    np.testing.assert_allclose(Y, sim_final_Y, rtol=1e-5)
    print("Dataflow Simulator Passed!")

    if hls.is_available("vitis_hls"):
        allo_final_Y = np.zeros((BS, NUM_CLASSES), dtype=np.float32)
        mod = s.build(target="vitis_hls", mode="csim", project="top.prj")
        mod(X, allo_final_Y)
        np.testing.assert_allclose(Y, allo_final_Y, rtol=1e-5)
        print("PASSED CSIM!")
        # hls
        mod = s.build(
            target="vitis_hls", mode="hw", project="df-mlp3-relu-unroll-new.prj"
        )
        mod(X, allo_final_Y)
        np.testing.assert_allclose(Y, allo_final_Y, rtol=1e-5)
        print("PASSED HW!")


if __name__ == "__main__":
    # we need to set OMP_NUM_THREADS to a large number here for simulator
    os.environ["OMP_NUM_THREADS"] = "128"
    test_mlp()
    del os.environ["OMP_NUM_THREADS"]
