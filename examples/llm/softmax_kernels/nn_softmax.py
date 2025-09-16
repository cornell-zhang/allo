import allo
from allo.ir.types import float32, int32
from allo.customize import Partition
import allo
import allo.backend.hls as hls
import allo.dsl as dsl

Ty = float32
L = 64


def softmax(X: "Ty[L, L]") -> "Ty[L, L]":
    Z: Ty[L, L]
    E: Ty[L, L]
    M: Ty[L] = -1000000000000.0
    S: Ty[L] = 0.0

    for i, j in dsl.grid(L, L, name="row_max"):
        if X[i, j] > M[i]:
            M[i] = X[i, j]

    # compute exp and sum
    for i, j in dsl.grid(L, L, name="exp_sum"):
        E[i, j] = dsl.exp(X[i, j] - M[i])
        S[i] += E[i, j]

    for i, j in dsl.grid(L, L, name="update"):
        Z[i, j] = E[i, j] / S[i]

    return Z


def schedule_softmax(s):
    lj = s.get_loops(s.top_func_name)["exp_sum"]["j"]
    s.pipeline(lj)
    lj = s.get_loops(s.top_func_name)["update"]["j"]
    s.pipeline(lj)
    return s

if __name__ == "__main__":
    s = allo.customize(softmax)
    schedule_softmax(s)
    #print(s.module)
    mod = s.build(target="vitis_hls", mode="csyn", project="nn_softmax.prj")()
    #print(mod)