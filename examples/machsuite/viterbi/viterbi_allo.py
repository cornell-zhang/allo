import os
import sys
import allo
from allo.ir.types import float32, int32
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from viterbi import viterbi as viterbi_ref

N_OBS:int32 = 140
N_STATES:int32 = 64
N_TOKENS:int32 = 64

def viterbi(obs: int32[N_OBS], init: float32[N_STATES], transition: float32[N_STATES, N_STATES], emission: float32[N_STATES, N_TOKENS]) -> int32[N_OBS]:

    llike:float32[N_OBS, N_STATES]

    for s in range(N_STATES):
        llike[0, s] = init[s] + emission[s, obs[0]]

    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            min_p:float32 = llike[t-1, 0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p:float32 = llike[t-1, prev] + transition[prev, curr] + emission[curr, obs[t]]
                if p < min_p:
                    min_p = p
            llike[t, curr] = min_p

    min_s:int32 = 0
    min_p:float32 = llike[N_OBS-1, 0]
    for s in range(1, N_STATES):
        p:float32 = llike[N_OBS-1, s]
        if p < min_p:
            min_p = p
            min_s = s

    path:int32[N_OBS]
    path[N_OBS-1] = min_s

    for t in range(N_OBS-1):
        actual_t:int32 = N_OBS - 2 - t
        min_s:int32 = 0
        min_p:float32 = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
        for s in range(1, N_STATES):
            p:float32 = llike[actual_t, s] + transition[s, path[actual_t + 1]]
            if p < min_p:
                min_p = p
                min_s = s
        path[actual_t] = min_s

    return path

if __name__ == "__main__":
    np.random.seed(42)

    # Generate random HMM parameters in -log probability space
    init = np.random.rand(N_STATES).astype(np.float32) * 10.0
    transition = np.random.rand(N_STATES, N_STATES).astype(np.float32) * 10.0
    emission = np.random.rand(N_STATES, N_TOKENS).astype(np.float32) * 10.0
    obs = np.random.randint(0, N_TOKENS, size=N_OBS).astype(np.int32)

    s = allo.customize(viterbi)
    mod = s.build()

    path = mod(obs, init, transition, emission)

    # Run Python reference for comparison
    ref_path = viterbi_ref(obs, init, transition, emission)

    np.testing.assert_array_equal(path, ref_path)
    print("PASS!")
