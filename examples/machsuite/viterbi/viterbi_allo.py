import allo
from allo.ir.types import float32, int32
import numpy as np
from read import read_viterbi_input 
from write import write_output_data

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

import os
from viterbi import viterbi as viterbi_ref

s = allo.customize(viterbi)
mod = s.build()

data_dir = os.path.dirname(os.path.abspath(__file__))
inputfile = os.path.join(data_dir, 'input.data')
init, transition, emission, obs = read_viterbi_input(inputfile)

init = np.array(init, dtype=np.float32)
transition = np.array(transition, dtype=np.float32)
emission = np.array(emission, dtype=np.float32)
obs = np.array(obs, dtype=np.int32)

path = mod(obs, init, transition, emission)

# Run Python reference for comparison
ref_path = viterbi_ref(obs, init, transition, emission)

np.testing.assert_array_equal(path, ref_path)
print("PASS!")
