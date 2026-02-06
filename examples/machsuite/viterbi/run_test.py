# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import viterbi_allo


def viterbi_ref(obs, init, transition, emission):
    """Python/NumPy reference implementation of Viterbi algorithm."""
    N_OBS = len(obs)
    N_STATES = len(init)

    llike = np.zeros((N_OBS, N_STATES))
    path = np.zeros(N_OBS, dtype=int)

    for s in range(N_STATES):
        llike[0][s] = init[s] + emission[s, obs[0]]

    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            min_p = llike[t - 1][0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p = llike[t - 1][prev] + transition[prev, curr] + emission[curr, obs[t]]
                if p < min_p:
                    min_p = p
            llike[t][curr] = min_p

    min_s = 0
    min_p = llike[N_OBS - 1][0]
    for s in range(1, N_STATES):
        p = llike[N_OBS - 1][s]
        if p < min_p:
            min_p = p
            min_s = s
    path[N_OBS - 1] = min_s

    for t in range(N_OBS - 2, -1, -1):
        min_s = 0
        min_p = llike[t][0] + transition[0, path[t + 1]]
        for s in range(1, N_STATES):
            p = llike[t][s] + transition[s, path[t + 1]]
            if p < min_p:
                min_p = p
                min_s = s
        path[t] = min_s

    return path


def test_viterbi(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["viterbi"][psize]

    N_OBS = params["N_OBS"]
    N_STATES = params["N_STATES"]
    N_TOKENS = params["N_TOKENS"]

    # Patch module constants
    viterbi_allo.N_OBS = N_OBS
    viterbi_allo.N_STATES = N_STATES
    viterbi_allo.N_TOKENS = N_TOKENS

    np.random.seed(42)

    # Generate random HMM parameters in -log probability space
    init = np.random.rand(N_STATES).astype(np.float32) * 10.0
    transition = np.random.rand(N_STATES, N_STATES).astype(np.float32) * 10.0
    emission = np.random.rand(N_STATES, N_TOKENS).astype(np.float32) * 10.0
    obs = np.random.randint(0, N_TOKENS, size=N_OBS).astype(np.int32)

    s = allo.customize(viterbi_allo.viterbi)
    mod = s.build()

    path = mod(obs, init, transition, emission)

    # Run Python reference for comparison
    ref_path = viterbi_ref(obs, init, transition, emission)

    np.testing.assert_array_equal(path, ref_path)
    print("PASS!")


if __name__ == "__main__":
    test_viterbi("full")
