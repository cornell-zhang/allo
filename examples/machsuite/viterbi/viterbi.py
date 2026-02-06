# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def viterbi(obs, init, transition, emission):
    N_OBS = len(obs)  # Number of observations
    N_STATES = len(init)  # Number of states
    N_TOKENS = emission.shape[1]  # Number of tokens

    # Initialize log-likelihood matrix
    llike = np.zeros((N_OBS, N_STATES))
    path = np.zeros(N_OBS, dtype=int)

    # Initialize with the first observation and initial probabilities
    for s in range(N_STATES):
        llike[0][s] = init[s] + emission[s, obs[0]]

    # Iteratively compute the probabilities over time
    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            min_p = llike[t - 1][0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p = llike[t - 1][prev] + transition[prev, curr] + emission[curr, obs[t]]
                if p < min_p:
                    min_p = p
            llike[t][curr] = min_p

    # Identify end state
    min_s = 0
    min_p = llike[N_OBS - 1][0]
    for s in range(1, N_STATES):
        p = llike[N_OBS - 1][s]
        if p < min_p:
            min_p = p
            min_s = s
    path[N_OBS - 1] = min_s

    # Backtrack to recover full path
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


if __name__ == "__main__":
    from read import read_viterbi_input

    inputfile = "input.data"
    init, transition, emission, obs = read_viterbi_input(inputfile)

    # Taking -log of probabilities for -log space
    # init = -np.log(init)
    # transition = -np.log(transition)
    # emission = -np.log(emission)

    path = viterbi(obs, init, transition, emission)
    print(path)

    # output_file = 'check.data'
    # write_output_data(output_file, path)
