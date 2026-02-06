import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import viterbi_allo
from viterbi import viterbi as viterbi_ref


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
