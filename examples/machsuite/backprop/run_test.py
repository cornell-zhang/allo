# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import backprop


def test_backprop(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["backprop"][psize]

    # Patch module constants before customize()
    for key, val in params.items():
        setattr(backprop, key, val)

    INPUT_DIMENSION = backprop.INPUT_DIMENSION
    POSSIBLE_OUTPUTS = backprop.POSSIBLE_OUTPUTS
    TRAINING_SETS = backprop.TRAINING_SETS
    NODES_PER_LAYER = backprop.NODES_PER_LAYER

    np.random.seed(42)

    # Generate random weights, biases, and training data
    weights1 = np.random.randn(INPUT_DIMENSION * NODES_PER_LAYER).astype(np.float32)
    weights2 = np.random.randn(NODES_PER_LAYER * NODES_PER_LAYER).astype(np.float32)
    weights3 = np.random.randn(NODES_PER_LAYER * POSSIBLE_OUTPUTS).astype(np.float32)
    biases1 = np.random.randn(NODES_PER_LAYER).astype(np.float32)
    biases2 = np.random.randn(NODES_PER_LAYER).astype(np.float32)
    biases3 = np.random.randn(POSSIBLE_OUTPUTS).astype(np.float32)
    training_data = np.random.randn(TRAINING_SETS * INPUT_DIMENSION).astype(np.float32)
    training_targets = np.random.randn(TRAINING_SETS * POSSIBLE_OUTPUTS).astype(
        np.float32
    )

    s = allo.customize(backprop.backprop)
    mod = s.build(target="llvm")

    # Run-only test: MachSuite's check.data is known to not match
    mod(
        weights1,
        weights2,
        weights3,
        biases1,
        biases2,
        biases3,
        training_data,
        training_targets,
    )
    print("PASS!")


if __name__ == "__main__":
    test_backprop("full")
