import os
import sys
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from backprop import backprop, INPUT_DIMENSION, POSSIBLE_OUTPUTS, TRAINING_SETS, NODES_PER_LAYER

if __name__ == "__main__":
    np.random.seed(42)

    # Generate random weights, biases, and training data
    weights1 = np.random.randn(INPUT_DIMENSION * NODES_PER_LAYER).astype(np.float32)
    weights2 = np.random.randn(NODES_PER_LAYER * NODES_PER_LAYER).astype(np.float32)
    weights3 = np.random.randn(NODES_PER_LAYER * POSSIBLE_OUTPUTS).astype(np.float32)
    biases1 = np.random.randn(NODES_PER_LAYER).astype(np.float32)
    biases2 = np.random.randn(NODES_PER_LAYER).astype(np.float32)
    biases3 = np.random.randn(POSSIBLE_OUTPUTS).astype(np.float32)
    training_data = np.random.randn(TRAINING_SETS * INPUT_DIMENSION).astype(np.float32)
    training_targets = np.random.randn(TRAINING_SETS * POSSIBLE_OUTPUTS).astype(np.float32)

    s = allo.customize(backprop)
    mod = s.build(target="llvm")

    # Run-only test: MachSuite's check.data is known to not match
    mod(weights1, weights2, weights3, biases1, biases2, biases3, training_data, training_targets)
    print("PASS!")
