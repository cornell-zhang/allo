# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import random
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_dir)
sys.path.insert(0, _dir)
sys.path.insert(0, _parent)
import generate
import bfs_queue_python
import bfs_queue as bfs_queue_mod


def _patch_bfs_sizes(params):
    """Patch BFS size constants across all relevant modules."""
    N_NODES = params["N_NODES"]
    N_EDGES = params["N_EDGES"]
    N_LEVELS = params["N_LEVELS"]

    # Compute SCALE from N_NODES (log2)
    scale = 0
    n = N_NODES
    while n > 1:
        n >>= 1
        scale += 1

    # Patch generate.py
    generate.N_NODES = N_NODES
    generate.N_EDGES = N_EDGES
    generate.SCALE = scale

    # Patch python reference module
    bfs_queue_python.N_NODES = N_NODES
    bfs_queue_python.N_EDGES = N_EDGES
    bfs_queue_python.N_LEVELS = N_LEVELS

    # Patch allo kernel module
    bfs_queue_mod.N_NODES = N_NODES
    bfs_queue_mod.N_NODES_2 = N_NODES * 2
    bfs_queue_mod.N_EDGES = N_EDGES
    bfs_queue_mod.N_LEVELS = N_LEVELS


def _generate_and_run(mod_func, ref_func, params):
    """Build, generate graph, run kernel and reference, compare."""
    random.seed(42)

    s = allo.customize(mod_func)
    mod = s.build(target="llvm")

    generated_data = generate.generate_random_graph()

    nodes_list = []
    for node in generated_data["nodes"]:
        nodes_list.append(node.edge_begin)
        nodes_list.append(node.edge_end)
    edges_list = [edge.dst for edge in generated_data["edges"]]

    np_A = np.array(nodes_list, np.int32)
    np_B = np.array(edges_list, np.int32)
    np_C = generated_data["starting_node"]

    (D, F) = mod(np_A, np_B, np_C)
    (golden_D, golden_F) = ref_func(np_A, np_B, np_C)

    np.testing.assert_allclose(D, golden_D, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(F, golden_F, rtol=1e-5, atol=1e-5)


def test_bfs_queue(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["bfs"][psize]
    _patch_bfs_sizes(params)
    _generate_and_run(bfs_queue_mod.bfs_queue, bfs_queue_python.bfs_queue_test, params)
    print("BFS Queue PASS!")


if __name__ == "__main__":
    test_bfs_queue("full")
