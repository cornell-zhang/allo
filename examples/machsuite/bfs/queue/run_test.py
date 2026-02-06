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
import bfs_queue as bfs_queue_mod

N_NODES = 256
N_EDGES = 4096
N_LEVELS = 10
MAX_LEVEL = 999999


def bfs_queue_ref(nodes, edges, starting_node):
    """Python reference implementation of BFS (queue-based)."""
    level = [MAX_LEVEL] * N_NODES
    level_counts = [0] * N_LEVELS
    queue = [0] * N_NODES
    front = 0
    rear = 0

    level[starting_node] = 0
    level_counts[0] = 1
    queue[rear] = starting_node
    rear = (rear + 1) % N_NODES

    while front != rear:
        n = queue[front]
        front = (front + 1) % N_NODES
        tmp_begin = nodes[2 * n]
        tmp_end = nodes[2 * n + 1]
        for e in range(tmp_begin, tmp_end):
            tmp_dst = edges[e]
            tmp_level = level[tmp_dst]

            if tmp_level == MAX_LEVEL:
                tmp_level = level[n] + 1
                level[tmp_dst] = tmp_level
                level_counts[tmp_level] += 1
                queue[rear] = tmp_dst
                rear = (rear + 1) % N_NODES

    return level, level_counts


def _patch_bfs_sizes(params):
    """Patch BFS size constants across all relevant modules."""
    global N_NODES, N_EDGES, N_LEVELS, MAX_LEVEL
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

    # Patch allo kernel module
    bfs_queue_mod.N_NODES = N_NODES
    bfs_queue_mod.N_NODES_2 = N_NODES * 2
    bfs_queue_mod.N_EDGES = N_EDGES
    bfs_queue_mod.N_LEVELS = N_LEVELS


def test_bfs_queue(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["bfs"][psize]
    _patch_bfs_sizes(params)

    random.seed(42)

    s = allo.customize(bfs_queue_mod.bfs_queue)
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
    (golden_D, golden_F) = bfs_queue_ref(np_A, np_B, np_C)

    np.testing.assert_allclose(D, golden_D, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(F, golden_F, rtol=1e-5, atol=1e-5)
    print("BFS Queue PASS!")


if __name__ == "__main__":
    test_bfs_queue("full")
