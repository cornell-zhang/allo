# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import allo
from allo.ir.types import int32
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from generate import generate_random_graph
from bfs_bulk_python import bfs_bulk_test

N_NODES: int32 = 256
N_NODES_2: int32 = 512
N_EDGES: int32 = 4096
N_LEVELS: int32 = 10
MAX_LEVEL: int32 = 999999


def bfs_bulk(
    nodes: int32[N_NODES_2], edges: int32[N_EDGES], starting_node: int32
) -> (int32[N_NODES], int32[N_LEVELS]):
    level: int32[N_NODES] = MAX_LEVEL
    level_counts: int32[N_LEVELS] = 0
    level[starting_node] = 0
    level_counts[0] = 1

    for horizon in range(N_LEVELS):
        cnt: int32 = 0
        for n in range(N_NODES):
            if level[n] == horizon:
                tmp_begin: int32 = nodes[2 * n]
                tmp_end: int32 = nodes[2 * n + 1]
                for e in range(tmp_begin, tmp_end):
                    tmp_dst: int32 = edges[e]
                    tmp_level: int32 = level[tmp_dst]

                    if tmp_level == MAX_LEVEL:
                        level[tmp_dst] = horizon + 1
                        cnt += 1

        if cnt != 0:
            level_counts[horizon + 1] = cnt

    return level, level_counts


if __name__ == "__main__":
    import random

    random.seed(42)

    s = allo.customize(bfs_bulk)
    mod = s.build(target="llvm")

    # Generate graph programmatically
    generated_data = generate_random_graph()

    # Convert to numpy arrays matching the data file format
    nodes_list = []
    for node in generated_data["nodes"]:
        nodes_list.append(node.edge_begin)
        nodes_list.append(node.edge_end)
    edges_list = [edge.dst for edge in generated_data["edges"]]

    np_A = np.array(nodes_list, np.int32)
    np_B = np.array(edges_list, np.int32)
    np_C = generated_data["starting_node"]

    (D, F) = mod(np_A, np_B, np_C)

    (golden_D, golden_F) = bfs_bulk_test(np_A, np_B, np_C)

    np.testing.assert_allclose(D, golden_D, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(F, golden_F, rtol=1e-5, atol=1e-5)
    print("PASS!")
