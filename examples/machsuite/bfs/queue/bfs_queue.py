# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32

N_NODES: int32 = 256
N_NODES_2: int32 = 512
N_EDGES: int32 = 4096
N_LEVELS: int32 = 10
MAX_LEVEL: int32 = 999999


def bfs_queue(
    nodes: int32[N_NODES_2], edges: int32[N_EDGES], starting_node: int32
) -> (int32[N_NODES], int32[N_LEVELS]):
    level: int32[N_NODES] = MAX_LEVEL
    level_counts: int32[N_LEVELS] = 0
    queue: int32[N_NODES] = 0
    front: int32 = 0
    rear: int32 = 0

    level[starting_node] = 0
    level_counts[0] = 1
    queue[rear] = starting_node
    rear = (rear + 1) % N_NODES

    while front != rear:
        n: int32 = queue[front]
        front = (front + 1) % N_NODES
        tmp_begin: int32 = nodes[2 * n]
        tmp_end: int32 = nodes[2 * n + 1]
        for e in range(tmp_begin, tmp_end):
            tmp_dst: int32 = edges[e]
            tmp_level: int32 = level[tmp_dst]

            if tmp_level == MAX_LEVEL:
                tmp_level = level[n] + 1
                level[tmp_dst] = tmp_level
                level_counts[tmp_level] += 1
                queue[rear] = tmp_dst
                rear = (rear + 1) % N_NODES

    return level, level_counts
