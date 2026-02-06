# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

N_NODES = 256
N_EDGES = 4096
N_LEVELS = 10
MAX_LEVEL = 999999


def bfs_queue_test(nodes, edges, starting_node):
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


#     0
#    / \
#   1   2
#  / \ / \
# 3   4   5
#  \ / \ / \
#   6   7   8

# Define nodes and edges for the example graph
# nodes = [ 0,2,2,4,4,6,6,7,7,9,9,11,11,11,11,11,11,11]
# edges = [ 1,2,3,4,4,5,6,6,7,7,8]


# Run BFS starting from node 0
# level, level_counts = bfs_queue_test(nodes, edges, 0)

# Print the results
# print("Node Levels:", level)
# print("Level Counts:", level_counts)
