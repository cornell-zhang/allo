import allo
from allo.ir.types import int32
import numpy as np
from support import read_data_from_file, write_data_to_file, write_data_to_file_2
from generate import generate_random_graph
from bfs_queue_python import bfs_queue_test

N_NODES:int32 = 256
N_NODES_2:int32 = 512
N_EDGES:int32 = 4096
N_LEVELS:int32 = 10
MAX_LEVEL:int32 = 999999



def bfs_queue(nodes: int32[N_NODES_2], edges: int32[N_EDGES], starting_node: int32) -> (int32[N_NODES], int32[N_LEVELS]):
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

s = allo.customize(bfs_queue)
mod = s.build(target="llvm")

#Prepare Input Data
# generated_data = generate_random_graph()
# write_data_to_file(generated_data)
input_data = read_data_from_file("input.data")

np_A = np.array(input_data['nodes'],np.int32)
np_B = np.array(input_data['edges'],np.int32)
np_C = input_data['starting_node'][0]

(D, F)= mod(np_A, np_B, np_C)

#Write output to a file
write_data_to_file_2(F,"check2.data")

(golden_D, golden_F) = bfs_queue_test(np_A, np_B, np_C)


print(F)
print(golden_F)

np.testing.assert_allclose(D, golden_D, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(F, golden_F, rtol=1e-5, atol=1e-5)
