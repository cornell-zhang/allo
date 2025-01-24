import random
from support import write_data_to_file
# Constants
A = 57
B = 19
C = 19
D = 5


N_NODES = 256  # Adjust the size of the graph
N_EDGES = 4096
SCALE = 8
MAX_WEIGHT = 10
MIN_WEIGHT = 1

class Node:
    def __init__(self):
        self.edge_begin = 0
        self.edge_end = 0

class Edge:
    def __init__(self, dst):
        self.dst = dst


def generate_random_graph():
    adjmat = [[0] * N_NODES for _ in range(N_NODES)]

    e = 0
    while e < (N_EDGES // 2):
        r, c = 0, 0
        for scale in range(SCALE, 0, -1):
            rint = random.randint(0, 99)
            if rint >= (A + B):  
                r += (1 << (scale - 1))
            if ((rint >= A) and (rint < (A + B))) or (rint >= (A + B + C)):
                c += (1 << (scale - 1))

        if (adjmat[r][c] == 0) and (r != c):
            adjmat[r][c] = 1
            adjmat[c][r] = 1
            e += 1


    # Shuffle matrix
    for s in range(N_NODES):
        rint = random.randint(0, N_NODES - 1)
        # Swap row s with row rint
        adjmat[s], adjmat[rint] = adjmat[rint], adjmat[s]
        # Swap col s with col rint
        for r in range(N_NODES):
            adjmat[r][s], adjmat[r][rint] = adjmat[r][rint], adjmat[r][s]


    data = {'nodes': [Node() for _ in range(N_NODES)], 'edges': []}
    # Scan rows for edge list lengths, and fill edges while we're at it
    e = 0
    for r in range(N_NODES):
        data['nodes'][r].edge_begin = 0
        data['nodes'][r].edge_end = 0
        for c in range(N_NODES):
            if adjmat[r][c]:
                data['nodes'][r].edge_end += 1
                data['edges'].append(Edge(dst=c))
                e += 1

    for r in range(1, N_NODES):
        data['nodes'][r].edge_begin = data['nodes'][r - 1].edge_end
        data['nodes'][r].edge_end += data['nodes'][r - 1].edge_end

    # Pick Starting Node
    starting_node = random.randint(0, N_NODES - 1)
    while data['nodes'][starting_node].edge_end - data['nodes'][starting_node].edge_begin < 2:
        starting_node = random.randint(0, N_NODES - 1)
    data['starting_node'] = starting_node

    return data


if __name__ == "__main__":
    generated_data = generate_random_graph()
    write_data_to_file(generated_data)
