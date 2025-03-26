def write_data_to_file(data, filename="input.data"):
    with open(filename, 'w') as f:
        f.write(f"%%\n{data['starting_node']}\n")
        f.write(f"%%\n")
        for node in data['nodes']:
            f.write(f"{node.edge_begin}\n{node.edge_end}\n")
        f.write(f"%%\n")
        for edge in data['edges']:
            f.write(f"{edge.dst}\n")

def write_data_to_file_2(data, filename="check.data"):
    with open(filename, 'w') as f:
        f.write(f"%%\n")
        for item in data:
            f.write(f"{item}\n")


def read_data_from_file(filename="input.data"):
    data = {'nodes': [], 'edges': [], 'starting_node':[]}
    count = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("%"):
                count += 1
            elif count == 1:
                data['starting_node'].append(int(line))
            elif count == 2:
                data['nodes'].append(int(line))
            elif count == 3:
                data['edges'].append(int(line))

    return data
