def write_output_data(filename, path):
    with open(filename, 'w') as f:
        f.write("%%\n")
        
        for i in range(len(path)):
            f.write(f"{path[i]}\n")
