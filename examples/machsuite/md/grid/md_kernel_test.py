import random
import md
import allo
import numpy as np

# Assuming nAtoms and maxNeighbors are predefined constants
nAtoms = 256  # Example value
maxNeighbors = 16  # Example value
domainEdge=20

def distance(position_x, position_y, position_z, i, j):
    if(i!=j):
        delx = position_x[i] - position_x[j]
        dely = position_y[i] - position_y[j]
        delz = position_z[i] - position_z[j]
        r2inv = delx**2 + dely**2 + delz**2
    else:
        r2inv=(domainEdge*domainEdge*3.0)*1000
    return r2inv

def insert_in_order(curr_dist, curr_list, j, dist_ij):
    pos = maxNeighbors - 1
    curr_max = curr_dist[pos]
    if dist_ij > curr_max:
        return
    
    for dist in range(pos, 0, -1):
        if dist_ij < curr_dist[dist]:
            curr_dist[dist] = curr_dist[dist - 1]
            curr_list[dist] = curr_list[dist - 1]
        else:
            break
        pos -= 1
    
    curr_dist[dist] = dist_ij
    curr_list[dist] = j

def build_neighbor_list(position_x, position_y, position_z):
    total_pairs = 0
    NL = [[0 for _ in range(maxNeighbors)] for _ in range(nAtoms)]
    
    for i in range(nAtoms):
        curr_list = [0] * maxNeighbors
        curr_dist = [float('inf')] * maxNeighbors
        
        for j in range(nAtoms):
            if i == j:
                continue
            dist_ij = distance(position_x, position_y, position_z, i, j)
            insert_in_order(curr_dist, curr_list, j, dist_ij)
        
        for k in range(maxNeighbors):
            NL[i][k] = curr_list[k]
            if curr_dist[k] != float('inf'):
                total_pairs += 1
    
    return total_pairs, NL

def populate_neighbor_list(curr_dist, curr_list, i, NL):
    valid_pairs = 0
    for neighbor_iter in range(maxNeighbors):
        NL[i][neighbor_iter] = curr_list[neighbor_iter]
        valid_pairs += 1
    return valid_pairs

def parse_data(file):# refers to 
    data_arrays = []
    current_array = []

    with open(file, 'r') as f:
        for line in f:
            if line.strip() == '%%':
                if current_array:
                    data_arrays.append(current_array)
                    current_array = []
            else:
                num = float(line.strip())
                current_array.append(num)

    data_arrays.append(current_array)
    return data_arrays

# Main function equivalent in Python
if __name__ == "__main__":
    input = parse_data("input.data")
    check = parse_data("check.data")
    check_x=np.array(check[0][0:640]).astype(np.float64).reshape((4,4,4,10))
    check_y=np.array(check[0][640:1280]).astype(np.float64).reshape((4,4,4,10))
    check_z=np.array(check[0][1280:1920]).astype(np.float64).reshape((4,4,4,10))
    
   
    forceX=np.zeros(shape=((4,4,4,10)), dtype=float)
    forceY=np.zeros(shape=((4,4,4,10)), dtype=float)
    forceZ=np.zeros(shape=((4,4,4,10)), dtype=float)

    n_points=np.array(input[0]).astype(np.int32).reshape((4,4,4))
    position_x = np.array(input[1][0:640]).astype(np.float64).reshape((4,4,4,10))
    position_y = np.array(input[1][640:1280]).astype(np.float64).reshape((4,4,4,10))
    position_z = np.array(input[1][1280:1920]).astype(np.float64).reshape((4,4,4,10))
    
    print("here")
    
    s_x = allo.customize(md.md_x)
    mod_x = s_x.build()
    s_y = allo.customize(md.md_y)
    mod_y = s_y.build()
    s_z = allo.customize(md.md_z)
    mod_z = s_z.build()
   

    forceX=mod_x(n_points,position_x, position_y, position_z)
    forceY=mod_y(n_points,position_x, position_y, position_z)
    forceZ=mod_z(n_points,position_x, position_y, position_z)
    #The actual output has more accurate output than check data
    np.testing.assert_allclose(forceX,check_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceY,check_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceZ,check_z, rtol=1e-5, atol=1e-5)




    
