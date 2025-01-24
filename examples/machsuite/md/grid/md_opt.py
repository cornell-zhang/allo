import allo
from allo.ir.types import int32, float32
import allo.ir.types as T
import os
import json
import pytest
import numpy as np

# Define MIN and MAX functions to replicate the C macros
nAtoms:int32=256
domainEdge:float32=20.0
blockSide:int32=4
nBlocks:int32=blockSide*blockSide*blockSide
blockEdge:float32=domainEdge/blockSide
densityFactor:int32=10
lj1:float32=1.5
lj2:float32=2.0



def md_top(concrete_type,N,M):
    
    def md[T: (float32, int32), N: int32 , M:int32
                     ](n_points:"T[N,N,N]",position_x:"T[N,N,N,M]",
                       position_y:"T[N,N,N,M]",position_z:"T[N,N,N,M]",
                       force_x:"T[N,N,N,M]",force_y:"T[N,N,N,M]",
                       force_z:"T[N,N,N,M]"):
        b1_x_hi:int32  
        b1_x_lo:int32 
        b1_y_hi:int32  
        b1_y_lo:int32  
        b1_z_hi:int32  
        b1_z_lo:int32   
        p_x:float32
        p_y:float32
        p_z:float32
        sum_x:float32
        sum_y:float32
        sum_z:float32
        q_z:float32
        q_x:float32
        q_y:float32
        dx:float32
        dy:float32
        dz:float32
        r2inv:float32
        r6inv:float32
        potential:float32
        f:float32

            
        # Iterate over the grid, block by block
        for b0_x in allo.grid(N):
            for b0_y in allo.grid(N):
                for b0_z in allo.grid(N):
                    if(b0_x==1):
                        b1_x_lo=0
                    else:
                        b1_x_lo=b0_x-1
                    if(b0_y==1):
                        b1_y_lo=0
                    else:
                        b1_y_lo=b0_y-1
                    if(b0_z==1):
                        b1_z_lo=0
                    else:
                        b1_z_lo=b0_z-1
                    if(b0_x==N-1):
                        b1_x_hi=N
                    else:
                        b1_x_hi=b0_x+2
                    if(b0_y==N-1):
                        b1_y_hi=N
                    else:
                        b1_y_hi=b0_y+2
                    if(b0_z==N-1):
                        b1_z_hi=N
                    else:
                        b1_z_hi=b0_z+2
                    # Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
                    for b1_x in range(b1_x_lo, b1_x_hi):
                        for b1_y in range(b1_y_lo, b1_y_hi):
                            for b1_z in range(b1_z_lo, b1_z_hi):
                                #q_idx_range = n_points[b1_x,b1_y,b1_z]
                                # for i in range(densityFactor):
                                #     base_q_x[i] = position_x[b1_x,b1_y,b1_z, i]
                                #     base_q_y[i] = position_y[b1_x,b1_y,b1_z, i]
                                #     base_q_z[i] = position_z[b1_x,b1_y,b1_z, i]
                                ub0: int32 = n_points[b1_x,b1_y,b1_z]
                                for p_idx in range(ub0):
                                    p_x = position_x[b0_x,b0_y,b0_z,p_idx]
                                    p_y = position_y[b0_x,b0_y,b0_z,p_idx]
                                    p_z = position_z[b0_x,b0_y,b0_z,p_idx]
                                    sum_x = force_x[b0_x,b0_y,b0_z,p_idx]
                                    sum_y = force_y[b0_x,b0_y,b0_z,p_idx]
                                    sum_z = force_z[b0_x,b0_y,b0_z,p_idx]
                                    ub1: int32 = n_points[b1_x,b1_y,b1_z]
                                    for q_idx in range(ub1):
                                        q_x = position_x[b1_x,b1_y,b1_z, q_idx]
                                        q_y = position_y[b1_x,b1_y,b1_z, q_idx]
                                        q_z = position_z[b1_x,b1_y,b1_z, q_idx]
                                        if (q_x != p_x or q_y != p_y or q_z != p_z):
                                            dx = p_x - q_x
                                            dy = p_y - q_y
                                            dz = p_z - q_z
                                            r2inv = 1.0 / (dx * dx + dy * dy + dz * dz)
                                        else:
                                            r2inv=(domainEdge*domainEdge*3.0)*1000   
                                        r6inv = r2inv * r2inv * r2inv
                                        potential = r6inv * (lj1 * r6inv - lj2)
                                        f = r2inv * potential
                                        sum_x += f * dx
                                        sum_y += f * dy
                                        sum_z += f * dz
                                        
                                    force_x[b0_x,b0_y,b0_z,p_idx]=sum_x 
                                    force_y[b0_x,b0_y,b0_z,p_idx]=sum_y 
                                    force_z[b0_x,b0_y,b0_z,p_idx]=sum_z  

    sch=allo.customize(md, instantiate=[concrete_type,N,M])
    print(sch.module)
    return sch


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

def test_md():
    N=4
    M=10
    concrete_type=float32
    sch=md_top(concrete_type,N,M)
    mod=sch.build(target="vivado_hls",mode="csynth", project="grid.prj")
    print(mod)
    
    input = parse_data("input.data")
    check = parse_data("check.data")
    check_x=np.array(check[0][0:640]).astype(np.float32).reshape((4,4,4,10))
    check_y=np.array(check[0][640:1280]).astype(np.float32).reshape((4,4,4,10))
    check_z=np.array(check[0][1280:1920]).astype(np.float32).reshape((4,4,4,10))
    
   
    force_x=np.zeros((4,4,4,10)).astype(np.float32)
    force_y=np.zeros((4,4,4,10)).astype(np.float32)
    force_z=np.zeros((4,4,4,10)).astype(np.float32)
    n_points=np.array(input[0]).astype(np.int32).reshape((4,4,4))
    position_x = np.array(input[1][0:640]).astype(np.float32).reshape((4,4,4,10))
    position_y = np.array(input[1][640:1280]).astype(np.float32).reshape((4,4,4,10))
    position_z = np.array(input[1][1280:1920]).astype(np.float32).reshape((4,4,4,10))
    mod(n_points,position_x,position_y,position_z,force_x,force_y,force_z)
    np.testing.assert_allclose(force_x,check_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force_y,check_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(force_z,check_z, rtol=1e-5, atol=1e-5)
    
    
    

if __name__ == "__main__":
    test_md()
