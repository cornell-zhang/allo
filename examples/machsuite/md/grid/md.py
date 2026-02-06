import allo
from allo.ir.types import float64,int32, Int

# Define MIN and MAX functions to replicate the C macros
nAtoms:int32=256
domainEdge:float64=20.0
blockSide:int32=4
nBlocks:int32=blockSide*blockSide*blockSide
blockEdge:float64=domainEdge/blockSide
densityFactor:int32=10
lj1:float64=1.5
lj2:float64=2.0

def MIN(x:int32, y:Int(34))->int32:
    return x if x < y else y

def MAX(x:int32, y:Int(34))->int32:
    return x if x > y else y

# class dvector_t:
#     def __init__(self, x:float64, y:float64, z:float64):
#         self.x = x
#         self.y = y
#         self.z = z

# class ivector_t:
#     def __init__(self, x:int32, y:int32, z:int32):
#         self.x = x
#         self.y = y
#         self.z = z


def md_x(n_points:int32[blockSide,blockSide,blockSide],
        
        position_x:float64[blockSide,blockSide,blockSide,densityFactor],
        position_y:float64[blockSide,blockSide,blockSide,densityFactor],
        position_z:float64[blockSide,blockSide,blockSide,densityFactor])->float64[blockSide,blockSide,blockSide,densityFactor]:
       
    
    q_x:  float64=0.0
    q_y:  float64=0.0
    q_z:  float64=0.0
    p_x:  float64=0.0
    p_y:  float64=0.0
    p_z:  float64=0.0

    dx:float64=0.0
    dy:float64=0.0
    dz:float64=0.0
    r2inv:float64=0.0
    r6inv:float64=0.0
    potential:float64=0.0
    f:float64=0.0
    base_q_x: float64[densityFactor]=0.0
    base_q_y: float64[densityFactor]=0.0
    base_q_z: float64[densityFactor]=0.0
    sum_x:float64=0.0
    force_x:float64[blockSide,blockSide,blockSide,densityFactor]=0.0

    q_idx_range:int32=0

    # Iterate over the grid, block by block
    for b0_x,b0_y,b0_z in allo.grid(blockSide,blockSide,blockSide):
        # Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        for b1_x in range(MAX(0, b0_x - 1), MIN(blockSide, b0_x + 2)):
            for b1_y in range(MAX(0, b0_y - 1), MIN(blockSide, b0_y + 2)):
                for b1_z in range(MAX(0, b0_z - 1), MIN(blockSide, b0_z +2)):
                    q_idx_range = n_points[b1_x,b1_y,b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x,b1_y,b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x,b1_y,b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x,b1_y,b1_z, q_idx]
                    for p_idx in range(n_points[b0_x,b0_y,b0_z]):
                        p_x = position_x[b0_x,b0_y,b0_z,p_idx]
                        p_y = position_y[b0_x,b0_y,b0_z,p_idx]
                        p_z = position_z[b0_x,b0_y,b0_z,p_idx]
                        sum_x = force_x[b0_x,b0_y,b0_z,p_idx]

                        for q_idx in range(q_idx_range):
                            q_x = base_q_x[q_idx]
                            q_y = base_q_y[q_idx]
                            q_z = base_q_z[q_idx]
                            if (q_x != p_x or q_y != p_y or q_z != p_z):
                                dx = p_x - q_x
                                dy = p_y - q_y
                                dz = p_z - q_z
                                r2inv = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv = r2inv * r2inv * r2inv
                                potential = r6inv * (lj1 * r6inv - lj2)
                                f = r2inv * potential
                                sum_x += f * dx

                        force_x[b0_x,b0_y,b0_z,p_idx]=sum_x
    return force_x

def md_y(n_points:int32[blockSide,blockSide,blockSide],
         
        position_x:float64[blockSide,blockSide,blockSide,densityFactor],
        position_y:float64[blockSide,blockSide,blockSide,densityFactor],
        position_z:float64[blockSide,blockSide,blockSide,densityFactor])->float64[blockSide,blockSide,blockSide,densityFactor]:
       
    
    q_x:  float64=0.0
    q_y:  float64=0.0
    q_z:  float64=0.0
    p_x:  float64=0.0
    p_y:  float64=0.0
    p_z:  float64=0.0

    dx:float64=0.0
    dy:float64=0.0
    dz:float64=0.0
    r2inv:float64=0.0
    r6inv:float64=0.0
    potential:float64=0.0
    f:float64=0.0
    base_q_x: float64[densityFactor]=0.0
    base_q_y: float64[densityFactor]=0.0
    base_q_z: float64[densityFactor]=0.0
    sum_y:float64=0.0
    force_y:float64[blockSide,blockSide,blockSide,densityFactor]=0.0

    q_idx_range:int32=0

    # Iterate over the grid, block by block
    for b0_x,b0_y,b0_z in allo.grid(blockSide,blockSide,blockSide):
        # Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        for b1_x in range(MAX(0, b0_x - 1), MIN(blockSide, b0_x + 2)):
            for b1_y in range(MAX(0, b0_y - 1), MIN(blockSide, b0_y + 2)):
                for b1_z in range(MAX(0, b0_z - 1), MIN(blockSide, b0_z +2)):
                    q_idx_range = n_points[b1_x,b1_y,b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x,b1_y,b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x,b1_y,b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x,b1_y,b1_z, q_idx]
                    for p_idx in range(n_points[b0_x,b0_y,b0_z]):
                        p_x = position_x[b0_x,b0_y,b0_z,p_idx]
                        p_y = position_y[b0_x,b0_y,b0_z,p_idx]
                        p_z = position_z[b0_x,b0_y,b0_z,p_idx]
                        sum_y = force_y[b0_x,b0_y,b0_z,p_idx]

                        for q_idx in range(q_idx_range):
                            q_x = base_q_x[q_idx]
                            q_y = base_q_y[q_idx]
                            q_z = base_q_z[q_idx]
                            if (q_x != p_x or q_y != p_y or q_z != p_z):
                                dx = p_x - q_x
                                dy = p_y - q_y
                                dz = p_z - q_z
                                r2inv = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv = r2inv * r2inv * r2inv
                                potential = r6inv * (lj1 * r6inv - lj2)
                                f = r2inv * potential
                                sum_y += f * dy

                        force_y[b0_x,b0_y,b0_z,p_idx]=sum_y
    return force_y

def md_z(n_points:int32[blockSide,blockSide,blockSide],
         
        position_x:float64[blockSide,blockSide,blockSide,densityFactor],
        position_y:float64[blockSide,blockSide,blockSide,densityFactor],
        position_z:float64[blockSide,blockSide,blockSide,densityFactor])->float64[blockSide,blockSide,blockSide,densityFactor]:
       
    
    q_x:  float64=0.0
    q_y:  float64=0.0
    q_z:  float64=0.0
    p_x:  float64=0.0
    p_y:  float64=0.0
    p_z:  float64=0.0

    dx:float64=0.0
    dy:float64=0.0
    dz:float64=0.0
    r2inv:float64=0.0
    r6inv:float64=0.0
    potential:float64=0.0
    f:float64=0.0
    base_q_x: float64[densityFactor]=0.0
    base_q_y: float64[densityFactor]=0.0
    base_q_z: float64[densityFactor]=0.0
    sum_z:float64=0.0
    force_z:float64[blockSide,blockSide,blockSide,densityFactor]=0.0

    q_idx_range:int32=0

    # Iterate over the grid, block by block
    for b0_x,b0_y,b0_z in allo.grid(blockSide,blockSide,blockSide):
        # Iterate over the 3x3x3 (modulo boundary conditions) cube of blocks around b0
        for b1_x in range(MAX(0, b0_x - 1), MIN(blockSide, b0_x + 2)):
            for b1_y in range(MAX(0, b0_y - 1), MIN(blockSide, b0_y + 2)):
                for b1_z in range(MAX(0, b0_z - 1), MIN(blockSide, b0_z +2)):
                    q_idx_range = n_points[b1_x,b1_y,b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x,b1_y,b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x,b1_y,b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x,b1_y,b1_z, q_idx]
                    for p_idx in range(n_points[b0_x,b0_y,b0_z]):
                        p_x = position_x[b0_x,b0_y,b0_z,p_idx]
                        p_y = position_y[b0_x,b0_y,b0_z,p_idx]
                        p_z = position_z[b0_x,b0_y,b0_z,p_idx]
                        sum_z = force_z[b0_x,b0_y,b0_z,p_idx]

                        for q_idx in range(q_idx_range):
                            q_x = base_q_x[q_idx]
                            q_y = base_q_y[q_idx]
                            q_z = base_q_z[q_idx]
                            if (q_x != p_x or q_y != p_y or q_z != p_z):
                                dx = p_x - q_x
                                dy = p_y - q_y
                                dz = p_z - q_z
                                r2inv = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv = r2inv * r2inv * r2inv
                                potential = r6inv * (lj1 * r6inv - lj2)
                                f = r2inv * potential
                                sum_z += f * dz

                        force_z[b0_x,b0_y,b0_z,p_idx]=sum_z
    return force_z

if __name__ == "__main__":
    s_x=allo.customize(md_x)
    print(s_x.module)
    s_x.build()

    s_y=allo.customize(md_y)
    print(s_y.module)
    s_y.build()

    s_z=allo.customize(md_z)
    print(s_z.module)
    s_z.build()

    print("build success")
