import allo
from allo.ir.types import int32

m, n, k = 512, 512, 512
    
def matmul_kernel(A: int32[m, k], B: int32[k, n]) -> int32[m, n]:
    return allo.matmul(A, B)

s = allo.customize(matmul_kernel)
aie_module = s.build(target="aie", mode="sim", project="air_prj")
aie_module()