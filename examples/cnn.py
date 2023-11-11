import numpy as np
import pytest
import allo
import allo.dsl as dsl

from allo.ir.types import int32, float32, index, TypeVar


def padding_builder(inp_size, padding_size):
    B, C, H, W = inp_size
    PH, PW = padding_size
    T_A = float32

    def padding(inp: float32[B, C, H, W]) -> (float32[B, C, H+2*PH, W+2*PW]):
        out: float32[B, C, H+2*PH, W+2*PW]
        for b, c, h, w in allo.grid(B, C, H+2*PH, W+2*PW):
            v: T_A = 0
            if PH <= h and h < H+PH and PW <= w and w < W+PW:
                v = inp[b, c, h-PH, w-PW]
            out[b, c, h, w] = v
        return out
    
    s = allo.customize(padding, enable_tensor=False)
    return s, padding


def conv2d_builder(inp_size, weight_size, stride):
    # High order function that returns a function that takes in the input and weight
    B, CI, H, W = inp_size
    CO, CI, KH, KW = weight_size
    HO = (H - KH) // stride + 1
    WO = (W - KW) // stride + 1
    T_A = float32

    def conv2d(inp: T_A[B, CO, H, W], weight: T_A[B, CI, KH, KH]) -> (T_A[B, CO, HO, WO]):
        out: T_A[B, CO, HO, WO]
        for b, co, ho, wo in allo.grid(B, CO, HO, WO):
            v: T_A = 0
            for ci, r, c in allo.reduction(CI, KH, KW):
                v += inp[b, ci, stride*ho+r, stride*wo+c] * weight[co, ci, r, c]
            out[b, co, ho, wo] = v
        return out
    
    s = allo.customize(conv2d, enable_tensor=False)
    LB = s.reuse_at(s.inp, axis="ho")
    WB = s.reuse_at(LB, axis="wo")
    s.partition(LB, dim=1)
    s.partition(WB)
    print(s.module)

    return s, conv2d


def maxpool_builder(inp_size, kernel_size, stride):
    # High order function that returns a function that takes in the input and weight
    B, CI, H, W = inp_size
    KH, KW = kernel_size
    HO = (H - KH) // stride + 1
    WO = (W - KW) // stride + 1
    T_A = float32

    def maxpool(inp: T_A[B, CI, H, W]) -> (T_A[B, CI, HO, WO]):
        out: T_A[B, CI, HO, WO]
        for b, ci, ho, wo in allo.grid(B, CI, HO, WO):
            v: T_A = 0
            for r, c in allo.reduction(KH, KW):
                if inp[b, ci, stride*ho+r, stride*wo+c] > v:
                    v = inp[b, ci, stride*ho+r, stride*wo+c]
            out[b, ci, ho, wo] = v
        return out
    
    s = allo.customize(maxpool, enable_tensor=False)
    LB = s.reuse_at(s.inp, axis="ho")
    WB = s.reuse_at(LB, axis="wo")
    s.partition(LB, dim=1)
    s.partition(WB)

    return s, maxpool
            

g_conv1_weight: float32[64, 3, 7, 7] = np.random.rand(64, 3, 7, 7).astype(np.float32)
g_bn1_weight: float32[64] = np.random.rand(64).astype(np.float32)
g_bn1_bias: float32[64] = np.random.rand(64).astype(np.float32)

s0, padding = padding_builder((1, 3, 224, 224), (3, 3))
s1, conv2d = conv2d_builder((1, 3, 230, 230), (64, 3, 7, 7), 2)
# s2, maxp = maxpool_builder((1, 64, 112, 112), (3, 3), 2)


def forward(x: float32[1, 3, 224, 224]) -> (float32[1, 64, 112, 112]):
    conv1_weight: float32[64, 3, 7, 7] = g_conv1_weight
    bn1_weight: float32[64] = g_bn1_weight
    bn1_bias: float32[64] = g_bn1_bias

    pad1 = padding(x)
    conv1 = conv2d(pad1, conv1_weight)

    bn1 = dsl.batchnorm(conv1, bn1_weight, bn1_bias)
    relu = dsl.relu(bn1)

    # x2 = pad(relu, dsl.ones((3, 3), dtype=int))
    # o = maxp(x2, dsl.ones((3, 3), dtype=int), stride=2)
    return relu


s = allo.customize(forward, enable_tensor=False)
s.compose(s0)
s.compose(s1)
# s.compose(s2)


print(s.module)

# .to() must be applied to 
s.to(s.pad1, "conv")