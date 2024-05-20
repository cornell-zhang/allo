# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, float32
import argparse
from allo.library.systolic import systolic
from allo.library.nn import scaled_dot_product_attention, softmax, layer_norm, GeLU

argparser = argparse.ArgumentParser()
argparser.add_argument("--func", type=str, default="gemm")
argparser.add_argument("--dtype", type=str, default="float32")
argparser.add_argument("--H", type=int, default=12)
argparser.add_argument("--L", type=int, default=512)
argparser.add_argument("--D", type=int, default=768)
argparser.add_argument("--Dffn", type=int, default=3072)
argparser.add_argument("--M0", type=int, default=16)
argparser.add_argument("--M1", type=int, default=16)

args = argparser.parse_args()
func = args.func
dtype = int8 if args.dtype == "int8" else float32
H, L, D, Dffn, M0, M1 = args.H, args.L, args.D, args.Dffn, args.M0, args.M1

if func == "gemm":
    s = allo.customize(systolic, instantiate=[dtype, dtype, dtype, L, D, D, M0, M1])
    s.compose(systolic, instantiate=[dtype, dtype, dtype, L, D, D, M0, M1])
    print(s.build(target="vhls"))
elif func == "sdp":
    assert dtype == float32, "Only float32 is supported for SDP currently"
    s = allo.customize(scaled_dot_product_attention, instantiate=[dtype, H, L, D])
    s.compose(systolic, instantiate=[dtype, dtype, dtype, L, D, D, M0, M1])
    print(s.build(target="vhls"))
elif func == "softmax":
    assert dtype == float32, "Only float32 is supported for softmax currently"
    s = allo.customize(softmax, instantiate=[dtype, D])
    print(s.build(target="vhls"))
elif func == "layernorm":
    assert dtype == float32, "Only float32 is supported for softmax currently"
    s = allo.customize(layer_norm, instantiate=[dtype, L, D])
    print(s.build(target="vhls"))
elif func == "gelu":
    assert dtype == float32, "Only float32 is supported for softmax currently"
    s = allo.customize(GeLU, instantiate=[dtype, L, D])
    print(s.build(target="vhls"))
else:
    raise ValueError(f"Unknown function: {func}")
