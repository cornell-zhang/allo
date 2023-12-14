# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import allo
from allo.ir.types import int8, int32


def test_cascaded_int8_gemm():
    from allo.library.systolic import systolic

    # (seq, hidden) x (hidden, 4*hidden) = (seq, 4*hidden)
    # (seq, 4*hidden) x (4*hidden, hidden) = (seq, hidden)
    # L, D = 512, 768
    # M0, M1 = 16, 16
    L, D = 4, 4
    M0, M1 = 2, 2
    W_A_cst = np.random.randint(-4, 4, size=(D, 4 * D)).astype(np.int8)
    W_B_cst = np.random.randint(-4, 4, size=(4 * D, D)).astype(np.int8)

    def top(X: int8[L, D]) -> int8[L, D]:
        Z: int8[L, 4 * D]
        Y: int8[L, D]
        W_A: int8[D, 4 * D] = W_A_cst
        W_B: int8[4 * D, D] = W_B_cst
        systolic[int8, int8, int8, L, D, 4 * D, M0, M1, "FFN1"](X, W_A, Z)
        systolic[int8, int8, int8, L, 4 * D, D, M0, M1, "FFN2"](Z, W_B, Y)
        return Y

    s_top = allo.customize(top)
    # CPU testing
    mod = s_top.build()
    X = np.random.randint(-4, 4, size=(L, D)).astype(np.int8)
    allo_C = mod(X)
    np_C = X @ W_A_cst @ W_B_cst
    np.testing.assert_allclose(allo_C, np_C, atol=1e-3)
    print("Passed!")
    # Submodule customization
    # Compose with submodule
    s_top.use_def_chain.dump_graph("top")
    s_top.compose(
        systolic, instantiate=[int8, int8, int8, L, D, 4 * D, M0, M1], id="FFN1"
    )
    s_top.compose(
        systolic, instantiate=[int8, int8, int8, L, 4 * D, D, M0, M1], id="FFN2"
    )
    s_top.to(s_top.Z, "systolic_FFN2", depth=M0 * 4 * D)
    # HLS testing
    code = s_top.build("vhls")
    if os.system("which vitis_hls >> /dev/null") == 0:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="hw",
            project=f"FFN_{L}x{D}_tile_{M0}x{M1}.prj",
        )
        hls_mod()


def test_int8_gemm():
    from allo.library.systolic import packed_systolic

    # (seq, hidden) x (hidden, 4*hidden) = (seq, 4*hidden)
    # (seq, 4*hidden) x (4*hidden, hidden) = (seq, hidden)
    L, D = 512, 768
    M0, M1 = 16, 16
    PP = 4
    W_A_cst = np.random.randint(-4, 4, size=(D, 4 * D)).astype(np.int8)
    W_A_packed = W_A_cst.view(np.int32)

    def top(X: int32[L, D // PP]) -> int32[L, 4 * D // PP]:
        Z: int32[L, 4 * D // PP]
        W_A: int32[D, 4 * D // PP] = W_A_packed
        packed_systolic[int8, int8, int8, L, D, 4 * D, M0, M1, PP](X, W_A, Z)
        return Z

    s_top = allo.customize(top)
    # CPU testing
    mod = s_top.build()
    X = np.random.randint(-4, 4, size=(L, D)).astype(np.int8)
    packed_X = X.view(np.int32)
    allo_C = mod(packed_X)
    np_C = X @ W_A_cst
    np_C_packed = np_C.view(np.int32)
    np.testing.assert_allclose(allo_C, np_C_packed, atol=1e-3)
    print("Passed!")
    # Compose with submodule
    s_top.compose(
        packed_systolic, instantiate=[int32, int32, int32, L, D, 4 * D, M0, M1, PP]
    )
    code = s_top.build("vhls")
    if os.system("which vitis_hls >> /dev/null") == 0:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="hw",
            project=f"single_packed_{PP}_{L}x{D}_tile_{M0}x{M1}.prj",
        )
        hls_mod()


if __name__ == "__main__":
    test_int8_gemm()
