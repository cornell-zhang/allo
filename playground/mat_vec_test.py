"""Runs a comprehensive set of tests on
the matrix vector multiply module."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, unrecognized-inline-option, import-outside-toplevel, too-many-locals

import os
import sys
import multiprocessing as mp
import numpy as np
import allo
from allo import dsl
from allo.ir.types import index, int8, int16, int32, int64, int128, int256, int512
from allo.ir.types import uint8, uint16, uint32, uint64, uint128, uint256, uint512
from allo.utils import get_np_struct_type
from allo.backend import hls

T, F = True, False


def get_types(pp=1, csim=False):
    """Returns the appropritate np and allo types
    for the given packing factor."""

    if pp == 1:
        table = {"np": np.uint8, "allo": uint8}
    elif pp == 2:
        table = {"np": np.uint16, "allo": uint16}
    elif pp == 4:
        table = {"np": np.uint32, "allo": uint32}
    elif pp == 8:
        table = {"np": np.uint64, "allo": uint64}
    elif pp == 16:
        assert not csim, f"csim not supported for: {pp}"
        table = {"np": get_np_struct_type(128), "allo": uint128}
    elif pp == 32:
        assert not csim, f"csim not supported for: {pp}"
        table = {"np": get_np_struct_type(256), "allo": uint256}
    elif pp == 64:
        assert not csim, f"csim not supported for: {pp}"
        table = {"np": get_np_struct_type(512), "allo": uint512}
    else:
        raise ValueError(f"Unsupported packing factor: {pp}")

    return table


def test_basic_int8_gemv_pe(runs=1, ll=4, dd=2, pp=1, p_loc=0):
    """Runs selected test configuration on PE kernel."""
    from allo.library.gemv import pe_kernel

    np.random.seed(seed=1)

    def top(
        x_in: "int8[pp, (dd // pp)*ll]", y_in: "int8[pp, (dd // pp)*ll]"
    ) -> "int8[pp,(dd // pp)]":
        d: index
        d = p_loc
        z_out: int8[pp, (dd // pp)]
        pe_kernel[ll, dd, pp](x_in, y_in, z_out, d)
        return z_out

    mod = allo.customize(top).build()

    x = np.random.randint(-4, 4, size=(runs, pp, (dd // pp) * ll)).astype(np.int8)
    y = np.random.randint(-4, 4, size=(runs, pp, ll)).astype(np.int8)
    y_stack = np.ascontiguousarray(
        np.concatenate([y for _ in range((dd // pp))], axis=2)
    )

    c_allo = [mod(x_itr, y_stack_itr)[p_loc] for x_itr, y_stack_itr in zip(x, y_stack)]
    c_np = [
        [
            np.dot(x_itr[p_loc], Y_part[p_loc])
            for x_itr in np.split(X_part, (dd // pp), axis=1)
        ]
        for X_part, Y_part in zip(x, y)
    ]

    np.testing.assert_equal(c_allo, c_np)
    print(f"Passed PE! {ll} {dd} {pp}")


# Test setup for matrix vector multiply
# For best perf P factor should equal MM
def test_basic_int8_gemv(ll=2, dd=2, pp=1, hw=False, runs=(1, 1)):
    """Runs selected test configuration of mat vec multiply."""
    from allo.library.gemv import int8xint8_mat_vec

    # np.random.seed(seed=400)
    np.set_printoptions(formatter={"int": hex})

    # np_type, allo_type = get_types(pp, csim)
    types = get_types(pp, runs[1] != 0)

    def top[Ty](x_in: "Ty[dd // pp, ll]", y_in: "int8[ll]") -> "int8[dd]":
        z_out: int8[dd]
        int8xint8_mat_vec[ll, dd, pp](x_in, y_in, z_out)
        return z_out

    s_top = allo.customize(top, instantiate=[types["allo"]])

    # CPU testing
    mod = s_top.build()

    x = np.random.randint(0, 4, size=(runs[0], dd, ll)).astype(np.int8)
    # x_packed = np.array([x.view(types["np"]) for x in X]) # Horizontal pack
    x_packed = np.array(
        [
            np.ascontiguousarray(
                np.ascontiguousarray(x_itr.transpose()).view(types["np"]).transpose()
            )
            for x_itr in x
        ]
    )  # Vertical Pack

    np.array(
        [
            np.ascontiguousarray(
                np.ascontiguousarray(x_itr.transpose()).view(types["np"]).transpose
            )
            for x_itr in x
        ]
    )

    y = np.random.randint(-4, 4, size=(runs[0], ll)).astype(np.int8)

    c_allo = [mod(x_packed_itr, y_itr) for x_packed_itr, y_itr in zip(x_packed, y)]
    c_np = [np.dot(x_itr, y_itr) for x_itr, y_itr in zip(x, y)]

    np.testing.assert_equal(c_allo, c_np)
    print(f"Passed! {ll} {dd} {pp}")

    # confirm vitis usage
    if not ((runs[1] != 0) or hw):
        return
    assert hls.is_available("vitis_hls"), "Vitis HLS not found"

    # Compose with submodule
    s_top.compose(int8xint8_mat_vec, instantiate=[ll, dd, pp])
    s_top.dataflow("top")  # important

    if runs[1] != 0:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="csim",
            project=f"mat_vec_{ll}x{dd}_{pp}_csim.prj",
        )

        c_csim = np.zeros((runs[1], dd), dtype=np.int8)
        for x_packed_itr, y_itr, c_csim_itr in zip(x_packed, y, c_csim):
            hls_mod(x_packed_itr, y_itr, c_csim_itr)

        np.testing.assert_equal(c_csim, c_np[0 : runs[1]])
        print(f"Passed csim! {ll} {dd} {pp}")

    if hw:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="hw",
            project=f"mat_vec_{ll}x{dd}_{pp}.prj",
        )
        hls_mod()


if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    mp.set_start_method("spawn")

    # Run kernel test
    # fmt: off
    PE_tests = [
      # (runs, ll, dd, pp),
        (10,   6,  1,  1 ),
        (10,   6,  2,  1 ),
        (10,   53, 1,  1 ),
        (10,   29, 4,  1 ),
        (10,   29, 4,  2 ),
        (10,   29, 4,  4 ),
        (10,   29, 64, 4 ),
    ]
    # fmt: on

    processes = [
        mp.Process(target=test_basic_int8_gemv_pe, args=args) for args in PE_tests
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # All full tests
    # fmt: off
    full_tests = [
      # (L,   D,   P,  hw, #sw,#sim,),
        (2,   2,   2,  F, (100, 2), ),
        (3,   3,   1,  F, (100, 2), ),
        (2,   4,   1,  F, (100, 2), ),
        (3,   4,   1,  F, (100, 2), ),
        (4,   4,   1,  F, (100, 2), ),
        (4,   2,   2,  F, (100, 2), ),
        (4,   4,   2,  F, (100, 2), ),
        (4,   4,   4,  F, (100, 2), ),
        (16,  8,   8,  F, (100, 2), ),
        (50,  20,  4,  F, (100, 2), ),
        # (256, 128, 8,  T, (100, 2), ),
        (1024,1024,64, F, (10,  0), ),
        # Currently types over 64 bit aren't supported by Allo for compilation
    ]
    # fmt: on

    # Run each test in it's own process
    processes = [
        mp.Process(target=test_basic_int8_gemv, args=args) for args in full_tests
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
