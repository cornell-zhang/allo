# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import allo
from allo.ir.types import uint64, uint256, int32, float32, int512, bool
from allo.utils import get_np_struct_type
import allo.dataflow as df
from allo.backend import hls


# ##############################################################
# Test basic
# ##############################################################
def test_grid_for_gemm():
    # from `test_builder.py`, with return value

    # This test is to make sure the whole flow works properly.
    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        # Use grid_for with name annotation
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    # 1. Create customization
    s = allo.customize(gemm)
    print(s.module)

    # 2. Apply transformations and make sure each step the module can be printed
    s.split("i", 8)
    print(s.module)
    s.split("j", 8)
    print(s.module)
    s.reorder("i.outer", "j.outer", "i.inner", "j.inner")
    print(s.module)
    # Make sure the generated loops are correct and ordered
    loops = s.get_loops()
    expected = ["i.outer", "j.outer", "i.inner", "j.inner", "k"]
    assert expected == list(loops.C.loops.keys())

    # 5. HLS CSIM
    if hls.is_available("vitis_hls"):
        np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
        np_C = np.matmul(np_A, np_B)
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_mod = s.build(
                target="vitis_hls",
                mode="sw_emu",
                project=tmpdir,
            )
            sw_out = np.zeros((32, 32), dtype=np.int32)
            hls_mod(np_A, np_B, sw_out)
            np.testing.assert_allclose(sw_out, np_C, atol=1e-3)
            print("Passed HLS test!")


def test_vitis_gemm_template_int32():
    # from `test_vhls.py`
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[int32, 32, 32, 32])
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((32, 32), dtype=np.int32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
            print("Passed!")


def test_vitis_gemm_template_float32():
    # from `test_vhls.py`
    def gemm[T, M, N, K](A: "T[M, K]", B: "T[K, N]") -> "T[M, N]":
        C: T[M, N] = 0
        for i, j, k in allo.grid(M, N, K, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm, instantiate=[float32, 64, 64, 64])
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            np_A = np.random.random(size=(64, 64)).astype(np.float32)
            np_B = np.random.random(size=(64, 64)).astype(np.float32)
            np_C = np.matmul(np_A, np_B)
            np_C_allo = np.zeros((64, 64), dtype=np.float32)
            mod(np_A, np_B, np_C_allo)
            np.testing.assert_allclose(np_C, np_C_allo, rtol=1e-4)
            print("Passed!")


def test_vitis_io_stream():
    # from `test_vhls.py`
    def foo(A: int32[32, 32], B: int32[32, 32]):
        pass

    def top(A: int32[32, 32]) -> int32[32, 32]:
        B: int32[32, 32]
        foo(A, B)
        return B

    s = allo.customize(top)
    s.dataflow("top")
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            print(s.module)
            np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
            np_B = np.zeros((32, 32), dtype=np.int32)
            hls_mod(np_A, np_B)
            print("Passed!")


def test_scalar():
    # from `test_vhls.py`, test scalar
    def case1(C: int32) -> int32:
        return C + 1

    s = allo.customize(case1)
    mod = s.build()
    assert mod(1) == 2
    print("Passed CPU simulation!")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        if hls.is_available("vitis_hls"):
            ret = np.zeros((1,), dtype=np.int32)
            mod(1, ret)
            assert ret == 2
            print("Passed!")


def test_pointer_generation():
    # from `test_vhls.py`, test bool and scalar
    def top(inst: bool, C: int32[3]):
        if inst:
            C[0] = C[0] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        assert "bool v" in mod.hls_code and ",," not in mod.hls_code
        if hls.is_available("vitis_hls"):
            inst = np.array([1], dtype=np.bool_)
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(inst, C)
            np.testing.assert_allclose(C, [2, 2, 3], rtol=1e-5)
            print("Passed!")

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            inst = np.array([0], dtype=np.bool_)
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(inst, C)
            np.testing.assert_allclose(C, [1, 2, 3], rtol=1e-5)
            print("Passed!")


def test_bool_array():
    # modified from `test_vhls.py`, test bool array
    def top(inst: bool[3], C: int32[3]):
        for i in range(3):
            if inst[i]:
                C[i] = C[i] + 1

    s = allo.customize(top)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
        if hls.is_available("vitis_hls"):
            inst = np.array([1, 0, 1], dtype=np.bool_)
            C = np.array([1, 2, 3], dtype=np.int32)
            mod(inst, C)
            np.testing.assert_allclose(C, [2, 2, 4], rtol=1e-5)
            print("Passed!")


# ##############################################################
# Test large bitwidth
# ##############################################################
def test_vadd():
    # test 256 bits
    VLEN = 256
    ELEN = 32

    np_256 = get_np_struct_type(VLEN)

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            A: uint256[1],
            B: uint256[1],
            C: uint256[1],
        ):
            for i in allo.grid(VLEN // ELEN, name="vec_nest"):
                C[0][i * ELEN : (i + 1) * ELEN] = (
                    A[0][i * ELEN : (i + 1) * ELEN] + B[0][i * ELEN : (i + 1) * ELEN]
                )

    A = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    B = np.random.randint(0, 64, (VLEN // ELEN,)).astype(np.uint32)
    C = np.zeros(
        VLEN // ELEN,
    ).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256)
    packed_B = np.ascontiguousarray(B).view(np_256)
    packed_C = np.ascontiguousarray(C).view(np_256)

    mod = df.build(top, target="simulator")
    mod(packed_A, packed_B, packed_C)
    unpacked_C = packed_C.view(np.uint32)
    np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
    print("PASSED!")

    s = df.customize(top)
    # unroll the lanes
    nest_loop_i = s.get_loops("VEC_0")["vec_nest"]["i"]
    s.unroll(nest_loop_i)
    print(s.module)

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(
                target="vitis_hls",
                mode="sw_emu",
                project=tmpdir,
                wrap_io=False,
            )
            mod(packed_A, packed_B, packed_C)
            unpacked_C = packed_C.view(np.uint32)
            np.testing.assert_allclose(A + B, unpacked_C, rtol=1e-5, atol=1e-5)
            print(unpacked_C)
            print("Passed Test!")


def test_packed_add():
    # test 512 bits
    np_512 = get_np_struct_type(512)

    def packed_add(input_: int512[1], output_: int512[1]):
        unpacked: int32[16]
        packed: int512

        for i in range(16):
            unpacked[i] = input_[0][i * 32 : (i + 1) * 32]

        for i in range(16):
            unpacked[i] = unpacked[i] + 1

        for i in range(16):
            packed[i * 32 : (i + 1) * 32] = unpacked[i]

        output_[0] = packed

    A = np.random.randint(0, 64, (512 // 32,)).astype(np.int32)
    B = np.zeros(512 // 32).astype(np.int32)
    packed_A = np.ascontiguousarray(A).view(np_512)
    packed_B = np.ascontiguousarray(B).view(np_512)
    s = allo.customize(packed_add)
    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
            mod(packed_A, packed_B)
            unpacked_B = packed_B.view(np.int32)
            np.testing.assert_allclose(A + 1, unpacked_B, rtol=1e-5, atol=1e-5)
            print(unpacked_B)
            print("Passed Test!")


def test_hbm_mapping_config():
    """Test that HBM mapping generates the correct configuration file."""
    import os

    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)

    # Test with HBM mapping configuration
    # Use actual argument names from the function definition
    # Return values are named "output_0", "output_1", etc.
    hbm_mapping = {
        "A": 0,  # HBM channel 0 (using int)
        "B": "HBM[1]",  # HBM channel 1 (using string)
        "output_0": "DDR[0]",  # Return value -> DDR bank 0
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build with HBM mapping
        mod = s.build(
            target="vitis_hls",
            mode="csyn",  # Just use csyn to generate files without running HLS
            project=tmpdir,
            configs={"hbm_mapping": hbm_mapping},
        )

        # Check that the configuration file was created
        cfg_file = os.path.join(tmpdir, "gemm.cfg")
        assert os.path.exists(
            cfg_file
        ), f"Configuration file {cfg_file} was not created"

        # Read and verify the configuration file content
        with open(cfg_file, "r") as f:
            cfg_content = f.read()

        print("Generated config file content:")
        print(cfg_content)

        # Verify the content - user names should be mapped to HLS argument names
        # The generated HLS code uses names like v15, v16, v17
        assert "[connectivity]" in cfg_content
        assert ":HBM[0]" in cfg_content  # Check memory spec is present
        assert ":HBM[1]" in cfg_content
        assert ":DDR[0]" in cfg_content
        # Verify HLS argument names are used (vXX format) with kernel instance suffix
        assert (
            "sp=gemm_1.v" in cfg_content
        ), "Config should use kernel instance name with HLS argument names"

        # Check that VPP_LDFLAGS in makefile includes the config file
        makefile_us = os.path.join(tmpdir, "makefile_us_alveo.mk")
        if os.path.exists(makefile_us):
            with open(makefile_us, "r") as f:
                makefile_content = f.read()
            assert (
                "--config gemm.cfg" in makefile_content
            ), "Makefile should reference the config file"

    print("HBM mapping test passed!")


def test_hbm_mapping_function():
    """Test the generate_hbm_config function directly."""
    from allo.backend.vitis import generate_hbm_config

    # Test with mixed input types
    hbm_mapping = {
        "inp_addr_0": 0,
        "inp_addr_1": "HBM[0]",
        "wk_addr_0": 1,
        "wv_addr_0": "DDR[0]",
    }

    cfg_content = generate_hbm_config("my_kernel", hbm_mapping)
    print("Generated config content:")
    print(cfg_content)

    # Verify content
    assert "[connectivity]" in cfg_content
    assert "sp=my_kernel_1.inp_addr_0:HBM[0]" in cfg_content
    assert "sp=my_kernel_1.inp_addr_1:HBM[0]" in cfg_content
    assert "sp=my_kernel_1.wk_addr_0:HBM[1]" in cfg_content
    assert "sp=my_kernel_1.wv_addr_0:DDR[0]" in cfg_content

    print("generate_hbm_config test passed!")


if __name__ == "__main__":
    test_grid_for_gemm()
    test_vitis_gemm_template_int32()
    test_vitis_gemm_template_float32()
    test_vitis_io_stream()
    test_scalar()
    test_pointer_generation()
    test_bool_array()

    test_vadd()
    test_packed_add()

    # Test HBM mapping
    test_hbm_mapping_function()
    test_hbm_mapping_config()
