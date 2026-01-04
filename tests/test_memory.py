# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
from allo.ir.types import int32, float32
from allo.memory import Memory, Layout, DTensor

S = Layout.Shard
R = Layout.Replicate


class TestMemoryClass:
    """Test the Memory class for interface customization."""

    def test_memory_default(self):
        """Test default Memory creation."""
        mem = Memory()
        assert mem.resource == "AUTO"
        assert mem.storage_type is None
        assert mem.latency is None
        assert mem.depth is None

    def test_memory_bram(self):
        """Test Memory with BRAM implementation."""
        mem = Memory(resource="BRAM")
        assert mem.resource == "BRAM"
        assert mem.storage_type is None

    def test_memory_uram(self):
        """Test Memory with URAM implementation."""
        mem = Memory(resource="URAM")
        assert mem.resource == "URAM"

    def test_memory_lutram(self):
        """Test Memory with LUTRAM implementation."""
        mem = Memory(resource="LUTRAM")
        assert mem.resource == "LUTRAM"

    def test_memory_storage_type(self):
        """Test Memory with storage type specification."""
        mem = Memory(resource="BRAM", storage_type="RAM_2P")
        assert mem.resource == "BRAM"
        assert mem.storage_type == "RAM_2P"

    def test_memory_all_options(self):
        """Test Memory with all options specified."""
        mem = Memory(resource="URAM", storage_type="RAM_T2P", latency=3, depth=1024)
        assert mem.resource == "URAM"
        assert mem.storage_type == "RAM_T2P"
        assert mem.latency == 3
        assert mem.depth == 1024

    def test_memory_case_insensitive(self):
        """Test that resource and storage_type are case insensitive."""
        mem = Memory(resource="bram", storage_type="ram_2p")
        assert mem.resource == "BRAM"
        assert mem.storage_type == "RAM_2P"

    def test_memory_invalid_resource(self):
        """Test that invalid resource raises ValueError."""
        with pytest.raises(ValueError, match="Invalid resource"):
            Memory(resource="INVALID")

    def test_memory_invalid_storage_type(self):
        """Test that invalid storage_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid storage_type"):
            Memory(resource="BRAM", storage_type="INVALID")

    def test_memory_repr(self):
        """Test Memory string representation."""
        mem = Memory(resource="URAM")
        assert 'resource="URAM"' in repr(mem)

        mem2 = Memory(resource="BRAM", storage_type="RAM_2P", latency=2)
        repr_str = repr(mem2)
        assert 'resource="BRAM"' in repr_str
        assert 'storage_type="RAM_2P"' in repr_str
        assert "latency=2" in repr_str

    def test_memory_equality(self):
        """Test Memory equality comparison."""
        mem1 = Memory(resource="BRAM", storage_type="RAM_2P")
        mem2 = Memory(resource="BRAM", storage_type="RAM_2P")
        mem3 = Memory(resource="URAM")

        assert mem1 == mem2
        assert mem1 != mem3

    def test_memory_hash(self):
        """Test Memory can be used in sets/dicts."""
        mem1 = Memory(resource="BRAM")
        mem2 = Memory(resource="BRAM")
        mem3 = Memory(resource="URAM")

        mem_set = {mem1, mem3}
        assert len(mem_set) == 2
        assert mem2 in mem_set


class TestMemoryWithDTensor:
    """Test Memory integration with DTensor."""

    def test_dtensor_with_memory(self):
        """Test creating DTensor with Memory spec."""
        mem = Memory(resource="URAM")
        dtensor = DTensor(
            rank=None, mapping=None, shape=(32, 32), dtype=int32, spec=mem, name="A"
        )
        assert dtensor.memory == mem
        assert dtensor.layout is None
        assert dtensor.name == "A"
        assert dtensor.shape == (32, 32)

    def test_dtensor_with_layout(self):
        """Test creating DTensor with Layout spec (backward compatibility)."""
        layout = Layout([S(0), R])
        dtensor = DTensor(
            rank=None,
            mapping=None,
            shape=(32, 32),
            dtype=int32,
            spec=layout,
            name="B",
        )
        assert dtensor.layout == layout
        assert dtensor.memory is None

    def test_dtensor_str_with_memory(self):
        """Test DTensor string representation includes memory."""
        mem = Memory(resource="BRAM", storage_type="RAM_2P")
        dtensor = DTensor(
            rank=None, mapping=None, shape=(64,), dtype=float32, spec=mem, name="C"
        )
        str_repr = str(dtensor)
        assert "memory=" in str_repr


class TestMemoryAnnotation:
    """Test Memory type annotation in kernels."""

    def test_kernel_with_memory_annotation(self):
        """Test that kernels can be defined with Memory annotations."""

        def kernel(a: int32[32] @ Memory(resource="URAM")) -> int32[32]:
            b: int32[32]
            for i in range(32):
                b[i] = a[i] + 1
            return b

        s = allo.customize(kernel)
        ir_str = str(s.module)
        print(ir_str)
        # Verify MLIR memory space attribute for URAM (2*16 = 32)
        assert "memref<32xi32, 32 : i32>" in ir_str, "URAM memory space (32) not found"

        mod = s.build()
        np_a = np.arange(32, dtype=np.int32)
        np_b = mod(np_a)
        np.testing.assert_array_equal(np_b, np_a + 1)

    def test_kernel_with_bram_annotation(self):
        """Test kernel with BRAM memory annotation."""

        def kernel(a: float32[16, 16] @ Memory(resource="BRAM")):
            for i, j in allo.grid(16, 16):
                a[i, j] = a[i, j] * 2.0

        s = allo.customize(kernel)
        ir_str = str(s.module)
        print(ir_str)
        # Verify MLIR memory space attribute for BRAM (1*16 = 16)
        assert (
            "memref<16x16xf32, 16 : i32>" in ir_str
        ), "BRAM memory space (16) not found"

        mod = s.build()
        np_a = np.random.rand(16, 16).astype(np.float32)
        expected = np_a * 2.0
        mod(np_a)
        np.testing.assert_allclose(np_a, expected, rtol=1e-5)

    def test_kernel_multiple_memory_annotations(self):
        """Test kernel with multiple Memory annotations."""

        def kernel(
            a: int32[32] @ Memory(resource="BRAM"),
            b: int32[32] @ Memory(resource="URAM"),
            c: int32[32] @ Memory(resource="LUTRAM"),
        ):
            for i in range(32):
                c[i] = a[i] + b[i]

        s = allo.customize(kernel)
        ir_str = str(s.module)
        print(ir_str)
        # Verify MLIR memory space attributes are set correctly
        # BRAM = 1 -> memory_space = 1*16 = 16
        # URAM = 2 -> memory_space = 2*16 = 32
        # LUTRAM = 3 -> memory_space = 3*16 = 48
        assert "memref<32xi32, 16 : i32>" in ir_str, "BRAM memory space (16) not found"
        assert "memref<32xi32, 32 : i32>" in ir_str, "URAM memory space (32) not found"
        assert (
            "memref<32xi32, 48 : i32>" in ir_str
        ), "LUTRAM memory space (48) not found"

        mod = s.build(target="vhls")
        hls_code = mod.hls_code
        print(hls_code)
        # Verify HLS bind_storage pragmas are generated
        assert "#pragma HLS bind_storage" in hls_code
        assert "impl=bram" in hls_code
        assert "impl=uram" in hls_code
        assert "impl=lutram" in hls_code

    def test_kernel_memory_with_storage_type(self):
        """Test kernel with Memory including storage_type."""

        def kernel(
            a: int32[64] @ Memory(resource="BRAM", storage_type="RAM_2P"),
        ) -> int32[64]:
            b: int32[64]
            for i in range(64):
                b[i] = a[i] * 3
            return b

        s = allo.customize(kernel)
        ir_str = str(s.module)
        print(ir_str)
        # Verify MLIR memory space attribute for BRAM + RAM_2P (1*16 + 2 = 18)
        assert (
            "memref<64xi32, 18 : i32>" in ir_str
        ), "BRAM+RAM_2P memory space (18) not found"

        mod = s.build()
        np_a = np.arange(64, dtype=np.int32)
        np_b = mod(np_a)
        np.testing.assert_array_equal(np_b, np_a * 3)


class TestMemoryValid:
    """Test valid Memory resource options."""

    def test_all_valid_resource_types(self):
        """Test all valid resource types."""
        valid_resources = ["BRAM", "URAM", "LUTRAM", "SRL", "AUTO"]
        for resource in valid_resources:
            mem = Memory(resource=resource)
            assert mem.resource == resource

    def test_all_valid_storage_types(self):
        """Test all valid storage types."""
        valid_storage = [
            "RAM_1P",
            "RAM_2P",
            "RAM_T2P",
            "RAM_1WNR",
            "RAM_S2P",
            "ROM_1P",
            "ROM_2P",
            "ROM_NP",
        ]
        for storage in valid_storage:
            mem = Memory(resource="BRAM", storage_type=storage)
            assert mem.storage_type == storage


# Module-level Memory definitions and kernel functions for HLS codegen tests
# (Functions need to be at module level for proper AST parsing)
_MemBram = Memory(resource="BRAM", storage_type="RAM_2P")
_MemUram = Memory(resource="URAM")
_MemBramMulti = Memory(resource="BRAM", storage_type="RAM_1P")


def _kernel_bram(a: int32[32] @ _MemBram) -> int32[32]:
    """Kernel with BRAM + RAM_2P memory annotation."""
    b: int32[32]
    for i in range(32):
        b[i] = a[i] + 1
    return b


def _kernel_uram(a: float32[64] @ _MemUram):
    """Kernel with URAM memory annotation."""
    for i in range(64):
        a[i] = a[i] * 2.0


def _kernel_multi(a: int32[32] @ _MemBramMulti, b: int32[32] @ _MemUram, c: int32[32]):
    """Kernel with multiple memory annotations."""
    for i in range(32):
        c[i] = a[i] + b[i]


class TestMemoryHLSCodegen:
    """Test HLS code generation with Memory pragmas."""

    def test_hls_bind_storage_bram(self):
        """Test that BRAM Memory generates bind_storage pragma."""
        s = allo.customize(_kernel_bram)
        hls_mod = s.build(target="vhls")
        hls_code = hls_mod.hls_code
        print(hls_code)
        # Verify pragma is generated with BRAM implementation
        assert "#pragma HLS bind_storage variable=" in hls_code
        assert "impl=bram" in hls_code
        assert "type=ram_2p" in hls_code

    def test_hls_bind_storage_uram(self):
        """Test that URAM Memory generates bind_storage pragma."""
        s = allo.customize(_kernel_uram)
        hls_mod = s.build(target="vhls")
        hls_code = hls_mod.hls_code
        print(hls_code)
        # Verify pragma is generated with URAM implementation
        assert "#pragma HLS bind_storage variable=" in hls_code
        assert "impl=uram" in hls_code

    def test_hls_multiple_memories(self):
        """Test multiple Memory annotations generate multiple pragmas."""
        s = allo.customize(_kernel_multi)
        hls_mod = s.build(target="vhls")
        hls_code = hls_mod.hls_code
        print(hls_code)
        # Both BRAM and URAM pragmas should be present
        assert "impl=bram" in hls_code
        assert "impl=uram" in hls_code
        assert "type=ram_1p" in hls_code

    def test_memory_space_encoding(self):
        """Test Memory space encoding is correct."""
        # BRAM = 1, RAM_2P = 2 -> memory_space = 1*16 + 2 = 18
        mem = Memory(resource="BRAM", storage_type="RAM_2P")
        assert mem.get_memory_space() == 18

        # URAM = 2, no storage -> memory_space = 2*16 + 0 = 32
        mem = Memory(resource="URAM")
        assert mem.get_memory_space() == 32

        # AUTO = 0 -> memory_space = 0
        mem = Memory(resource="AUTO")
        assert mem.get_memory_space() == 0


if __name__ == "__main__":
    pytest.main([__file__])
