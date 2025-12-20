# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import uint256, uint32, uint8, bool, int8, int16, int32
import allo.dataflow as df
from allo.utils import get_np_struct_type
from allo.backend import hls

VLEN = 256
ELEN = 32

import numpy as np

np_256 = get_np_struct_type(VLEN)

# [NOTE] reference: https://github.com/riscvarchive/riscv-v-spec/releases/tag/v1.0

# VLEN: The number of bits in a single vector register
VLEN = 256
# ELEN: The number of bits in an element within a vector register
ELEN = 32

# EEW: effective element width, the size of all the elements within a vector register
EEW8 = 0  # 8bits
EEW16 = 1  # 16bits
EEW32 = 2  # 32bits


# ############################
# Vector Instructions
# ############################
# - min/max
#   vmax.vv: vector-vector element-wise signed maximum
#   vmax.vx: vector-scalar element-wise signed maximum
VMAX = 0
#   vmin.vv: vector-vector element-wise signed minimum
#   vmin.vx: vector-scalar element-wise signed minimum
VMIN = 1

# - add/sub
#   vadd.vv: vector-vector element-wise add
#   vadd.vx: vector-scalar element-wise add
VADD = 2
#   vsub.vv: vector-vector element-wise subtract
#   vsub.vx: vector-scalar element-wise subtract vd[i] = vs2[i] - rs1
VSUB = 3

# - mul
#   vmul.vv: vector-vector element-wise multiply
#   vmul.vx: vector-scalar element-wise multiply
VMUL = 4

# - reduction
#   vredmax.vs: vd = max( vs1 , vs2[*] )
VREDMAX = 5
#   vredsum.vs: vd = sum( vs1 , vs2[*] )
VREDSUM = 6
# ############################


def test_vec():
    INST_NUM = 4

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def VEC(
            inst_: uint8[INST_NUM],  # instruction
            ele_type_: uint8[INST_NUM],  # EEW: effective element width
            vv_: bool[INST_NUM],  # vector-vector (True) or vector-scalar (False)
            vs1_: uint256[INST_NUM],  # input vector register
            vs2_: uint256[INST_NUM],  # input vector register
            rs1_: uint32[INST_NUM],  # input scalar
            vd_: uint256[INST_NUM],  # output vector register
        ):
            for idx in allo.grid(INST_NUM, name="inst_outer"):
                ele_type: uint8 = ele_type_[idx]
                vv: bool = vv_[idx]
                vs1: uint256 = vs1_[idx]
                rs1: uint32 = rs1_[idx]

                # prepare operand
                operand1_8b: uint256
                for i in allo.grid(VLEN // 8, name="scalar_to_vector_8"):
                    operand1_8b[i * 8 : (i + 1) * 8] = rs1[0:8]
                operand1_16b: uint256
                for i in allo.grid(VLEN // 16, name="scalar_to_vector_16"):
                    operand1_16b[i * 16 : (i + 1) * 16] = rs1[0:16]
                operand1_32b: uint256
                for i in allo.grid(VLEN // 32, name="scalar_to_vector_32"):
                    operand1_32b[i * 32 : (i + 1) * 32] = rs1

                operand1: uint256
                operand2: uint256 = vs2_[idx]
                if vv:
                    operand1 = vs1
                else:
                    if ele_type == EEW8:
                        operand1 = operand1_8b
                    if ele_type == EEW16:
                        operand1 = operand1_16b
                    if ele_type == EEW32:
                        operand1 = operand1_32b

                # compute
                # - min/max
                min_8b: uint256
                min_16b: uint256
                min_32b: uint256
                max_8b: uint256
                max_16b: uint256
                max_32b: uint256
                for i in allo.grid(VLEN // 8, name="min_max_8"):
                    scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                    scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                    condition: bool = scalar1 > scalar2
                    if condition:
                        min_8b[i * 8 : (i + 1) * 8] = scalar2
                        max_8b[i * 8 : (i + 1) * 8] = scalar1
                    else:
                        max_8b[i * 8 : (i + 1) * 8] = scalar2
                        min_8b[i * 8 : (i + 1) * 8] = scalar1
                for i in allo.grid(VLEN // 16, name="min_max_16"):
                    scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                    scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                    condition: bool = scalar1 > scalar2
                    if condition:
                        min_16b[i * 16 : (i + 1) * 16] = scalar2
                        max_16b[i * 16 : (i + 1) * 16] = scalar1
                    else:
                        max_16b[i * 16 : (i + 1) * 16] = scalar2
                        min_16b[i * 16 : (i + 1) * 16] = scalar1
                for i in allo.grid(VLEN // 32, name="min_max_32"):
                    scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                    scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                    condition: bool = scalar1 > scalar2
                    if condition:
                        min_32b[i * 32 : (i + 1) * 32] = scalar2
                        max_32b[i * 32 : (i + 1) * 32] = scalar1
                    else:
                        max_32b[i * 32 : (i + 1) * 32] = scalar2
                        min_32b[i * 32 : (i + 1) * 32] = scalar1

                # - add/sub/mul
                add_8b: uint256
                add_16b: uint256
                add_32b: uint256
                sub_8b: uint256
                sub_16b: uint256
                sub_32b: uint256
                mul_8b: uint256
                mul_16b: uint256
                mul_32b: uint256
                for i in allo.grid(VLEN // 8, name="arith_8"):
                    scalar1: int8 = operand1[i * 8 : (i + 1) * 8]
                    scalar2: int8 = operand2[i * 8 : (i + 1) * 8]
                    add_8b[i * 8 : (i + 1) * 8] = scalar2 + scalar1
                    sub_8b[i * 8 : (i + 1) * 8] = scalar2 - scalar1
                    mul_8b[i * 8 : (i + 1) * 8] = scalar2 * scalar1
                for i in allo.grid(VLEN // 16, name="arith_16"):
                    scalar1: int16 = operand1[i * 16 : (i + 1) * 16]
                    scalar2: int16 = operand2[i * 16 : (i + 1) * 16]
                    add_16b[i * 16 : (i + 1) * 16] = scalar2 + scalar1
                    sub_16b[i * 16 : (i + 1) * 16] = scalar2 - scalar1
                    mul_16b[i * 16 : (i + 1) * 16] = scalar2 * scalar1
                for i in allo.grid(VLEN // 32, name="arith_32"):
                    scalar1: int32 = operand1[i * 32 : (i + 1) * 32]
                    scalar2: int32 = operand2[i * 32 : (i + 1) * 32]
                    add_32b[i * 32 : (i + 1) * 32] = scalar2 + scalar1
                    sub_32b[i * 32 : (i + 1) * 32] = scalar2 - scalar1
                    mul_32b[i * 32 : (i + 1) * 32] = scalar2 * scalar1

                # reduction
                # reduction max
                # - 8 bits
                base_8b: int8 = operand1[0:8]
                operand1_8bs: int8[VLEN // 8]
                for i in range(VLEN // 8):
                    operand1_8bs[i] = operand2[i * 8 : (i + 1) * 8]
                redmax_8b_l0s: int8[VLEN // 16]
                for i in range(VLEN // 16):
                    if operand1_8bs[i * 2] > operand1_8bs[i * 2 + 1]:
                        redmax_8b_l0s[i] = operand1_8bs[i * 2]
                    else:
                        redmax_8b_l0s[i] = operand1_8bs[i * 2 + 1]
                redmax_8b_l1s: int8[VLEN // 32]
                for i in range(VLEN // 32):
                    if redmax_8b_l0s[i * 2] > redmax_8b_l0s[i * 2 + 1]:
                        redmax_8b_l1s[i] = redmax_8b_l0s[i * 2]
                    else:
                        redmax_8b_l1s[i] = redmax_8b_l0s[i * 2 + 1]
                redmax_8b_l2s: int8[VLEN // 64]
                for i in range(VLEN // 64):
                    if redmax_8b_l1s[i * 2] > redmax_8b_l1s[i * 2 + 1]:
                        redmax_8b_l2s[i] = redmax_8b_l1s[i * 2]
                    else:
                        redmax_8b_l2s[i] = redmax_8b_l1s[i * 2 + 1]
                redmax_8b_l3s: int8[VLEN // 128]
                for i in range(VLEN // 128):
                    if redmax_8b_l2s[i * 2] > redmax_8b_l2s[i * 2 + 1]:
                        redmax_8b_l3s[i] = redmax_8b_l2s[i * 2]
                    else:
                        redmax_8b_l3s[i] = redmax_8b_l2s[i * 2 + 1]
                redmax_8b_l4_0: int8
                if redmax_8b_l3s[0] > redmax_8b_l3s[1]:
                    redmax_8b_l4_0 = redmax_8b_l3s[0]
                else:
                    redmax_8b_l4_0 = redmax_8b_l3s[1]
                redmax_8b: int8
                if base_8b > redmax_8b_l4_0:
                    redmax_8b = base_8b
                else:
                    redmax_8b = redmax_8b_l4_0

                # - 16 bits
                base_16b: int16 = operand1[0:16]
                operand1_16bs: int16[VLEN // 16]
                for i in range(VLEN // 16):
                    operand1_16bs[i] = operand2[i * 16 : (i + 1) * 16]
                redmax_16b_l0s: int16[VLEN // 32]
                for i in range(VLEN // 32):
                    if operand1_16bs[i * 2] > operand1_16bs[i * 2 + 1]:
                        redmax_16b_l0s[i] = operand1_16bs[i * 2]
                    else:
                        redmax_16b_l0s[i] = operand1_16bs[i * 2 + 1]
                redmax_16b_l1s: int16[VLEN // 64]
                for i in range(VLEN // 64):
                    if redmax_16b_l0s[i * 2] > redmax_16b_l0s[i * 2 + 1]:
                        redmax_16b_l1s[i] = redmax_16b_l0s[i * 2]
                    else:
                        redmax_16b_l1s[i] = redmax_16b_l0s[i * 2 + 1]
                redmax_16b_l2s: int16[VLEN // 128]
                for i in range(VLEN // 128):
                    if redmax_16b_l1s[i * 2] > redmax_16b_l1s[i * 2 + 1]:
                        redmax_16b_l2s[i] = redmax_16b_l1s[i * 2]
                    else:
                        redmax_16b_l2s[i] = redmax_16b_l1s[i * 2 + 1]
                redmax_16b_l3_0: int16
                if redmax_16b_l2s[0] > redmax_16b_l2s[1]:
                    redmax_16b_l3_0 = redmax_16b_l2s[0]
                else:
                    redmax_16b_l3_0 = redmax_16b_l2s[1]
                redmax_16b: int16
                if base_16b > redmax_16b_l3_0:
                    redmax_16b = base_16b
                else:
                    redmax_16b = redmax_16b_l3_0

                # - 32 bits
                base_32b: int32 = operand1[0:32]
                operand1_32bs: int32[VLEN // 32]
                for i in range(VLEN // 32):
                    operand1_32bs[i] = operand2[i * 32 : (i + 1) * 32]
                redmax_32b_l0s: int32[VLEN // 64]
                for i in range(VLEN // 64):
                    if operand1_32bs[i * 2] > operand1_32bs[i * 2 + 1]:
                        redmax_32b_l0s[i] = operand1_32bs[i * 2]
                    else:
                        redmax_32b_l0s[i] = operand1_32bs[i * 2 + 1]
                redmax_32b_l1s: int32[VLEN // 128]
                for i in range(VLEN // 128):
                    if redmax_32b_l0s[i * 2] > redmax_32b_l0s[i * 2 + 1]:
                        redmax_32b_l1s[i] = redmax_32b_l0s[i * 2]
                    else:
                        redmax_32b_l1s[i] = redmax_32b_l0s[i * 2 + 1]
                redmax_32b_l2_0: int32
                if redmax_32b_l1s[0] > redmax_32b_l1s[1]:
                    redmax_32b_l2_0 = redmax_32b_l1s[0]
                else:
                    redmax_32b_l2_0 = redmax_32b_l1s[1]
                redmax_32b: int32
                if base_32b > redmax_32b_l2_0:
                    redmax_32b = base_32b
                else:
                    redmax_32b = redmax_32b_l2_0

                redsum_8b: int8 = operand1[0:8]
                redsum_16b: int16 = operand1[0:16]
                redsum_32b: int32 = operand1[0:32]

                for i in allo.grid(VLEN // 8, name="reduction_8"):
                    scalar: int8 = operand2[i * 8 : (i + 1) * 8]
                    redsum_8b = redsum_8b + scalar
                for i in allo.grid(VLEN // 16, name="reduction_16"):
                    scalar: int16 = operand2[i * 16 : (i + 1) * 16]
                    redsum_16b = redsum_16b + scalar
                for i in allo.grid(VLEN // 32, name="reduction_32"):
                    scalar: int32 = operand2[i * 32 : (i + 1) * 32]
                    redsum_32b = redsum_32b + scalar

                inst: uint8 = inst_[idx]
                vd: uint256
                # write back
                if inst == VMAX:
                    if ele_type == EEW8:
                        vd = max_8b
                    if ele_type == EEW16:
                        vd = max_16b
                    if ele_type == EEW32:
                        vd = max_32b
                if inst == VMIN:
                    if ele_type == EEW8:
                        vd = min_8b
                    if ele_type == EEW16:
                        vd = min_16b
                    if ele_type == EEW32:
                        vd = min_32b
                if inst == VADD:
                    if ele_type == EEW8:
                        vd = add_8b
                    if ele_type == EEW16:
                        vd = add_16b
                    if ele_type == EEW32:
                        vd = add_32b
                if inst == VSUB:
                    if ele_type == EEW8:
                        vd = sub_8b
                    if ele_type == EEW16:
                        vd = sub_16b
                    if ele_type == EEW32:
                        vd = sub_32b
                if inst == VMUL:
                    if ele_type == EEW8:
                        vd = mul_8b
                    if ele_type == EEW16:
                        vd = mul_16b
                    if ele_type == EEW32:
                        vd = mul_32b
                if inst == VREDMAX:
                    if ele_type == EEW8:
                        vd[0:8] = redmax_8b
                    if ele_type == EEW16:
                        vd[0:16] = redmax_16b
                    if ele_type == EEW32:
                        vd[0:32] = redmax_32b
                if inst == VREDSUM:
                    if ele_type == EEW8:
                        vd[0:8] = redsum_8b
                    if ele_type == EEW16:
                        vd[0:16] = redsum_16b
                    if ele_type == EEW32:
                        vd[0:32] = redsum_32b
                vd_[idx] = vd

    A = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    B = np.random.randint(0, 64, (INST_NUM, VLEN // ELEN)).astype(np.uint32)
    C = np.zeros((INST_NUM, VLEN // ELEN)).astype(np.uint32)
    packed_A = np.ascontiguousarray(A).view(np_256).reshape((INST_NUM,))
    packed_B = np.ascontiguousarray(B).view(np_256).reshape((INST_NUM,))
    packed_C = np.ascontiguousarray(C).view(np_256).reshape((INST_NUM,))
    scalar = np.array([5] * INST_NUM, dtype=np.uint32)
    ele_type = np.array([EEW32] * INST_NUM, dtype=np.uint8)
    mod = df.build(top, target="simulator")

    s = df.customize(top)

    s.pipeline(s.get_loops("VEC_0")["inst_outer"]["idx"])

    if hls.is_available("vitis_hls"):
        print("Starting Test...")
        mod = s.build(
            target="vitis_hls",
            mode="hw_emu",
            project=f"vec.prj",
            wrap_io=False,
        )
        inst = np.array([VMUL] * INST_NUM, dtype=np.uint8)
        vv = np.array([True] * INST_NUM, dtype=np.bool_)
        mod(inst, ele_type, vv, packed_A, packed_B, scalar, packed_C)
        unpacked_C = packed_C.view(np.uint32).reshape(INST_NUM, VLEN // ELEN)
        print(np.multiply(A, B))
        print(unpacked_C)
        np.testing.assert_allclose(np.multiply(A, B), unpacked_C, rtol=1e-5, atol=1e-5)
        print("Passed hw Test!")


if __name__ == "__main__":
    test_vec()
