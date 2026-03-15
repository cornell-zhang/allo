# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import numpy as np
from .handler import BuiltinHandler, register_builtin_handler, TypingRule
from ..config import get_typing_rule_config
from allo._mlir.extras.dialects.affine import AffExpr
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    linalg as linalg_d,
)
from allo._mlir.ir import (
    IntegerType,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    UnitAttr,
    AffineExpr,
    AffineConstantExpr,
)
from allo.ir.types import (
    AlloType,
    Index,
    Float,
    Int,
    UInt,
    Fixed,
    UFixed,
    int32,
    bool as allo_bool,
)

##################################################
# Binary Arithmetic Operations
#
# [NOTE]: the typing rules are not fully tested!!!
##################################################


# =======================================================================
# Default rules
# =======================================================================
def dummy_binary_arith_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Int, UInt): lambda t1, t2: (
            (UInt(t2.bits), UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= t1.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (Int(t1.bits), Int(t1.bits), Int(t1.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (Int, Index): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Int, Float): lambda t1, t2: (
            (t2, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        # python native value
        (Int, int): lambda t1, v2: (t1, t1, t1),
        (int, Int): lambda v1, t2: (t2, t2, t2),
        (Int, float): lambda t1, v2: (Float(64), Float(64), Float(64)),
        (float, Int): lambda v1, t2: (Float(64), Float(64), Float(64)),
        # numpy array
        (Int, np.ndarray): lambda t1, v2: (t1, t1, t1),
        (np.ndarray, Int): lambda v1, t2: (t2, t2, t2),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (
            (UInt(t1.bits), UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= t2.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (Int(t2.bits), Int(t2.bits), Int(t2.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (UInt, UInt): lambda t1, t2: (
            (
                UInt(max(t1.bits, t2.bits)),
                UInt(max(t1.bits, t2.bits)),
                UInt(max(t1.bits, t2.bits)),
            )
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (UInt, Index): lambda t1, t2: (
            (UInt(t1.bits), UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= 32 and t1.bits in {8, 16, 32, 64}
            else (
                (Index(), Index(), Index())
                if t1.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (UInt, Float): lambda t1, t2: (
            (t2, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        # python native value
        (UInt, int): lambda t1, v2: (t1, t1, t1),
        (int, UInt): lambda v1, t2: (t2, t2, t2),
        (UInt, float): lambda t1, v2: (Float(64), Float(64), Float(64)),
        (float, UInt): lambda v1, t2: (Float(64), Float(64), Float(64)),
        # numpy array
        (UInt, np.ndarray): lambda t1, v2: (t1, t1, t1),
        (np.ndarray, UInt): lambda v1, t2: (t2, t2, t2),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Index, UInt): lambda t1, t2: (
            (UInt(t2.bits), UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= 32 and t2.bits in {8, 16, 32, 64}
            else (
                (Index(), Index(), Index())
                if t2.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (Index, Index): lambda t1, t2: (UInt(32), UInt(32), UInt(32)),
        (Index, Float): lambda t1, t2: (t2, t2, t2),
        # python native value
        (Index, int): lambda t1, v2: (UInt(32), UInt(32), UInt(32)),
        (int, Index): lambda v1, t2: (UInt(32), UInt(32), UInt(32)),
        # numpy array
        (Index, np.ndarray): lambda t1, v2: (UInt(32), UInt(32), UInt(32)),
        (np.ndarray, Index): lambda v1, t2: (UInt(32), UInt(32), UInt(32)),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (
            (t1, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Float, UInt): lambda t1, t2: (
            (t1, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Float, Index): lambda t1, t2: (t1, t1, t1),
        (Float, Float): lambda t1, t2: (
            (t1, t1, t1) if t1.bits >= t2.bits else (t2, t2, t2)
        ),
        # python native value
        (Float, int): lambda t1, v2: (t1, t1, t1),
        (int, Float): lambda v1, t2: (t2, t2, t2),
        (Float, float): lambda t1, v2: (t1, t1, t1),
        (float, Float): lambda v1, t2: (t2, t2, t2),
        # numpy array
        (Float, np.ndarray): lambda t1, v2: (t1, t1, t1),
        (np.ndarray, Float): lambda v1, t2: (t2, t2, t2),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules],
    )


DUMMY_BINARY_ARITH_RULE = dummy_binary_arith_rule()


# =======================================================================
# HLS rules
# =======================================================================
def add_sub_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (Int(max(t1.bits, t2.bits) + 1),) * 3,
        (Int, UInt): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1) + 1),) * 3,
        (Int, Index): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1) + 1),) * 3,
        (Int, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (Int, UFixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (Int, Float): lambda t1, t2: (t2,) * 3,
        (Int, int): lambda t1, v2: (Int(max(t1.bits + 1, 32)),) * 3,
        (int, Int): lambda v1, t2: (Int(max(t2.bits + 1, 32)),) * 3,
        (Int, float): lambda t1, v2: (Float(64),) * 3,
        (float, Int): lambda v1, t2: (Float(64),) * 3,
        (Int, np.ndarray): lambda t1, v2: (Int(max(t1.bits + 1, 32)),) * 3,
        (np.ndarray, Int): lambda v1, t2: (Int(max(t2.bits + 1, 32)),) * 3,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits) + 1),) * 3,
        (UInt, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits) + 1),) * 3,
        (UInt, Index): lambda t1, t2: (UInt(max(t1.bits, t2.bits) + 1),) * 3,
        (UInt, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (UInt, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (UInt, Float): lambda t1, t2: (t2,) * 3,
        (UInt, int): lambda t1, v2: (UInt(max(t1.bits + 1, 32)),) * 3,
        (int, UInt): lambda v1, t2: (UInt(max(t2.bits + 1, 32)),) * 3,
        (UInt, float): lambda t1, v2: (Float(64),) * 3,
        (float, UInt): lambda v1, t2: (Float(64),) * 3,
        (UInt, np.ndarray): lambda t1, v2: (UInt(max(t1.bits + 1, 32)),) * 3,
        (np.ndarray, UInt): lambda v1, t2: (UInt(max(t2.bits + 1, 32)),) * 3,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits) + 1),) * 3,
        (Index, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits) + 1),) * 3,
        (Index, Index): lambda t1, t2: (UInt(32),) * 3,
        (Index, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (Index, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs),
        )
        * 3,
        (Index, Float): lambda t1, t2: (t2,) * 3,
        (Index, int): lambda t1, v2: (UInt(33),) * 3,
        (int, Index): lambda v1, t2: (UInt(33),) * 3,
        (Index, np.ndarray): lambda t1, v2: (UInt(33),) * 3,
        (np.ndarray, Index): lambda v1, t2: (UInt(33),) * 3,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (Fixed, UInt): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (Fixed, Index): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (Fixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs)
                + 1,
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (Fixed, UFixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs)
                + 1,
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (Fixed, Float): lambda t1, t2: (t2,) * 3,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (UFixed, UInt): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (UFixed, Index): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs),
        )
        * 3,
        (UFixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs)
                + 1,
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs)
                + 1,
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (UFixed, Float): lambda t1, t2: (t2,) * 3,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1,) * 3,
        (Float, UInt): lambda t1, t2: (t1,) * 3,
        (Float, Index): lambda t1, t2: (t1,) * 3,
        (Float, Fixed): lambda t1, t2: (t1,) * 3,
        (Float, UFixed): lambda t1, t2: (t1,) * 3,
        (Float, Float): lambda t1, t2: ((t1 if t1.bits >= t2.bits else t2),) * 3,
        (Float, int): lambda t1, v2: (t1,) * 3,
        (int, Float): lambda v1, t2: (t2,) * 3,
        (Float, float): lambda t1, v2: (t1,) * 3,
        (float, Float): lambda v1, t2: (t2,) * 3,
        (Float, np.ndarray): lambda t1, v2: (t1,) * 3,
        (np.ndarray, Float): lambda v1, t2: (t2,) * 3,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mul_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (Int(t1.bits + t2.bits),) * 3,
        (Int, UInt): lambda t1, t2: (Int(t1.bits + t2.bits),) * 3,
        (Int, Index): lambda t1, t2: (Int(t1.bits + t2.bits),) * 3,
        (Int, Fixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Int, UFixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Int, Float): lambda t1, t2: (t2,) * 3,
        (Int, int): lambda t1, v2: (Int(t1.bits + 32),) * 3,
        (int, Int): lambda v1, t2: (Int(t2.bits + 32),) * 3,
        (Int, float): lambda t1, v2: (Float(64),) * 3,
        (float, Int): lambda v1, t2: (Float(64),) * 3,
        (Int, np.ndarray): lambda t1, v2: (Int(t1.bits + 32),) * 3,
        (np.ndarray, Int): lambda v1, t2: (Int(t2.bits + 32),) * 3,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(t1.bits + t2.bits),) * 3,
        (UInt, UInt): lambda t1, t2: (UInt(t1.bits + t2.bits),) * 3,
        (UInt, Index): lambda t1, t2: (UInt(t1.bits + t2.bits),) * 3,
        (UInt, Fixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (UInt, UFixed): lambda t1, t2: (
            UFixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (UInt, Float): lambda t1, t2: (t2,) * 3,
        (UInt, int): lambda t1, v2: (UInt(t1.bits + 32),) * 3,
        (int, UInt): lambda v1, t2: (UInt(t2.bits + 32),) * 3,
        (UInt, float): lambda t1, v2: (Float(64),) * 3,
        (float, UInt): lambda v1, t2: (Float(64),) * 3,
        (UInt, np.ndarray): lambda t1, v2: (UInt(t1.bits + 32),) * 3,
        (np.ndarray, UInt): lambda v1, t2: (UInt(t2.bits + 32),) * 3,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(t1.bits + t2.bits),) * 3,
        (Index, UInt): lambda t1, t2: (UInt(t1.bits + t2.bits),) * 3,
        (Index, Index): lambda t1, t2: (UInt(32),) * 3,
        (Index, Fixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Index, UFixed): lambda t1, t2: (
            UFixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Index, Float): lambda t1, t2: (t2,) * 3,
        (Index, int): lambda t1, v2: (UInt(64),) * 3,
        (int, Index): lambda v1, t2: (UInt(64),) * 3,
        (Index, np.ndarray): lambda t1, v2: (UInt(64),) * 3,
        (np.ndarray, Index): lambda v1, t2: (UInt(64),) * 3,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Fixed, UInt): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Fixed, Index): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (Fixed, Fixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),)
        * 3,
        (Fixed, UFixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),)
        * 3,
        (Fixed, Float): lambda t1, t2: (t2,) * 3,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (UFixed, UInt): lambda t1, t2: (
            UFixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (UFixed, Index): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        )
        * 3,
        (UFixed, Fixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),)
        * 3,
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        )
        * 3,
        (UFixed, Float): lambda t1, t2: (t2,) * 3,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1,) * 3,
        (Float, UInt): lambda t1, t2: (t1,) * 3,
        (Float, Index): lambda t1, t2: (t1,) * 3,
        (Float, Fixed): lambda t1, t2: (t1,) * 3,
        (Float, UFixed): lambda t1, t2: (t1,) * 3,
        (Float, Float): lambda t1, t2: ((t1 if t1.bits >= t2.bits else t2),) * 3,
        (Float, int): lambda t1, v2: (t1,) * 3,
        (int, Float): lambda v1, t2: (t2,) * 3,
        (Float, float): lambda t1, v2: (t1,) * 3,
        (float, Float): lambda v1, t2: (t2,) * 3,
        (Float, np.ndarray): lambda t1, v2: (t1,) * 3,
        (np.ndarray, Float): lambda v1, t2: (t2,) * 3,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules]
    )


def div_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (t1,) * 3,
        (Int, UInt): lambda t1, t2: (t1,) * 3,
        (Int, Index): lambda t1, t2: (t1,) * 3,
        (Int, Fixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),)
        * 3,
        (Int, UFixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits + 1, t1.bits - t2.fracs),
        )
        * 3,
        (Int, Float): lambda t1, t2: (t2,) * 3,
        (Int, int): lambda t1, v2: (t1,) * 3,
        (int, Int): lambda v1, t2: (t2,) * 3,
        (Int, float): lambda t1, v2: (Float(64),) * 3,
        (float, Int): lambda v1, t2: (Float(64),) * 3,
        (Int, np.ndarray): lambda t1, v2: (t1,) * 3,
        (np.ndarray, Int): lambda v1, t2: (t2,) * 3,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(t1.bits),) * 3,
        (UInt, UInt): lambda t1, t2: (t1,) * 3,
        (UInt, Index): lambda t1, t2: (t1,) * 3,
        (UInt, Fixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),)
        * 3,
        (UInt, UFixed): lambda t1, t2: (UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),)
        * 3,
        (UInt, Float): lambda t1, t2: (t2,) * 3,
        (UInt, int): lambda t1, v2: (t1,) * 3,
        (int, UInt): lambda v1, t2: (t2,) * 3,
        (UInt, float): lambda t1, v2: (Float(64),) * 3,
        (float, UInt): lambda v1, t2: (Float(64),) * 3,
        (UInt, np.ndarray): lambda t1, v2: (t1,) * 3,
        (np.ndarray, UInt): lambda v1, t2: (t2,) * 3,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(t1.bits),) * 3,
        (Index, UInt): lambda t1, t2: (t1,) * 3,
        (Index, Index): lambda t1, t2: (UInt(32),) * 3,
        (Index, Fixed): lambda t1, t2: (Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),)
        * 3,
        (Index, UFixed): lambda t1, t2: (UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),)
        * 3,
        (Index, Float): lambda t1, t2: (t2,) * 3,
        (Index, int): lambda t1, v2: (t1,) * 3,
        (int, Index): lambda v1, t2: (t2,) * 3,
        (Index, float): lambda t1, v2: (Float(64),) * 3,
        (float, Index): lambda v1, t2: (Float(64),) * 3,
        (Index, np.ndarray): lambda t1, v2: (t1,) * 3,
        (np.ndarray, Index): lambda v1, t2: (t2,) * 3,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (Fixed(t1.bits + t2.bits, t2.bits + t1.fracs),)
        * 3,
        (Fixed, UInt): lambda t1, t2: (
            Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        )
        * 3,
        (Fixed, Index): lambda t1, t2: (
            Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        )
        * 3,
        (Fixed, Fixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs),
        )
        * 3,
        (Fixed, UFixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits + 1, t2.bits - t2.fracs + t1.fracs),
        )
        * 3,
        (Fixed, Float): lambda t1, t2: (t2,) * 3,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        )
        * 3,
        (UFixed, UInt): lambda t1, t2: (UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),)
        * 3,
        (UFixed, Index): lambda t1, t2: (UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),)
        * 3,
        (UFixed, Fixed): lambda t1, t2: (
            Fixed(t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs),
        )
        * 3,
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs),
        )
        * 3,
        (UFixed, Float): lambda t1, t2: (t2,) * 3,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1,) * 3,
        (Float, UInt): lambda t1, t2: (t1,) * 3,
        (Float, Index): lambda t1, t2: (t1,) * 3,
        (Float, Fixed): lambda t1, t2: (t1,) * 3,
        (Float, UFixed): lambda t1, t2: (t1,) * 3,
        (Float, Float): lambda t1, t2: ((t1 if t1.bits >= t2.bits else t2),) * 3,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mod_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (Int(max(t1.bits, t2.bits)),) * 3,
        (Int, UInt): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)),) * 3,
        (Int, Index): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)),) * 3,
        (Int, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        )
        * 3,
        (Int, UFixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs),
        )
        * 3,
        (Int, Float): lambda t1, t2: (t2,) * 3,
        (Int, int): lambda t1, v2: (Int(max(t1.bits, 32)),) * 3,
        (int, Int): lambda v1, t2: (Int(max(t2.bits, 32)),) * 3,
        (Int, float): lambda t1, v2: (Float(64),) * 3,
        (float, Int): lambda v1, t2: (Float(64),) * 3,
        (Int, np.ndarray): lambda t1, v2: (Int(max(t1.bits, 32)),) * 3,
        (np.ndarray, Int): lambda v1, t2: (Int(max(t2.bits, 32)),) * 3,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)),) * 3,
        (UInt, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)),) * 3,
        (UInt, Index): lambda t1, t2: (UInt(max(t1.bits, t2.bits)),) * 3,
        (UInt, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        )
        * 3,
        (UInt, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        )
        * 3,
        (UInt, Float): lambda t1, t2: (t2,) * 3,
        (UInt, int): lambda t1, v2: (UInt(max(t1.bits + 1, 32)),) * 3,
        (int, UInt): lambda v1, t2: (UInt(max(t2.bits + 1, 32)),) * 3,
        (UInt, float): lambda t1, v2: (Float(64),) * 3,
        (float, UInt): lambda v1, t2: (Float(64),) * 3,
        (UInt, np.ndarray): lambda t1, v2: (UInt(max(t1.bits + 1, 32)),) * 3,
        (np.ndarray, UInt): lambda v1, t2: (UInt(max(t2.bits + 1, 32)),) * 3,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)),) * 3,
        (Index, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)),) * 3,
        (Index, Index): lambda t1, t2: (UInt(32),) * 3,
        (Index, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        )
        * 3,
        (Index, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        )
        * 3,
        (Index, Float): lambda t1, t2: (t2,) * 3,
        (Index, int): lambda t1, v2: (Int(max(t1.bits + 1, 32)),) * 3,
        (int, Index): lambda v1, t2: (Int(max(t2.bits + 1, 32)),) * 3,
        (Index, np.ndarray): lambda t1, v2: (Int(max(t1.bits + 1, 32)),) * 3,
        (np.ndarray, Index): lambda v1, t2: (Int(max(t2.bits + 1, 32)),) * 3,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
        )
        * 3,
        (Fixed, UInt): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
        )
        * 3,
        (Fixed, Index): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
        )
        * 3,
        (Fixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (Fixed, UFixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (Fixed, Float): lambda t1, t2: (t2,) * 3,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs),
        )
        * 3,
        (UFixed, UInt): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
        )
        * 3,
        (UFixed, Index): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
        )
        * 3,
        (UFixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
        )
        * 3,
        (UFixed, Float): lambda t1, t2: (t2,) * 3,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1,) * 3,
        (Float, UInt): lambda t1, t2: (t1,) * 3,
        (Float, Index): lambda t1, t2: (t1,) * 3,
        (Float, Fixed): lambda t1, t2: (t1,) * 3,
        (Float, UFixed): lambda t1, t2: (t1,) * 3,
        (Float, Float): lambda t1, t2: ((t1 if t1.bits >= t2.bits else t2),) * 3,
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


HLS_ADD_SUB_RULE = add_sub_rule()
HLS_MUL_RULE = mul_rule()
HLS_DIV_RULE = div_rule()
HLS_MOD_RULE = mod_rule()


def type_compatible(types):
    """
    Helper function to check if all types are compatible. (to match linalg_d op's type constraint)
    """
    if len(types) <= 1:
        return True
    ref = types[0]
    for t in types[1:]:
        if not type(t) is type(ref):
            return False
        if t.element_type != ref.element_type or t.shape != ref.shape:
            return False
    return True


@register_builtin_handler("Add")
class AddHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.AddIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.AddFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.AddFixedOp(left, right, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        else:
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.add failed")
            alloc_op = self.builder.build_buffer(result_type, type_hint)
            with self.builder.get_ip():
                linalg_d.add(left, right, outs=[alloc_op])
            return alloc_op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_ADD_SUB_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r:
            return AffExpr.add(expr_l, expr_r)
        return None


@register_builtin_handler("Sub")
class SubHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.SubIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.SubFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.SubFixedOp(left, right, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        else:
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.sub failed")
            alloc_op = self.builder.build_buffer(result_type, type_hint)
            with self.builder.get_ip():
                linalg_d.sub(left, right, outs=[alloc_op])
            return alloc_op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_ADD_SUB_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r:
            return AffExpr.sub(expr_l, expr_r)
        return None


@register_builtin_handler("Mult")
class MultHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.MulIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.MulFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.MulFixedOp(left, right, ip=self.builder.get_ip())
            op.attributes[type_hint] = UnitAttr.get()
            return op
        else:
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.mul failed")
            alloc_op = self.builder.build_buffer(result_type, type_hint)
            with self.builder.get_ip():
                linalg_d.mul(left, right, outs=[alloc_op])
            return alloc_op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_MUL_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r:
            if isinstance(expr_l, AffineConstantExpr) or isinstance(
                expr_r, AffineConstantExpr
            ):
                return AffExpr.mul(expr_l, expr_r)
        return None


@register_builtin_handler("Div")
class DivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                if type_hint.startswith("u"):
                    op = arith_d.DivUIOp(left, right, ip=self.builder.get_ip())
                else:
                    op = arith_d.DivSIOp(left, right, ip=self.builder.get_ip())
                op.attributes[type_hint] = UnitAttr.get()
                return op
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                return arith_d.DivFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.DivFixedOp(left, right, ip=self.builder.get_ip())
                op.attributes[type_hint] = UnitAttr.get()
                return op
        else:
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.div failed")
            alloc_op = self.builder.build_buffer(result_type, type_hint)
            with self.builder.get_ip():
                linalg_d.div(left, right, outs=[alloc_op])
            return alloc_op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_DIV_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r and isinstance(expr_r, AffineConstantExpr):
            return AffExpr.div(expr_l, expr_r)
        return None


@register_builtin_handler("FloorDiv")
class FloorDivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType) and type_hint.startswith("s"):
                return arith_d.FloorDivSIOp(left, right, ip=self.builder.get_ip())
        raise RuntimeError("not supported")

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_DIV_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r and isinstance(expr_r, AffineConstantExpr):
            return AffExpr.div(expr_l, expr_r)
        return None


@register_builtin_handler("Mod")
class ModHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, type_hint = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                if type_hint.startswith("u"):
                    op = arith_d.RemUIOp(left, right, ip=self.builder.get_ip())
                else:
                    op = arith_d.RemSIOp(left, right, ip=self.builder.get_ip())
                op.attributes[type_hint] = UnitAttr.get()
                return op
            if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                return arith_d.RemFOp(left, right, ip=self.builder.get_ip())
        raise RuntimeError("not supported")

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_BINARY_ARITH_RULE,
            "hls": HLS_MOD_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])

    def get_affine_expr(self, node: ast.Call, ivs: list, symbols: list):
        expr_l = self.builder.get_affine_expr(node.args[0], ivs, symbols)
        expr_r = self.builder.get_affine_expr(node.args[1], ivs, symbols)
        if expr_l and expr_r and isinstance(expr_r, AffineConstantExpr):
            return AffExpr.mod(expr_l, expr_r)
        return None


##################################################
# Binary Comparison Operations
#
# [NOTE]: the typing rules are not fully tested!!!
##################################################


# =======================================================================
# Default rules
# =======================================================================
def dummy_comparison_rule():
    # [NOTE]: the return type is always bool (currently using i1)
    int_rules = {
        (Int, Int): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Int, UInt): lambda t1, t2: (
            (allo_bool, UInt(t2.bits), UInt(t2.bits), "u")
            if t2.bits >= t1.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (allo_bool, Int(t1.bits), Int(t1.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (Int, Index): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Int, Float): lambda t1, t2: (
            (allo_bool, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        # python native value
        (Int, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, Int): lambda v1, t2: (allo_bool, t2, t2),
        (Int, float): lambda t1, v2: (allo_bool, Float(64), Float(64)),
        (float, Int): lambda v1, t2: (allo_bool, Float(64), Float(64)),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (
            (allo_bool, UInt(t1.bits), UInt(t1.bits), "u")
            if t1.bits >= t2.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (allo_bool, Int(t2.bits), Int(t2.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (UInt, UInt): lambda t1, t2: (
            (allo_bool, allo_bool, allo_bool, "u")
            if t1 == t2 == allo_bool
            else (
                (
                    allo_bool,
                    UInt(max(t1.bits, t2.bits)),
                    UInt(max(t1.bits, t2.bits)),
                    "u",
                )
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (UInt, Index): lambda t1, t2: (
            (allo_bool, UInt(t1.bits), UInt(t1.bits), "u")
            if t1.bits >= 32 and t1.bits in {8, 16, 32, 64}
            else (
                (allo_bool, Index(), Index())
                if t1.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (UInt, Float): lambda t1, t2: (
            (allo_bool, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        # python native value
        (UInt, int): lambda t1, v2: (allo_bool, t1, t1, "u"),
        (int, UInt): lambda v1, t2: (allo_bool, t2, t2, "u"),
        (UInt, float): lambda t1, v2: (allo_bool, Float(64), Float(64)),
        (float, UInt): lambda v1, t2: (allo_bool, Float(64), Float(64)),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Index, UInt): lambda t1, t2: (
            (allo_bool, UInt(t2.bits), UInt(t2.bits), "u")
            if t2.bits >= 32 and t2.bits in {8, 16, 32, 64}
            else (
                (allo_bool, Index(), Index())
                if t2.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (Index, Index): lambda t1, t2: (allo_bool, UInt(32), UInt(32)),
        (Index, Float): lambda t1, t2: (allo_bool, t2, t2),
        # python native value
        (Index, int): lambda t1, v2: (allo_bool, UInt(32), UInt(32)),
        (int, Index): lambda v1, t2: (allo_bool, UInt(32), UInt(32)),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (
            (allo_bool, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Float, UInt): lambda t1, t2: (
            (allo_bool, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Float, Index): lambda t1, t2: (allo_bool, t1, t1),
        (Float, Float): lambda t1, t2: (
            (allo_bool, t1, t1) if t1.bits >= t2.bits else (allo_bool, t2, t2)
        ),
        # python native value
        (Float, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, Float): lambda v1, t2: (allo_bool, t2, t2),
        (Float, float): lambda t1, v2: (allo_bool, t1, t1),
        (float, Float): lambda v1, t2: (allo_bool, t2, t2),
    }
    bool_rules = {
        (UInt, bool): lambda t1, v2: (
            (allo_bool, t1, t1, "u")
            if t1.bits == 1
            else TypeError(f"{t1}, {v2} fail binary comparison rule")
        ),
        (bool, UInt): lambda v1, t2: (
            (allo_bool, t2, t2, "u")
            if t2.bits == 1
            else TypeError(f"{v1}, {t2} fail binary comparison rule")
        ),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules, bool_rules],
    )


DUMMY_COMPARISON_RULE = dummy_comparison_rule()


# =======================================================================
# HLS rules
# =======================================================================
def cmp_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (
            allo_bool,
            Int(max(t1.bits, t2.bits)),
            Int(max(t1.bits, t2.bits)),
        ),
        (Int, UInt): lambda t1, t2: (
            allo_bool,
            UInt(max(t1.bits, t2.bits + 1)),
            UInt(max(t1.bits, t2.bits + 1)),
            "u",
        ),
        (Int, Index): lambda t1, t2: (
            allo_bool,
            Int(max(t1.bits, t2.bits + 1)),
            Int(max(t1.bits, t2.bits + 1)),
        ),
        (Int, Fixed): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        ),
        (Int, UFixed): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs),
            UFixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs),
            "u",
        ),
        (Int, Float): lambda t1, t2: (allo_bool, t2, t2),
        (Int, int): lambda t1, v2: (
            allo_bool,
            Int(max(t1.bits, 32)),
            Int(max(t1.bits, 32)),
        ),
        (int, Int): lambda v1, t2: (
            allo_bool,
            Int(max(t2.bits, 32)),
            Int(max(t2.bits, 32)),
        ),
        (Int, np.ndarray): lambda t1, v2: (
            allo_bool,
            Int(max(t1.bits, 32)),
            Int(max(t1.bits, 32)),
        ),
        (np.ndarray, Int): lambda v1, t2: (
            allo_bool,
            Int(max(t2.bits, 32)),
            Int(max(t2.bits, 32)),
        ),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (
            allo_bool,
            UInt(max(t1.bits + 1, t2.bits)),
            UInt(max(t1.bits + 1, t2.bits)),
            "u",
        ),
        (UInt, UInt): lambda t1, t2: (
            allo_bool,
            UInt(max(t1.bits, t2.bits)),
            UInt(max(t1.bits, t2.bits)),
            "u",
        ),
        (UInt, Index): lambda t1, t2: (
            allo_bool,
            UInt(max(t1.bits, t2.bits)),
            UInt(max(t1.bits, t2.bits)),
            "u",
        ),
        (UInt, Fixed): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        ),
        (UInt, UFixed): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            "u",
        ),
        (UInt, Float): lambda t1, t2: (allo_bool, t2, t2),
        (UInt, int): lambda t1, v2: (
            allo_bool,
            UInt(max(t1.bits + 1, 32)),
            UInt(max(t1.bits + 1, 32)),
        ),
        (int, UInt): lambda v1, t2: (
            allo_bool,
            UInt(max(t2.bits + 1, 32)),
            UInt(max(t2.bits + 1, 32)),
        ),
        (UInt, np.ndarray): lambda t1, v2: (
            allo_bool,
            UInt(max(t1.bits + 1, 32)),
            UInt(max(t1.bits + 1, 32)),
        ),
        (np.ndarray, UInt): lambda v1, t2: (
            allo_bool,
            UInt(max(t2.bits + 1, 32)),
            UInt(max(t2.bits + 1, 32)),
        ),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (
            allo_bool,
            Int(max(t1.bits + 1, t2.bits)),
            Int(max(t1.bits + 1, t2.bits)),
        ),
        (Index, UInt): lambda t1, t2: (
            allo_bool,
            UInt(max(t1.bits, t2.bits)),
            UInt(max(t1.bits, t2.bits)),
            "u",
        ),
        (Index, Index): lambda t1, t2: (allo_bool, UInt(32), UInt(32)),
        (Index, Fixed): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
        ),
        (Index, UFixed): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            "u",
        ),
        (Index, Float): lambda t1, t2: (allo_bool, t2, t2),
        (Index, int): lambda t1, v2: (allo_bool, UInt(32), UInt(32)),
        (int, Index): lambda v1, t2: (allo_bool, UInt(32), UInt(32)),
        (Index, np.ndarray): lambda t1, v2: (allo_bool, UInt(32), UInt(32)),
        (np.ndarray, Index): lambda v1, t2: (allo_bool, UInt(32), UInt(32)),
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
        ),
        (Fixed, UInt): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
        ),
        (Fixed, Index): lambda t1, t2: (
            allo_bool,
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
        ),
        (Fixed, Fixed): lambda t1, t2: (
            allo_bool,
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
        ),
        (Fixed, UFixed): lambda t1, t2: (
            allo_bool,
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            "u",
        ),
        (Fixed, Float): lambda t1, t2: (allo_bool, t2, t2),
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs),
            UFixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs),
            "u",
        ),
        (UFixed, UInt): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            "u",
        ),
        (UFixed, Index): lambda t1, t2: (
            allo_bool,
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            "u",
        ),
        (UFixed, Fixed): lambda t1, t2: (
            allo_bool,
            UFixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UFixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            "u",
        ),
        (UFixed, UFixed): lambda t1, t2: (
            allo_bool,
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            "u",
        ),
        (UFixed, Float): lambda t1, t2: (allo_bool, t2, t2),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (allo_bool, t1, t1),
        (Float, UInt): lambda t1, t2: (allo_bool, t1, t1),
        (Float, Index): lambda t1, t2: (allo_bool, t1, t1),
        (Float, Fixed): lambda t1, t2: (allo_bool, t1, t1),
        (Float, UFixed): lambda t1, t2: (allo_bool, t1, t1),
        (Float, Float): lambda t1, t2: (
            allo_bool,
            t1 if t1.bits >= t2.bits else t2,
            t1 if t1.bits >= t2.bits else t2,
        ),
    }
    bool_rules = {
        (UInt, bool): lambda t1, v2: (
            (allo_bool, t1, t1, "u")
            if t1.bits == 1
            else TypeError(f"{t1}, {v2} fail binary comparison rule")
        ),
        (bool, UInt): lambda v1, t2: (
            (allo_bool, t2, t2, "u")
            if t2.bits == 1
            else TypeError(f"{v1}, {t2} fail binary comparison rule")
        ),
    }
    return TypingRule(
        [
            int_rules,
            uint_rules,
            index_rules,
            fixed_rules,
            ufixed_rules,
            float_rules,
            bool_rules,
        ],
    )


HLS_CMP_RULE = cmp_rule()


@register_builtin_handler("Eq")
class EqHandler(BuiltinHandler):
    # - equal (mnemonic: `"eq"`; integer value: `0`)
    # - float equal (`"oeq"`; integer value: `1`)
    # - fixed equal (integer value: `0`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            op = arith_d.CmpIOp(0, left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(1, left, right, ip=self.builder.get_ip())
        # fixed
        op = allo_d.CmpFixedOp(0, left, right, ip=self.builder.get_ip())
        if is_unsigned:
            op.attributes["unsigned"] = UnitAttr.get()
        return op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])


@register_builtin_handler("NotEq")
class NotEqHandler(BuiltinHandler):
    # - not equal (mnemonic: `"ne"`; integer value: `1`)
    # - float not equal (`"one"` ：integer value: `6`)
    # - fixed not equal (integer value: `1`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            op = arith_d.CmpIOp(1, left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(6, left, right, ip=self.builder.get_ip())
        # fixed
        op = allo_d.CmpFixedOp(1, left, right, ip=self.builder.get_ip())
        if is_unsigned:
            op.attributes["unsigned"] = UnitAttr.get()
        return op

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])


# Less than
@register_builtin_handler("Lt")
class LtHandler(BuiltinHandler):
    # - signed less than (mnemonic: `"slt"`; integer value: `2`)
    # - unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    # - float less than (integer value: `4`)
    # - fixed less than (integer value: `2`)
    # - unsigned fixed less than (integer value: `6`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(6, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(2, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(4, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.CmpFixedOp(6, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.CmpFixedOp(2, left, right, ip=self.builder.get_ip())

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])


# Less than or equal
@register_builtin_handler("LtE")
class LtEHandler(BuiltinHandler):
    # - signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    # - unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    # - float less than or equal (integer value: `5`)
    # - fixed less than or equal (integer value: `3`)
    # - unsigned fixed less than or equal (integer value: `7`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(7, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(3, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(5, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.CmpFixedOp(7, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.CmpFixedOp(3, left, right, ip=self.builder.get_ip())

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])


# Greater than
@register_builtin_handler("Gt")
class GtHandler(BuiltinHandler):
    # - signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    # - unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    # - float greater than (integer value: `2`)
    # - fixed greater than (integer value: `4`)
    # - unsigned fixed greater than (integer value: `8`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(8, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(4, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(2, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.CmpFixedOp(8, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.CmpFixedOp(4, left, right, ip=self.builder.get_ip())

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])


# Greater than or equal
@register_builtin_handler("GtE")
class GtEHandler(BuiltinHandler):
    # - signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    # - unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    # - float greater than or equal (integer value: `3`)
    # - fixed greater than or equal (integer value: `5`)
    # - unsigned fixed greater than or equal (integer value: `9`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        is_unsigned = len(args_) == 4  # with extra 'unsigned' annotation "u"
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(9, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(5, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(3, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.CmpFixedOp(9, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.CmpFixedOp(5, left, right, ip=self.builder.get_ip())

    @staticmethod
    def infer(*args):
        rules = {
            "default": DUMMY_COMPARISON_RULE,
            "hls": HLS_CMP_RULE,
        }
        return rules[get_typing_rule_config()](args[0], args[1])
