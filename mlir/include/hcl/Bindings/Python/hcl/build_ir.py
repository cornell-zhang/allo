# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import List

import numpy as np
from hcl_mlir.dialects import affine, arith, builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import math, memref, scf, func, tensor
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

def get_line_number(frame=0):
    fr = sys._getframe(frame + 1) # +1 to ignore this function call
    return (os.path.basename(fr.f_code.co_filename), fr.f_lineno)


class HCLFlags(object):
    def __init__(self):
        self.BUILD_INPLACE = False
        self.BIT_OP = False

    def enable_build_inplace(self):
        self.BUILD_INPLACE = True

    def disable_build_inplace(self):
        self.BUILD_INPLACE = False

    def is_build_inplace(self):
        return self.BUILD_INPLACE

    def reset(self):
        self.BUILD_INPLACE = False


flags = HCLFlags()
enable_build_inplace = flags.enable_build_inplace
disable_build_inplace = flags.disable_build_inplace
is_build_inplace = flags.is_build_inplace
reset_build_inplace = flags.reset


def is_floating_point_type(dtype):
    return isinstance(dtype, (F16Type, F32Type, F64Type))


def get_floating_point_width(dtype):
    if F16Type.isinstance(dtype):
        return 16
    elif F32Type.isinstance(dtype):
        return 32
    elif F64Type.isinstance(dtype):
        return 64


def is_integer_type(dtype):
    return isinstance(dtype, IntegerType)


def is_unsigned_type(dtype):
    return isinstance(dtype, IntegerType) and dtype.is_unsigned


def is_signed_type(dtype):
    return isinstance(dtype, IntegerType) and dtype.is_signless


def is_fixed_type(dtype):
    return isinstance(dtype, (hcl_d.FixedType, hcl_d.UFixedType))


def is_signed_fixed_type(dtype):
    return isinstance(dtype, hcl_d.FixedType)


def is_unsigned_fixed_type(dtype):
    return isinstance(dtype, hcl_d.UFixedType)


def is_index_type(dtype):
    return isinstance(dtype, IndexType)


def is_struct_type(dtype):
    return isinstance(dtype, hcl_d.StructType)


def is_hcl_mlir_type(dtype):
    return (
        is_floating_point_type(dtype) or is_integer_type(
            dtype) or is_fixed_type(dtype)
    )


def get_mlir_type(dtype):
    """
    Get MLIR type from string.
    Note that the returned type is for ExprOp creation intead of ExprOp.build().
    This is because signedness infomation is preserved.
    i.e. "uint8" is returned as unsigned type instead of signless type. 
    @param: dtype: string or MLIR type
    """
    if (
        is_integer_type(dtype)
        or is_floating_point_type(dtype)
        or is_fixed_type(dtype)
        or is_index_type(dtype)
        or is_struct_type(dtype)
    ):
        return dtype
    elif isinstance(dtype, str):
        if dtype[0:5] == "index":
            return IndexType.get()
        elif dtype[0:3] == "int":
            return IntegerType.get_signless(int(dtype[3:]))
        elif dtype[0:4] == "uint":
            return IntegerType.get_unsigned(int(dtype[4:]))
        elif dtype[0:5] == "float":
            if dtype[5:] == "16":
                return F16Type.get()
            elif dtype[5:] == "32":
                return F32Type.get()
            elif dtype[5:] == "64":
                return F64Type.get()
            else:
                raise DTypeError(f"Not supported floating point type: {dtype}")
        elif dtype[0:5] == "fixed":
            strs = dtype[5:].split("_")
            return hcl_d.FixedType.get(int(strs[0]), int(strs[1]))
        elif dtype[0:6] == "ufixed":
            strs = dtype[6:].split("_")
            return hcl_d.UFixedType.get(int(strs[0]), int(strs[1]))
        else:
            raise DTypeError("Unrecognized data type: {}".format(dtype))
    else:
        raise DTypeError(
            "Unrecognized data type format: {} of Type({})".format(
                dtype, type(dtype))
        )


def get_concrete_type(dtype):
    if IntegerType.isinstance(dtype):
        return IntegerType(dtype)
    elif F16Type.isinstance(dtype):
        return F16Type(dtype)
    elif F32Type.isinstance(dtype):
        return F32Type(dtype)
    elif F64Type.isinstance(dtype):
        return F64Type(dtype)
    elif hcl_d.FixedType.isinstance(dtype):
        return hcl_d.FixedType(dtype)
    elif hcl_d.UFixedType.isinstance(dtype):
        return hcl_d.UFixedType(dtype)
    elif hcl_d.StructType.isinstance(dtype):
        return hcl_d.StructType(dtype)
    elif IndexType.isinstance(dtype):
        return IndexType(dtype)
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def get_bitwidth(dtype):
    if IntegerType.isinstance(dtype):
        return dtype.width
    elif F16Type.isinstance(dtype):
        return 16
    elif F32Type.isinstance(dtype):
        return 32
    elif F64Type.isinstance(dtype):
        return 64
    elif hcl_d.FixedType.isinstance(dtype):
        return dtype.width
    elif hcl_d.UFixedType.isinstance(dtype):
        return dtype.width
    elif hcl_d.StructType.isinstance(dtype):
        bitwidth = 0
        for field in dtype.field_types:
            field = get_concrete_type(field)
            bitwidth += get_bitwidth(field)
        return bitwidth
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def print_mlir_type(dtype):
    """ Print MLIR type to C/HLSC types
    @param dtype: MLIR type
    """
    if is_floating_point_type(dtype):
        if dtype.width == 32:
            return "float"
        elif dtype.width == 64:
            return "double"
        else:
            raise DTypeError("Not supported data type: {}".format(dtype))
    elif is_integer_type(dtype):
        if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
            if dtype.width == 32:
                return "int"
            elif dtype.width == 64:
                return "long int"
            elif dtype.width == 1:
                return "bool"
            else:
                return "ap_int<{}>".format(dtype.width)
        elif dtype.is_unsigned:
            if dtype.width == 32:
                return "unsigned int"
            elif dtype.width == 64:
                return "unsigned long int"
            elif dtype.width == 1:
                return "bool"
            else:
                return "ap_uint<{}>".format(dtype.width)
    elif is_fixed_type(dtype):
        if isinstance(dtype, hcl_d.FixedType):
            return "ap_fixed<{}, {}>".format(dtype.width, dtype.frac)
        elif isinstance(dtype, hcl_d.UFixedType):
            return "ap_ufixed<{}, {}>".format(dtype.width, dtype.frac)
        else:
            raise DTypeError("Not supported data type: {}".format(dtype))
    elif is_struct_type(dtype):
        raise HCLNotImplementedError("struct type printing to be implemented")
    else:
        raise DTypeError("Not supported data type: {}".format(dtype))


def mlir_type_to_str(dtype):
    """ Build HeteroCL-compatible type string from MLIR type
    @param dtype: MLIR type
    """
    if is_signed_type(dtype):
        return "int{}".format(get_bitwidth(dtype))
    elif is_unsigned_type(dtype):
        return "uint{}".format(get_bitwidth(dtype))
    elif is_floating_point_type(dtype):
        return "float{}".format(get_bitwidth(dtype))
    elif is_signed_fixed_type(dtype):
        if dtype.frac == 0:
            return "int{}".format(dtype.width)
        return "fixed{}_{}".format(dtype.width, dtype.frac)
    elif is_unsigned_fixed_type(dtype):
        if dtype.frac == 0:
            return "uint{}".format(dtype.width)
        return "ufixed{}_{}".format(dtype.width, dtype.frac)
    elif is_struct_type(dtype):
        type_str = "Struct("
        for ft in dtype.field_types:
            type_str += mlir_type_to_str(ft) + ", "
        type_str = type_str[:-2] + ")"
        return type_str
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def get_signless_type(dtype):
    if is_integer_type(dtype):
        return IntegerType.get_signless(get_bitwidth(dtype))
    elif is_struct_type(dtype):
        new_field_types = []
        for field_type in dtype.field_types:
            field_type = get_concrete_type(field_type)
            if is_integer_type(field_type):
                new_field_types.append(
                    get_signless_type(field_type)
                )
            elif is_struct_type(field_type):
                new_field_types.append(get_signless_type(field_type))
            else:
                new_field_types.append(field_type)
        dtype = hcl_d.StructType.get(new_field_types)
        return dtype
    else:
        return dtype

def is_all_field_int(dtype):
    """ Check if a struct type has all integer fields
    """
    if not is_struct_type(dtype):
        return False
    dtype = get_concrete_type(dtype)
    for field_type in dtype.field_types:
        field_type = get_concrete_type(field_type)
        if is_struct_type(field_type):
            if not is_all_field_int(field_type):
                return False
        elif not is_integer_type(field_type):
            return False
    return True

class HCLMLIRInsertionPoint(object):
    def __init__(self):
        self.ip_stack = []

    def clear(self):
        self.ip_stack = []

    def get(self):
        return self.ip_stack[-1]

    def get_global(self):
        return self.ip_stack[0]

    def save(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def restore(self):
        return self.ip_stack.pop()


GlobalInsertionPoint = HCLMLIRInsertionPoint()


def floating_point_error(op_name):
    return DTypeError("{} does not support floating point inputs".format(op_name))


def get_hcl_op(expr, dtype=None):
    if isinstance(expr, (int, float)):
        if dtype == None:
            if isinstance(expr, int):
                if expr < 0xFFFFFFFF:
                    return ConstantOp(IntegerType.get_signless(32), expr)
                else:
                    return ConstantOp(IntegerType.get_signless(64), expr)
            elif isinstance(expr, float):
                return ConstantOp(F32Type.get(), expr)
        else:
            return ConstantOp(dtype, expr)
    else:
        if dtype != None and dtype != expr.dtype:
            expr = CastOp(expr, dtype)
        return expr


def get_type_rank(dtype):
    """
    We always cast lower rank types to higher rank types.
    Base rank 1 (lowest): integer and fixed point types
    Base rank 2: index type
    Base rank 3 (highest): float types
    Types with larger dynamic range should have higher ranks.
    """
    if is_integer_type(dtype):
        base = 0
        width = dtype.width
        if width > 2048:
            raise DTypeError("Cannot support integer width larger than 2048")
        base += width
        return base
    elif is_fixed_type(dtype):
        base = 0
        width = dtype.width
        frac = dtype.frac
        return base + (width - frac)
    elif is_index_type(dtype):  # width 32
        base = 2049
        return base
    elif is_floating_point_type(dtype):
        base = 10000
        if isinstance(dtype, F16Type):
            base += 1
        elif isinstance(dtype, F32Type):
            base += 2
        elif isinstance(dtype, F64Type):
            base += 3
        else:
            raise DTypeError(
                "Unrecognized floating point type: {}".format(dtype))
        return base
    else:
        raise DTypeError("Unrecognized type: {}".format(dtype))


def cast_types(lhs, rhs):
    """
    Cast types for binary operations
    lhs always has higher rank than rhs
    Implementation based on
    https://en.cppreference.com/w/c/language/conversion
    """
    ltype = lhs.dtype
    rtype = rhs.dtype
    # 1) If one operand is long double (omitted)
    # 2) Otherwise, if lhs is double
    if isinstance(ltype, F64Type):
        # integer or real floating type to double
        res_type = F64Type.get()
        DTypeWarning("Casting value {} from {} to {}".format(
            rhs, rtype, res_type)).log()
        return lhs, CastOp(rhs, res_type)
    # 3) Otherwise, if lhs is float
    elif isinstance(ltype, F32Type):
        # integer type to float
        res_type = F32Type.get()
        DTypeWarning("Casting value {} from {} to {}".format(
            rhs, rtype, res_type)).log()
        return lhs, CastOp(rhs, res_type)
    # 4) Otherwise, if lhs is integer.
    elif isinstance(ltype, (IntegerType, IndexType)):
        # 4.1) lhs is int or index, rhs is int of lower rank, rhs gets promoted
        if isinstance(rtype, IntegerType):
            res_type = ltype
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return lhs, CastOp(rhs, res_type)
        # 4.2) lhs is index, rhs is also index, nothing to do
        elif isinstance(rtype, IndexType):
            return lhs, rhs
        # 4.3) lhs is int or index, rhs is fixed point of lower rank
        # e.g. Int(100) + Fixed(3, 2) -> Fixed(100 + 2, 2)
        elif is_signed_fixed_type(rtype):
            res_type = hcl_d.FixedType.get(
                ltype.width + rtype.frac, rtype.frac)
            DTypeWarning("Casting value {} from {} to {}".format(
                lhs, ltype, res_type)).log()
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return CastOp(lhs, res_type), CastOp(rhs, res_type)
        # 4.4) lhs is int or index, rhs is unsigned fixed point of lower rank
        # e.g. Int(100) + UFixed(3, 2) -> UFixed(100 + 2, 2)
        elif is_unsigned_fixed_type(rtype):
            res_type = hcl_d.UFixedType.get(
                ltype.width + rtype.frac, rtype.frac)
            DTypeWarning("Casting value {} from {} to {}".format(
                lhs, ltype, res_type)).log()
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return CastOp(lhs, res_type), CastOp(rhs, res_type)
        else:
            # unexpected type
            raise DTypeError("Unexpected type: {}".format(rtype))
    # 5) Otherwise, if lhs is fixed type.
    elif is_fixed_type(ltype):
        # 5.1) lhs is fixed point, rhs is integer or fixed point of lower rank, cast rhs to lhs
        if is_integer_type(rtype) or is_fixed_type(rtype):
            res_type = ltype
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return lhs, CastOp(rhs, res_type)
        else:
            # unexpected type
            raise DTypeError("Unexpected type: {}".format(rtype))
    else:
        raise DTypeError(
            "Type conversion failed, lhs type: {}, rhs type: {}".format(
                ltype, rtype)
        )


# TODO(Niansong): this should be covered by cast_types, double-check before removing
def regularize_fixed_type(lhs, rhs):
    if not is_fixed_type(lhs.dtype) or not is_fixed_type(rhs.dtype):
        raise DTypeError("Should be all fixed types")
    if not lhs.dtype.frac == rhs.dtype.frac:
        raise DTypeError("Should have the same frac")
    lwidth = lhs.dtype.width
    rwidth = rhs.dtype.width
    if lwidth < rwidth:
        res_type = hcl_d.FixedType.get(rwidth, lhs.dtype.frac)
        cast = CastOp(lhs, res_type)
        return cast, rhs
    elif lwidth > rwidth:
        res_type = hcl_d.FixedType.get(lwidth, rhs.dtype.frac)
        cast = CastOp(rhs, res_type)
        return lhs, cast
    else:
        return lhs, rhs


class ExprOp(object):
    def __init__(self, op, dtype=None, loc=None):
        self.op = op
        self.dtype = dtype
        self.built_op = None
        self.loc = loc

    @property
    def result(self):  # get_op_result_or_value
        if isinstance(self.built_op, BlockArgument):
            return self.built_op
        else:
            return self.built_op.result

    @staticmethod
    def generic_op(OpClass, lhs, rhs, arg=None, loc=None):
        if (hasattr(lhs, "v") and lhs.v is not None) or (
            hasattr(rhs, "v") and rhs.v is not None
        ):
            raise APIError(
                "Cannot use hcl.scalar to construct expression, "
                + "use hcl.scalar.v instead"
            )
        # turn py builtin op to hcl op
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)

        # type checking & conversion
        # get_type_rank has the valid type checking
        lhs.dtype = get_mlir_type(lhs.dtype)
        rhs.dtype = get_mlir_type(rhs.dtype)
        if lhs.dtype != rhs.dtype:
            lrank = get_type_rank(lhs.dtype)
            rrank = get_type_rank(rhs.dtype)
            # always ensure the first op has higher ranking
            if lrank > rrank:
                lhs, rhs = cast_types(lhs, rhs)
            else:
                rhs, lhs = cast_types(rhs, lhs)
            if is_fixed_type(lhs.dtype) or is_fixed_type(rhs.dtype):
                lhs, rhs = regularize_fixed_type(lhs, rhs)

        # create AST node based on different types
        dtype = lhs.dtype
        if arg == None:
            expr = OpClass(dtype, lhs, rhs, loc)
        else:
            expr = OpClass(lhs, rhs, arg, loc)
        return expr

    @staticmethod
    def generic_integer_op(OpClass, lhs, rhs):
        # turn py builtin op to hcl op
        lhs = get_hcl_op(lhs)
        rhs = get_hcl_op(rhs)

        # type checking & conversion
        if lhs.dtype != rhs.dtype:
            rhs = CastOp(rhs, lhs.dtype)
        expr = OpClass(lhs, rhs)
        return expr

    @staticmethod
    def generic_scalar_tensor_access(scalar):
        # check scalar shape
        if scalar.shape != (1,):
            raise TensorError(
                "Scalar should be 1D: got {} instead".format(scalar.shape)
            )
        return scalar[0]

    def __add__(self, other):
        # set filename and linenumber
        fn, ln = get_line_number(1)
        loc = Location.file(fn, ln, 0)
        return self.generic_op(AddOp, self, other, loc=loc)

    def __radd__(self, other):
        # if other is an hcl.scalar
        if hasattr(other, "op") and isinstance(other.op, TensorOp):
            other = self.generic_scalar_tensor_access(other)
        return self.generic_op(AddOp, other, self)

    def __sub__(self, other):
        return self.generic_op(SubOp, self, other)

    def __rsub__(self, other):
        return self.generic_op(SubOp, other, self)

    def __mul__(self, other):
        return self.generic_op(MulOp, self, other)

    def __rmul__(self, other):
        return self.generic_op(MulOp, other, self)

    def __div__(self, other):
        return self.generic_op(DivOp, self, other)

    def __rdiv__(self, other):
        return self.generic_op(DivOp, other, self)

    def __truediv__(self, other):
        return self.generic_op(DivOp, self, other)

    def __rtruediv__(self, other):
        return self.generic_op(DivOp, other, self)

    def __floordiv__(self, other):
        return self.generic_op(FloorDivOp, self, other)

    def __rfloordiv__(self, other):
        return self.generic_op(FloorDivOp, other, self)

    def __mod__(self, other):
        return self.generic_op(RemOp, self, other)

    def __neg__(self):
        if is_integer_type(self.dtype):
            return self.generic_op(MulOp, self, -1)
        return NegOp(self)

    def __lshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Left shift")
        return self.generic_integer_op(LeftShiftOp, self, other)

    def __rshift__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Right shift")
        return self.generic_integer_op(RightShiftOp, self, other)

    def __and__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise And")
        return self.generic_integer_op(AndOp, self, other)

    def __or__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise Or")
        return self.generic_integer_op(OrOp, self, other)

    def __xor__(self, other):
        if isinstance(self, float) or isinstance(other, float):
            raise floating_point_error("Bitwise XOr")
        return self.generic_integer_op(XOrOp, self, other)

    def __invert__(self):
        raise HCLNotImplementedError("__invert__ is not implemented")

    def __lt__(self, other):
        return self.generic_op(CmpOp, self, other, arg="lt")

    def __le__(self, other):
        return self.generic_op(CmpOp, self, other, arg="le")

    def __eq__(self, other):
        # In MLIR's auto-generated python code, there are cases where
        # `Expr == None` is used to check if the expression is None.
        # so we add this short circuit here.
        if other is None:
            return False
        return self.generic_op(CmpOp, self, other, arg="eq")

    def __ne__(self, other):
        # In MLIR's auto-generated python code, there are cases where
        # `Expr != None` is used to check if the expression is None.
        # so we add this short circuit here.
        if other is None:
            return True
        return self.generic_op(CmpOp, self, other, arg="ne")

    def __gt__(self, other):
        return self.generic_op(CmpOp, self, other, arg="gt")

    def __ge__(self, other):
        return self.generic_op(CmpOp, self, other, arg="ge")

    def __getitem__(self, indices):
        if not is_integer_type(self.dtype):
            raise APIError("Only integers can access the bits")
        if isinstance(indices, slice):
            lo, hi = indices.start, indices.stop
            if isinstance(lo, int) and isinstance(hi, int):
                if lo > hi:
                    raise APIError(
                        "Lower bound should be smaller than upper bound. Use `.reverse()` if you want to reverse the bits"
                    )
                elif lo == hi:
                    return self
                else:
                    return GetSliceOp(self, hi - 1, lo)
            else:
                return GetSliceOp(self, hi - 1, lo)
        else:
            if not isinstance(indices, tuple):
                indices = (indices,)
            if not len(indices) == 1:
                raise APIError("Can only access one bit of the integer")
            index = indices[0]
            return GetBitOp(self, index)

    def __setitem__(self, indices, expr):
        if not is_integer_type(self.dtype):
            raise APIError("Only integers can access the bits")
        if isinstance(indices, slice):
            lo, hi = indices.start, indices.stop
            if isinstance(lo, int) and isinstance(hi, int):
                if lo > hi:
                    raise APIError(
                        "Lower bound should be smaller than upper bound. Use `.reverse()` if you want to reverse the bits"
                    )
                elif lo == hi:  # e.g. [2:2]
                    if not isinstance(expr, LoadOp):
                        raise APIError(
                            "Please check the expression to make sure the lower bound not equal to the upper bound"
                        )
                    else:
                        return StoreOp(expr, self.tensor, self.indices)
                else:
                    return SetSliceOp(self, hi - 1, lo, expr)
            else:
                return SetSliceOp(self, hi - 1, lo, expr)
        else:
            if not isinstance(indices, tuple):
                indices = (indices,)
            if not len(indices) == 1:
                raise APIError("Can only access one bit of the integer")
            indices = indices[0]
            return SetBitOp(self, indices, expr)

    def reverse(self):
        if not is_integer_type(self.dtype):
            raise APIError("Only integers can reverse the bits")
        return BitReverseOp(self)

    def __nonzero__(self):
        raise APIError(
            "1) Cannot use and / or / not operator to Expr, "
            + "2) Cannot compare NumPy numbers with HeteroCL exprs, "
            + "hint: swap the operands"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other):
        """Build an equal check expression with other expr.

        Parameters
        ----------
        other : Expr
            The other expression

        Returns
        -------
        ret : Expr
            The equality expression.
        """
        return self.generic_op(self, other, arg="eq")

    def astype(self, dtype):
        """Cast the expression to other type.

        Parameters
        ----------
        dtype : str
            The type of new expression

        Returns
        -------
        expr : Expr
            Expression with new type
        """
        raise HCLNotImplementedError("astype is not implemented")

    def __getattr__(self, key):
        """Access a field from a struct value.

        Parameters
        ----------
        name : str
            The field name

        Returns
        -------
        expr : Expr
            The field expression
        """
        # bypass the attribute lookup to avoid infinite recursion
        if key in self.__dict__.keys():
            return self.__dict__[key]
        elif key == "result":
            if self.built_op is None:
                self.build()
            return self.result
        elif isinstance(self, LoadOp):
            # access a field from a struct tensor
            key_list = [k for k in self.tensor.hcl_dtype.dtype_dict.keys()]
            if key not in key_list:
                raise HCLValueError("No such field: " + key)
            key_idx = key_list.index(key)
            return StructGetOp(self, key_idx)
        else:
            # We don't throw an error here
            # because the user may be trying to test if
            # an attribute exists with hasattr().
            return


#################################################
#
# AST leaf nodes
#
#################################################


class IterVar(ExprOp):
    """loop induction variable (BlockArgument)"""

    def __init__(self, op, name="", loc=None):
        super().__init__(op, dtype="index")
        self.name = name
        self.built_op = op

    def update_op(self, op):
        self.op = op
        self.built_op = op


class ReduceVar(IterVar):
    """reduce_axis
    induction variable of reduction loop
    """

    def __init__(self, op, bound=None, name="", loc=None):
        super().__init__(op, name)
        self.bound = bound

    @property
    def lower_bound(self):
        return self.bound[0]

    @property
    def upper_bound(self):
        return self.bound[1]


class ConstantOp(ExprOp):
    # TODO(Niansong): Needs a robust way to handle overflow
    """
    Constant tensor is implemented as global memref in MLIR.
    To support anywidth integer and fixed point numbers,
    we use i64 global memref to represent the constant.
    When the constant is to be consumed, we cast it to the
    target width.
    """

    def __init__(self, dtype, val, name="const_tensor", loc=None):
        super().__init__(arith.ConstantOp)
        self.val = val
        self.name = name
        self.dtype = get_mlir_type(dtype)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if isinstance(self.val, (List, np.ndarray)):
            # val is numpy ndarray
            if is_integer_type(self.dtype):
                if self.dtype.width <= 64:
                    np_dtype = np.int64
                else:
                    raise DTypeError(
                        "Integer width ({}) too large, not supported by numpy".format(
                            self.dtype
                        )
                    )
            elif is_floating_point_type(self.dtype):
                if isinstance(self.dtype, F16Type):
                    np_dtype = np.float16
                elif isinstance(self.dtype, F32Type):
                    np_dtype = np.float32
                elif isinstance(self.dtype, F64Type):
                    np_dtype = np.float64
                else:
                    raise DTypeError("Unrecognized data type")
            elif is_fixed_type(self.dtype):  # Fixed point
                if is_signed_type(self.dtype):
                    sb = 1 << self.dtype.width
                    sb_limit = 1 << (self.dtype.width - 1)
                    self.val = self.val * (2 ** self.dtype.frac)
                    self.val = np.fix(self.val) % sb

                    def cast_func(x):
                        return x if x < sb_limit else x - sb

                    self.val = np.vectorize(cast_func)(self.val)
                else:
                    sb = 1 << self.dtype.width
                    self.val = self.val * (2 ** self.dtype.frac)
                    self.val = np.fix(self.val) % sb
                np_dtype = np.int64
            else:
                raise DTypeError(
                    "Unrecognized data type: {}".format(self.dtype))

            self.val = np.array(self.val, dtype=np_dtype)
            if is_integer_type(self.dtype) or is_fixed_type(self.dtype):
                dtype = IntegerType.get_signless(64)
            else:  # floating point
                dtype = self.dtype
            value_attr = DenseElementsAttr.get(self.val, type=dtype)
            sym_name = StringAttr.get(self.name)
            sym_visibility = StringAttr.get("private")
            memref_type = MemRefType.get(self.val.shape, dtype)
            type_attr = TypeAttr.get(memref_type)
            const_tensor = memref.GlobalOp(
                sym_name,
                type_attr,
                sym_visibility=sym_visibility,
                initial_value=value_attr,
                constant=True,
                alignment=None,
                ip=GlobalInsertionPoint.get_global(),
            )
            const_tensor.attributes["constant"] = UnitAttr.get()
            if is_unsigned_type(self.dtype):
                const_tensor.attributes["unsigned"] = UnitAttr.get()

            if is_fixed_type(self.dtype):
                tensor_wrapper = TensorOp(
                    self.val.shape, memref.AllocOp, self.dtype, "const_tensor"
                )
                tensor_wrapper.build()
                self.tensor = tensor_wrapper
                fixed_memref_type = MemRefType.get(self.val.shape, self.dtype)
                store = hcl_d.GetGlobalFixedOp(
                    fixed_memref_type,
                    FlatSymbolRefAttr.get(self.name),
                    ip=GlobalInsertionPoint.get(),
                )
            else:
                tensor_wrapper = TensorOp(
                    self.val.shape, memref.AllocOp, dtype, "const_tensor"
                )
                tensor_wrapper.build()
                self.tensor = tensor_wrapper
                store = memref.GetGlobalOp(
                    memref_type,
                    FlatSymbolRefAttr.get(self.name),
                    ip=GlobalInsertionPoint.get(),
                )
            # Note: Why do we have an update_op here?
            # memref.GetGlobalOp is not subscriptable,
            # meaning that we can't do something like
            # const_tensor[x] on it, so that we need to
            # create a tensor wrapper to do that.
            # Since tensor_wrapper only allows allocOp or
            # block arg as implementation, we just build
            # and AllocOp and then set the tensor's build_op
            # as memref.GetGlobalOp.
            # This way we end up with an extra memref.AllocOp
            # in the IR, but it's easy to remove with DCE.
            self.tensor.update_op(store)
            self.built_op = store
            return self.built_op
        else:  # val is not a numpy ndarray, it's a scalar
            # Int and Float
            if not is_fixed_type(self.dtype):
                if isinstance(self.dtype, IntegerType):
                    if self.dtype.width == 1:
                        value_attr = BoolAttr.get(self.val)
                    else:
                        if self.val == 0xFFFFFFFFFFFFFFFF:
                            attr_type = IntegerType.get_signless(
                                self.dtype.width)
                            self.val = -1
                        else:
                            attr_type = IntegerType.get_signless(
                                self.dtype.width)
                        value_attr = IntegerAttr.get(attr_type, self.val)
                elif isinstance(self.dtype, F16Type):
                    value_attr = FloatAttr.get(F16Type.get(), self.val)
                elif isinstance(self.dtype, F32Type):
                    value_attr = FloatAttr.get(F32Type.get(), self.val)
                elif isinstance(self.dtype, F64Type):
                    value_attr = FloatAttr.get(F64Type.get(), self.val)
                elif isinstance(self.dtype, IndexType):
                    value_attr = IntegerAttr.get(IndexType.get(), self.val)
                else:
                    raise DTypeError(
                        "Type error: unrecognized type: " + str(self.dtype)
                    )
                if is_unsigned_type(self.dtype):
                    dtype = IntegerType.get_signless(self.dtype.width)
                    self.built_op = self.op(
                        dtype, value_attr, ip=GlobalInsertionPoint.get()
                    )
                    self.built_op.attributes["unsigned"] = UnitAttr.get()
                else:
                    self.built_op = self.op(
                        self.dtype, value_attr, ip=GlobalInsertionPoint.get()
                    )
                return self.built_op
            else:  # fixed types
                self.val *= 2 ** self.dtype.frac
                self.val %= 2 ** self.dtype.width
                value_attr = IntegerAttr.get(
                    IntegerType.get_signless(self.dtype.width), self.val
                )
                self.built_op = self.op(
                    IntegerType.get_signless(64),
                    value_attr,
                    ip=GlobalInsertionPoint.get(),
                )
                return self.built_op


class TensorSlice(ExprOp):
    def __init__(self, full_shape, op, dtype, parent, indices, name=None, loc=None):
        super().__init__(op)
        self.op = op
        self.full_shape = full_shape
        self.dtype = dtype
        self.name = name
        self.parent = parent
        self.indices = indices
        # calculate tensor slice shape
        shape = list()
        dims = 0
        for index in indices:
            if isinstance(index, int):
                dims += 1
            elif isinstance(index, slice):
                step = index.step if index.step is not None else 1
                dim_size = (index.stop - index.start) / step
                shape.append(int(dim_size))
                dims += 1
            # index is an expr
            elif isinstance(index, ExprOp):
                if not hasattr(index, "dtype"):
                    raise HCLValueError("{} doesn't have dtype".format(index))
                if not (is_integer_type(index.dtype) or isinstance(index, IterVar)): 
                    raise HCLValueError("{} is not an integer type or index type".format(index))
                dims += 1
        for i, dim in enumerate(self.full_shape):
            if i < dims:
                continue
            shape.append(dim)
        self.shape = tuple(shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(self.indices + indices) < len(self.full_shape):
            return TensorSlice(
                self.full_shape,
                self.op,
                self.dtype,
                self.parent,
                self.indices + indices,
                self.name,
            )
        elif len(self.indices + indices) == len(self.full_shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            load = LoadOp(self.parent, new_indices)
            return load
        else:
            raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(self.indices + indices) < len(self.full_shape):
            # TODO(Niansong): I think this is doable actually
            raise HCLNotImplementedError(
                "Writing to a slice of tensor is not allowed.")
        elif len(self.indices + indices) == len(self.full_shape):
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            return StoreOp(expr, self.parent, list(self.indices) + new_indices)
        else:
            raise TensorError("Indices length > # of array dimensions," \
                + f"indices=[{self.indices + indices}], shape={self.full_shape}")


class TensorOp(ExprOp):
    def __init__(self, shape, op, dtype, name=None, loc=None):
        if op != memref.AllocOp and not isinstance(op, BlockArgument):
            raise TensorError("Not supported TensorOp. Got {}".format(op))
        super().__init__(op)
        self.shape = shape
        self.dtype = dtype
        self.hcl_dtype = dtype
        self.name = name

    def build(self):
        if self.op == memref.AllocOp:
            self.built_op = self.op(
                self.memref_type, [], [], ip=GlobalInsertionPoint.get()
            )
            if is_unsigned_type(self.dtype):
                self.built_op.attributes["unsigned"] = UnitAttr.get()
            self.built_op.attributes["name"] = StringAttr.get(self.name)
        elif isinstance(self.op, BlockArgument):
            self.built_op = self.op
        else:
            raise TensorError(
                "TensorOp should use memref.AllocOp or BlockArgument to implement. Got {}".format(
                    self.op
                )
            )
        return self.built_op

    def update_op(self, op):
        self.built_op = op

    @property
    def memref_type(self):
        if is_struct_type(self.dtype):
            # Replace unsigned field types with signless types
            dtype = get_signless_type(self.dtype)
            return MemRefType.get(self.shape, dtype)
        dtype = get_mlir_type(self.dtype)
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
        else:
            dtype = self.dtype
        return MemRefType.get(self.shape, dtype)

    def set_axis(self, _axis):
        self._axis = _axis

    @property
    def axis(self):
        return self._axis

    @property
    def v(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self.__getitem__(0)
        else:
            raise TensorError(".v can only be used on mutable scalars")

    @v.setter
    def v(self, value):
        """A syntactic sugar for setting the value of a single-element tensor.
        This is the same as using `a[0]=value`, where a is a single-element tensor.
        Parameters
        ----------
        value : Expr
            The value to be set
        """
        self.__setitem__(0, value)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # if we are slicing tensor
        if len(indices) < len(self.shape):
            return TensorSlice(
                self.shape, self.op, self.dtype, self, indices, self.name
            )
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            load = LoadOp(self, new_indices)
            # if flags.BUILD_INPLACE:
            #     load.build()
            return load
        else:
            raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) < len(self.shape):
            # TODO(Niansong): I think this is doable actually
            raise HCLNotImplementedError(
                "Writing to a slice of tensor is not allowed.")
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            expr = get_hcl_op(expr)
            return StoreOp(expr, self, new_indices)
        else:
            raise TensorError("Indices length > # of array dimensions")


#################################################
#
# AST inner nodes
#
#################################################


class UnaryOp(ExprOp):
    def __init__(self, op, dtype, val, loc=None):
        super().__init__(op)
        self.dtype = dtype
        self.val = val
        if isinstance(op, dict):
            if is_integer_type(dtype):
                self.op = op["int"]
            elif is_floating_point_type(dtype):
                self.op = op["float"]
            elif is_fixed_type(dtype):
                self.op = op["fixed"]
            else:
                raise DTypeError(
                    "Unsupported types for unary op: {}".format(dtype))
        else:
            self.op = op
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(self.val.result, ip=GlobalInsertionPoint.get())
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class BinaryOp(ExprOp):
    def __init__(self, op, dtype, lhs, rhs, loc=None):
        super().__init__(op)
        self.dtype = dtype
        self.lhs = lhs
        self.rhs = rhs
        self.loc = loc
        if isinstance(op, dict):
            if is_integer_type(dtype) or is_index_type(dtype):
                self.op = op["int"]
            elif is_floating_point_type(dtype):
                self.op = op["float"]
            elif is_fixed_type(dtype):
                self.op = op["fixed"]
            else:
                raise DTypeError(
                    "Unsupported types for binary op: {}".format(dtype))
        else:
            self.op = op
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.lhs.result, self.rhs.result, ip=GlobalInsertionPoint.get(),
            loc=self.loc
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class CmpOp(BinaryOp):
    """
    # Check mlir/Dialect/Arithmetic/IR/ArithmeticBase.td
    # s/u: signed/unsigned
    # o/u: ordered/unordered
    #      ordered means only one of < = > cases is true
    #      unordered happens for floating points with NaN
    // Opcode              U L G E    Intuitive operation
    FCMP_FALSE =  0,  ///< 0 0 0 0    Always false (always folded)
    FCMP_OEQ   =  1,  ///< 0 0 0 1    True if ordered and equal
    FCMP_OGT   =  2,  ///< 0 0 1 0    True if ordered and greater than
    FCMP_OGE   =  3,  ///< 0 0 1 1    True if ordered and greater than or equal
    FCMP_OLT   =  4,  ///< 0 1 0 0    True if ordered and less than
    FCMP_OLE   =  5,  ///< 0 1 0 1    True if ordered and less than or equal
    FCMP_ONE   =  6,  ///< 0 1 1 0    True if ordered and operands are unequal
    FCMP_ORD   =  7,  ///< 0 1 1 1    True if ordered (no nans)
    FCMP_UNO   =  8,  ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    FCMP_UEQ   =  9,  ///< 1 0 0 1    True if unordered or equal
    FCMP_UGT   = 10,  ///< 1 0 1 0    True if unordered or greater than
    FCMP_UGE   = 11,  ///< 1 0 1 1    True if unordered, greater than, or equal
    FCMP_ULT   = 12,  ///< 1 1 0 0    True if unordered or less than
    FCMP_ULE   = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
    FCMP_UNE   = 14,  ///< 1 1 1 0    True if unordered or not equal
    FCMP_TRUE  = 15,  ///< 1 1 1 1    Always true (always folded)
    """

    ATTR_MAP = {
        "int": {
            "eq": 0,
            "ne": 1,
            "slt": 2,
            "sle": 3,
            "sgt": 4,
            "sge": 5,
            "ult": 6,
            "ule": 7,
            "ugt": 8,
            "uge": 9,
        },
        "float": {
            "false": 0,
            "oeq": 1,
            "ogt": 2,
            "oge": 3,
            "olt": 4,
            "ole": 5,
            "one": 6,
            "ord": 7,
            "ueq": 8,
            "ugt": 9,
            "uge": 10,
            "ult": 11,
            "ule": 12,
            "une": 13,
            "uno": 14,
            "true": 15,
        },
        "fixed": {
            "eq": 0,
            "ne": 1,
            "slt": 2,
            "sle": 3,
            "sgt": 4,
            "sge": 5,
            "ult": 6,
            "ule": 7,
            "ugt": 8,
            "uge": 9,
        },
    }

    def __init__(self, lhs, rhs, arg, loc=None):
        self.arg = arg
        dtype = lhs.dtype
        if is_integer_type(dtype) or is_index_type(dtype):
            self.op = arith.CmpIOp
            if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
                self.arg = CmpOp.ATTR_MAP["int"][
                    "s" + arg if arg not in ["eq", "ne"] else arg
                ]
            else:
                self.arg = CmpOp.ATTR_MAP["int"][
                    "u" + arg if arg not in ["eq", "ne"] else arg
                ]
        elif is_floating_point_type(dtype):
            self.op = arith.CmpFOp
            self.arg = CmpOp.ATTR_MAP["float"]["o" + arg]
        elif is_fixed_type(dtype):
            self.op = hcl_d.CmpFixedOp
            if isinstance(dtype, hcl_d.FixedType):
                self.arg = CmpOp.ATTR_MAP["fixed"][
                    "s" + arg if arg not in ["eq", "ne"] else arg
                ]
            else:
                self.arg = CmpOp.ATTR_MAP["fixed"][
                    "u" + arg if arg not in ["eq", "ne"] else arg
                ]
        else:
            raise DTypeError("Unsupported types for CmpOp: {}".format(dtype))
        super().__init__(self.op, IntegerType.get_signless(1), lhs, rhs)

    def build(self):
        self.built_op = self.op(
            IntegerAttr.get(IntegerType.get_signless(64), self.arg),
            self.lhs.result,
            self.rhs.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class AddOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {"float": arith.AddFOp, "int": arith.AddIOp, "fixed": hcl_d.AddFixedOp},
            dtype,
            lhs,
            rhs,
            loc
        )


class SubOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {"float": arith.SubFOp, "int": arith.SubIOp, "fixed": hcl_d.SubFixedOp},
            dtype,
            lhs,
            rhs,
        )


class MulOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {"float": arith.MulFOp, "int": arith.MulIOp, "fixed": hcl_d.MulFixedOp},
            dtype,
            lhs,
            rhs,
        )


class DivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {
                "float": arith.DivFOp,
                "int": arith.DivSIOp,
                "uint": arith.DivUIOp,
                "fixed": hcl_d.DivFixedOp,
            },
            dtype,
            lhs,
            rhs,
        )


class FloorDivOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {
                "float": arith.DivFOp,
                "int": arith.DivSIOp,
                "uint": arith.DivUIOp,
            },  # not supported!
            dtype,
            lhs,
            rhs,
        )


class RemOp(BinaryOp):
    def __init__(self, dtype, lhs, rhs, loc=None):
        super().__init__(
            {"float": arith.RemFOp, "int": arith.RemSIOp, "uint": arith.RemUIOp},
            dtype,
            lhs,
            rhs,
        )


class LeftShiftOp(BinaryOp):
    def __init__(self, lhs, rhs, loc=None):
        if isinstance(rhs, int):
            new_type = IntegerType.get_signless(lhs.dtype.width + rhs)
            lhs = CastOp(lhs, new_type)
            rhs = CastOp(rhs, new_type)
        elif isinstance(rhs, CastOp):
            new_type = IntegerType.get_signless(
                lhs.dtype.width + rhs.dtype.width)
            lhs = CastOp(lhs, new_type)
            rhs = CastOp(rhs, new_type)
        else:
            new_type = lhs.dtype
        super().__init__(arith.ShLIOp, lhs.dtype, lhs, rhs)


class RightShiftOp(BinaryOp):
    def __init__(self, lhs, rhs, loc=None):
        super().__init__(arith.ShRUIOp, lhs.dtype, lhs, rhs)


class AndOp(BinaryOp):
    def __init__(self, lhs, rhs, loc=None):
        super().__init__(arith.AndIOp, lhs.dtype, lhs, rhs)


class OrOp(BinaryOp):
    def __init__(self, lhs, rhs, loc=None):
        super().__init__(arith.OrIOp, lhs.dtype, lhs, rhs)


class XOrOp(BinaryOp):
    def __init__(self, lhs, rhs, loc=None):
        super().__init__(arith.XOrIOp, lhs.dtype, lhs, rhs)


class NegOp(UnaryOp):
    def __init__(self, val, loc=None):
        if is_floating_point_type(val.dtype):
            super().__init__(arith.NegFOp, val.dtype, val)
        else:
            raise DTypeError("Unsupported types for NegOp: {}".format(val.dtype))


class BitReverseOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(hcl_d.BitReverseOp, val.dtype, val)


class BitCastOp(UnaryOp):
    def __init__(self, dtype, val, loc=None):
        super().__init__(arith.BitcastOp, dtype, val)

    def build(self):
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
            self.built_op = self.op(
                dtype, self.val.result, ip=GlobalInsertionPoint.get()
            )
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        else:
            dtype = self.dtype
            self.built_op = self.op(
                self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class MathExpOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.ExpOp, F32Type.get(), val)


class PrintOp(ExprOp):
    def __init__(self, val, format_str="", loc=None):
        self.val = val
        self.format_str = format_str
        self.operands = [v.result for v in val]
        super().__init__(hcl_d.PrintOp)
        if flags.BUILD_INPLACE:
            self.build()
    
    def build(self):
        self.built_op = self.op(
            self.operands, ip=GlobalInsertionPoint.get()
        )
        sign_str = ""
        for v in self.val:
            if is_unsigned_type(v.dtype):
                sign_str += "u"
            else:
                sign_str += "_"
        self.built_op.attributes["signedness"] = StringAttr.get(sign_str)
        # Attach format string as an attribute
        if self.format_str != "":
            self.built_op.attributes["format"] = StringAttr.get(self.format_str)
        return self.built_op


class PrintMemRefOp(UnaryOp):
    def __init__(self, val, dtype, loc=None):
        super().__init__(hcl_d.PrintMemRefOp, get_mlir_type(dtype), val)


class MathPowOp(BinaryOp):
    def __init__(self, x, y, loc=None):
        if not isinstance(x, (int, float)):
            dtype = x.dtype
        if not isinstance(y, (int, float)):
            dtype = y.dtype
        x = get_hcl_op(x, F32Type.get())
        y = get_hcl_op(y, F32Type.get())
        super().__init__(math.PowFOp, dtype, x, y)

    def build(self):
        self.built_op = self.op(
            self.lhs.result, self.rhs.result, ip=GlobalInsertionPoint.get()
        )
        if not is_floating_point_type(self.dtype):
            self.built_op = arith.FPToSIOp(
                self.dtype, self.built_op.result, ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class MathLogOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.LogOp, F32Type.get(), val)


class MathLog2Op(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.Log2Op, F32Type.get(), val)


class MathLog10Op(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.Log10Op, F32Type.get(), val)


class MathSqrtOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.SqrtOp, F32Type.get(), val)


class MathSinOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.SinOp, F32Type.get(), val)


class MathCosOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.CosOp, F32Type.get(), val)


class MathTanhOp(UnaryOp):
    def __init__(self, val, loc=None):
        super().__init__(math.TanhOp, F32Type.get(), val)


class CastOp(ExprOp):
    def __init__(self, val, res_type=None, loc=None):
        # dtype is the result type
        res_type = get_mlir_type(res_type)
        self.val = get_hcl_op(val)
        self.val.dtype = get_mlir_type(self.val.dtype)
        if res_type == self.val.dtype:
            op = None
        elif (is_index_type(res_type) and is_integer_type(self.val.dtype)) or (
            is_index_type(self.val.dtype) and is_integer_type(res_type)
        ):
            op = arith.IndexCastOp
        elif is_signed_type(self.val.dtype) and is_floating_point_type(res_type):
            op = arith.SIToFPOp
        elif is_unsigned_type(self.val.dtype) and is_floating_point_type(res_type):
            op = arith.UIToFPOp
        elif is_signed_type(res_type) and is_floating_point_type(self.val.dtype):
            op = arith.FPToSIOp
        elif is_unsigned_type(res_type) and is_floating_point_type(self.val.dtype):
            op = arith.FPToUIOp
        elif is_integer_type(res_type) and is_integer_type(self.val.dtype):
            if res_type.width < self.val.dtype.width:
                op = arith.TruncIOp
            elif res_type.width == self.val.dtype.width:
                op = None
            else:
                if (
                    isinstance(self.val, (GetBitOp, GetSliceOp, LeftShiftOp))
                    or self.val.dtype.width == 1
                ):
                    op = arith.ExtUIOp
                elif is_unsigned_type(self.val.dtype):
                    op = arith.ExtUIOp
                else:
                    op = arith.ExtSIOp
        elif is_floating_point_type(res_type) and is_floating_point_type(
            self.val.dtype
        ):
            res_width = get_floating_point_width(res_type)
            val_width = get_floating_point_width(self.val.dtype)
            if res_width < val_width:
                op = arith.TruncFOp
            elif res_width == val_width:
                op = None
            else:
                op = arith.ExtFOp
        elif is_fixed_type(res_type) and is_floating_point_type(self.val.dtype):
            op = hcl_d.FloatToFixedOp
        elif is_floating_point_type(res_type) and is_fixed_type(self.val.dtype):
            op = hcl_d.FixedToFloatOp
        elif is_fixed_type(res_type) and is_integer_type(self.val.dtype):
            op = hcl_d.IntToFixedOp
        elif is_integer_type(res_type) and is_fixed_type(self.val.dtype):
            op = hcl_d.FixedToIntOp
        elif is_fixed_type(res_type) and is_fixed_type(self.val.dtype):
            if (
                res_type.width == self.val.dtype.width
                and res_type.frac == self.val.dtype.frac
                and is_signed_fixed_type(res_type)
                == is_signed_fixed_type(self.val.dtype)
            ):
                op = None
            else:
                op = hcl_d.FixedToFixedOp
        elif is_struct_type(res_type) and is_struct_type(self.val.dtype):
            # We don't actually cast between struct types,
            # here we check if two structs are identical when all
            # integer fields are signless.
            res_field_types = res_type.field_types
            val_field_types = self.val.dtype.field_types
            if len(res_field_types) != len(val_field_types):
                raise HCLValueError(
                    "Casting between structs with different number of fields. " +
                    f"src type: {self.val.dtype}, dst type: {res_type}"
                )
            for res_ftype, val_ftype in zip(res_field_types, val_field_types):
                res_ftype = get_concrete_type(res_ftype)
                val_ftype = get_concrete_type(val_ftype)
                if is_integer_type(res_ftype) and is_integer_type(val_ftype):
                    # check bitwidth
                    if get_bitwidth(res_ftype) != get_bitwidth(val_ftype):
                        raise HCLValueError(
                            "Casting between structs with different field bitwidth. " +
                            f"src type: {self.val.dtype}, dst type: {res_type}"
                        )
                else:
                    # check if the field types are identical
                    if res_ftype != val_ftype:
                        raise HCLValueError(
                            "Casting between structs with different field types. " +
                            f"src type: {self.val.dtype}, dst type: {res_type}"
                        )
            op = None
        elif is_struct_type(res_type) and is_integer_type(self.val.dtype):
            if not is_all_field_int(res_type):
                raise HCLValueError(
                        "Casting from integer to struct with non-integer fields. " +
                        f"src type: {self.val.dtype}, dst type: {res_type}"
                    )
            total_width = get_bitwidth(res_type)
            cvtype = get_concrete_type(self.val.dtype)
            if total_width != get_bitwidth(cvtype):
                raise HCLValueError(
                    "Casting between integer and struct with different bitwidth. " +
                    f"src type: {self.val.dtype}, dst type: {res_type}"
                )
            op = hcl_d.IntToStructOp
        else:
            op = builtin.UnrealizedConversionCastOp
            raise DTypeError(
                "Unrealized conversion cast: {} -> {}".format(
                    self.val.dtype, res_type)
            )

        super().__init__(op, res_type)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if self.op in [
            arith.IndexCastOp,
            arith.SIToFPOp,
            arith.UIToFPOp,
            arith.FPToSIOp,
            arith.FPToUIOp,
            arith.TruncIOp,
            arith.TruncFOp,
            arith.ExtUIOp,
            arith.ExtSIOp,
            arith.ExtFOp,
            hcl_d.FixedToIntOp,
            hcl_d.IntToFixedOp,
            hcl_d.FixedToFloatOp,
            hcl_d.FloatToFixedOp,
            hcl_d.FixedToFixedOp,
            hcl_d.IntToStructOp
        ]:
            if is_unsigned_type(self.dtype) or is_struct_type(self.dtype):
                dtype = get_signless_type(self.dtype)
                self.built_op = self.op(
                    dtype, self.val.result, ip=GlobalInsertionPoint.get()
                )
                self.built_op.attributes["unsigned"] = UnitAttr.get()
            else:
                self.built_op = self.op(
                    self.dtype, self.val.result, ip=GlobalInsertionPoint.get()
                )
        elif self.op == None:
            if self.val.built_op is None:
                self.val.build()
            self.built_op = self.val.built_op
        else:  # builtin.UnrealizedConversionCastOp
            self.built_op = self.op(
                [self.dtype], [self.val.result], ip=GlobalInsertionPoint.get()
            )
        return self.built_op


class GetBitOp(ExprOp):
    def __init__(self, num, index, loc=None):
        super().__init__(hcl_d.GetIntBitOp, IntegerType.get_signless(1))
        self.num = num
        if isinstance(index, int):
            index = ConstantOp(IndexType.get(), index)
        self.index = index
        if not isinstance(self.index.dtype, IndexType):
            DTypeWarning(
                "GetBitOp's input is not an index. Cast from {} to {}.".format(
                    self.index.dtype, IndexType.get()
                )
            ).warn()
            self.index = CastOp(self.index, IndexType.get())
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
        else:
            dtype = self.dtype
        self.built_op = self.op(
            dtype,
            self.num.result,
            self.index.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        flags.BIT_OP = True
        return self.built_op


class LogicalAndOp(ExprOp):
    def __init__(self, *cond, loc=None):
        super().__init__(hcl_d.LogicalAndOp, IntegerType.get_signless(1))
        self.cond_lst = cond

    def build(self):
        raise APIError("Do not build logical_and op")


class LogicalOrOp(ExprOp):
    def __init__(self, *cond, loc=None):
        super().__init__(hcl_d.LogicalOrOp, IntegerType.get_signless(1))
        self.cond_lst = cond
        raise MLIRLimitationError("LogicalOrOp not implemented")

    def build(self):
        raise APIError("Do not build logical_or op")


class SetBitOp(ExprOp):
    def __init__(self, num, index, val, loc=None):
        super().__init__(hcl_d.SetIntBitOp, None)  # No return value!
        self.num = num  # actually a LoadOp
        if isinstance(index, int):
            index = ConstantOp(IndexType.get(), index)
        self.index = index
        if isinstance(val, int):
            val = ConstantOp(IntegerType.get_signless(1), val)
        if not (is_integer_type(val.dtype) and val.dtype.width == 1):
            raise HCLValueError(
                "Can only set a bit of 0/1. Got {} with dtype {}.".format(
                    val, val.dtype
                )
            )
        self.val = val
        if not isinstance(self.index.dtype, IndexType):
            DTypeWarning(
                "SetBitOp's input is not an index. Cast from {} to {}.".format(
                    self.index.dtype, IndexType.get()
                )
            ).warn()
            self.index = CastOp(self.index, IndexType.get())
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.num.result,
            self.index.result,
            self.val.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        if isinstance(self.num, LoadOp):
            self.built_op = StoreOp(
                self.num, self.num.tensor, self.num.indices)
        flags.BIT_OP = True
        return self.built_op


class GetSliceOp(ExprOp):
    def __init__(self, num, hi, lo, loc=None):
        super().__init__(hcl_d.GetIntSliceOp, num.dtype)
        self.num = num

        def normalize(index):
            if isinstance(index, int):
                index = ConstantOp(IndexType.get(), index)
            if not isinstance(index.dtype, IndexType):
                DTypeWarning(
                    "GetSliceOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                ).warn()
                index = CastOp(index, IndexType.get())
            return index

        self.hi = normalize(hi)
        self.lo = normalize(lo)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(self.dtype.width)
        else:
            dtype = self.dtype
        self.built_op = self.op(
            dtype,
            self.num.result,
            self.hi.result,
            self.lo.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        flags.BIT_OP = True
        return self.built_op


class SetSliceOp(ExprOp):
    def __init__(self, num, hi, lo, val, loc=None):
        super().__init__(hcl_d.SetIntSliceOp, None)  # No return value!
        self.num = num  # actually a LoadOp

        def normalize(index):
            if isinstance(index, int):
                index = ConstantOp(IndexType.get(), index)
            if not isinstance(index.dtype, IndexType):
                DTypeWarning(
                    "SetSliceOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                ).warn()
                index = CastOp(index, IndexType.get())
            return index

        self.hi = normalize(hi)
        self.lo = normalize(lo)
        if isinstance(val, int):
            val = ConstantOp(num.dtype, val)
        self.val = val
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.num.result,
            self.hi.result,
            self.lo.result,
            self.val.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        if isinstance(self.num, LoadOp):
            self.built_op = StoreOp(
                self.num, self.num.tensor, self.num.indices)
        flags.BIT_OP = True
        return self.built_op


class LoadOp(ExprOp):
    def __init__(self, tensor, indices, loc=None):
        super().__init__(affine.AffineLoadOp, tensor.dtype)
        self.tensor = tensor
        self.indices = []
        for index in indices:
            if not isinstance(get_mlir_type(index.dtype), IndexType):
                DTypeWarning(
                    "LoadOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                ).warn()
                index = CastOp(index, IndexType.get())
            self.indices.append(index)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        # test if affine expressions
        visitor = ASTVisitor(mode="profile")
        exprs = []
        flag = True
        for index in self.indices:
            try:
                affine_expr = visitor.visit_affine_expr(index)
                exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            remover = ASTVisitor(mode="remove")
            for index in self.indices:
                if index.built_op is not None:
                    remover.visit(index)
            affine_map = AffineMap.get(
                dim_count=len(visitor.iv), symbol_count=0, exprs=exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            self.built_op = self.op(
                self.tensor.result,
                visitor.iv,
                affine_attr,
                ip=GlobalInsertionPoint.get(),
            )
        else:
            new_indices = []
            for index in self.indices:
                new_indices.append(index.result)
            self.built_op = memref.LoadOp(
                self.tensor.result, new_indices, ip=GlobalInsertionPoint.get()
            )
        self.built_op.attributes["from"] = StringAttr.get(self.tensor.name)
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class StoreOp(ExprOp):
    def __init__(self, val, to_tensor, indices, loc=None):
        super().__init__(affine.AffineStoreOp)
        val = get_hcl_op(val)
        if val.dtype != to_tensor.dtype:
            DTypeWarning(
                "StoreOp has different input types. Cast from {} to {}.".format(
                    val.dtype, to_tensor.dtype)
            ).warn()
            val = CastOp(val, to_tensor.dtype)
        self.val = val
        self.to_tensor = to_tensor
        self.indices = []
        for index in indices:
            if isinstance(index, int):
                index = ConstantOp(IndexType.get(), index)
            elif not isinstance(get_mlir_type(index.dtype), IndexType):
                DTypeWarning(
                    "StoreOp's input is not an index. Cast from {} to {}.".format(
                        index.dtype, IndexType.get()
                    )
                ).warn()
                index = CastOp(index, IndexType.get())
            self.indices.append(index)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        # test if affine expressions
        visitor = ASTVisitor(mode="profile")
        exprs = []
        flag = True
        for index in self.indices:
            try:
                affine_expr = visitor.visit_affine_expr(index)
                exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            affine_map = AffineMap.get(
                dim_count=len(visitor.iv), symbol_count=0, exprs=exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            self.built_op = self.op(
                self.val.result,
                self.to_tensor.result,
                visitor.iv,
                affine_attr,
                ip=GlobalInsertionPoint.get(),
            )
        else:
            new_indices = []
            for index in self.indices:
                new_indices.append(index.result)
            self.built_op = memref.StoreOp(
                self.val.result,
                self.to_tensor.result,
                new_indices,
                ip=GlobalInsertionPoint.get(),
            )
        self.built_op.attributes["to"] = StringAttr.get(self.to_tensor.name)
        if is_unsigned_type(self.to_tensor.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class CallOp(ExprOp):
    def __init__(self, dtype, func_name, inputs, loc=None):
        # here we only accept one result
        super().__init__(func.CallOp, dtype)
        self.func_name = func_name
        self.inputs = inputs
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            [self.dtype] if self.dtype != None else [],
            FlatSymbolRefAttr.get(self.func_name),
            self.inputs,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class SelectOp(ExprOp):
    """Ternary operation"""

    def __init__(self, cond, true_val, false_val, loc=None):
        super().__init__(arith.SelectOp)
        # turn py builtin op to hcl op
        true_val = get_hcl_op(true_val)
        false_val = get_hcl_op(false_val)
        # do the testing
        if true_val.dtype != false_val.dtype:
            raise DTypeError(
                "SelectOp should have two same type of inputs. Got {} and {}".format(
                    true_val.dtype, false_val.dtype
                )
            )
        self.dtype = true_val.dtype
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.cond.result,
            self.true_val.result,
            self.false_val.result,
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class StructConstructOp(ExprOp):
    def __init__(self, fields, loc=None):
        super().__init__(hcl_d.StructConstructOp)
        self.fields = fields
        self.field_results = [f.result for f in fields]
        self.field_types = [f.type for f in self.field_results]
        self.dtype = hcl_d.StructType.get(self.field_types)
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        self.built_op = self.op(
            self.dtype,
            self.field_results,
            ip=GlobalInsertionPoint.get(),
        )
        return self.built_op


class StructGetOp(ExprOp):
    def __init__(self, struct, index, loc=None):
        super().__init__(hcl_d.StructGetOp)
        self.struct = struct
        self.index = index
        field_types = self.struct.dtype.field_types
        self.dtype = get_concrete_type(field_types[self.index])  # mlir type
        if flags.BUILD_INPLACE:
            self.build()

    def build(self):
        # Note(Niansong):
        # this was used to test dtype from HalideIR,
        # e.g. assert struct_value.field.dtype = "int8"
        # but this is no longer compatible with mlir.
        # self.dtype has to be a concrete MLIR type, for
        # the field value to be consumed by another operation.
        if is_unsigned_type(self.dtype):
            dtype = IntegerType.get_signless(get_bitwidth(self.dtype))
        else:
            dtype = self.dtype
        self.built_op = self.op(
            dtype,
            self.struct.result,
            IntegerAttr.get(IntegerType.get_signless(64), self.index),
            ip=GlobalInsertionPoint.get(),
        )
        if is_unsigned_type(self.dtype):
            self.built_op.attributes["unsigned"] = UnitAttr.get()
        return self.built_op


class ReduceOp(ExprOp):
    # cannot build inplace!!!
    def __init__(self, op, axis, dtype, prefix, init_val, reduce_op, loc=None):
        super().__init__(op, dtype=get_mlir_type(dtype))
        self.axis = axis
        self.prefix = prefix
        self.init_val = init_val
        self.reduce_op = reduce_op


class SumOp(ReduceOp):
    def __init__(self, op, axis, dtype, loc=None):
        super().__init__(
            op,
            axis,
            dtype,
            prefix="sum",
            init_val=0,
            reduce_op={
                "float": arith.AddFOp,
                "si": arith.AddIOp,
                "ui": arith.AddIOp,
                "fixed": hcl_d.AddFixedOp,
            },
        )


class MinOp(ReduceOp):
    def __init__(self, op, axis, dtype, loc=None):
        super().__init__(
            op,
            axis,
            dtype,
            prefix="min",
            init_val=0x3F3F3F3F,
            reduce_op={
                "float": arith.MinFOp,
                "si": arith.MinSIOp,
                "ui": arith.MinUIOp,
                "fixed": hcl_d.MinFixedOp,
            },
        )


class MaxOp(ReduceOp):
    def __init__(self, op, axis, dtype, loc=None):
        super().__init__(
            op,
            axis,
            dtype,
            prefix="max",
            init_val=-0x3F3F3F3F,
            reduce_op={
                "float": arith.MaxFOp,
                "si": arith.MaxSIOp,
                "ui": arith.MaxUIOp,
                "fixed": hcl_d.MaxFixedOp,
            },
        )


class ASTVisitor:
    def __init__(self, mode="build", op=None, tag_only=False):
        self.iv = []
        if mode not in ["build", "remove", "profile", "move_before"]:
            raise APIError(
                "ASTVisitor only supports build, remove, profile, or move_before mode"
            )
        self.mode = mode
        self.load = []
        self.store = []
        self.scf_cnt = 0
        # Move ops in the AST before the given op
        self.target_op = op
        self.tag_only = tag_only

    def visit(self, expr):
        """Apply the visitor to an expression."""
        if (
            self.mode == "build"
            and not isinstance(expr, tuple)
            and expr.built_op is not None
        ):
            return expr.built_op

        if isinstance(expr, UnaryOp):
            return self.visit_unary_op(expr)
        elif isinstance(expr, BinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, SelectOp):
            return self.visit_ternary_op(expr)
        elif isinstance(expr, LoadOp):
            return self.visit_load_op(expr)
        elif isinstance(expr, StoreOp):
            return self.visit_store_op(expr)
        elif isinstance(expr, GetBitOp):
            return self.visit_getbit_op(expr)
        elif isinstance(expr, SetBitOp):
            return self.visit_setbit_op(expr)
        elif isinstance(expr, GetSliceOp):
            return self.visit_getslice_op(expr)
        elif isinstance(expr, SetSliceOp):
            return self.visit_setslice_op(expr)
        elif isinstance(expr, CastOp):
            return self.visit_cast_op(expr)
        elif isinstance(expr, ReduceOp):
            return self.visit_reduce_op(expr)
        elif isinstance(expr, ConstantOp):
            return self.visit_constant_op(expr)
        elif isinstance(expr, tuple):
            # tuple expr corresponds to a struct construction
            return self.visit_struct_op(expr)
        elif isinstance(expr, StructGetOp):
            return self.visit_struct_get_op(expr)
        else:  # IterVar
            return self.visit_block_arg(expr)

    def visit_block_arg(self, expr):
        if self.mode == "profile":
            if isinstance(expr.op.owner.owner, scf.ForOp):
                self.scf_cnt += 1
        else:
            return expr.built_op

    def visit_affine_expr(self, expr):
        """Build affine expression.
        * Should all be binary op
        * AffineExpr can be automatically simplied
        """
        if not isinstance(expr, (IterVar, ConstantOp, CastOp, BinaryOp)):
            raise HCLValueError("Not an affine index!")
        if isinstance(expr, IterVar):
            if isinstance(expr.op.owner.owner, scf.ForOp):
                raise HCLValueError("Outer loop is not affine!")
            if expr.op not in self.iv:
                self.iv.append(expr.op)  # BlockArgument
                return AffineExpr.get_dim(len(self.iv) - 1)
            else:
                return AffineExpr.get_dim(self.iv.index(expr.op))
        elif isinstance(expr, ConstantOp):
            return AffineExpr.get_constant(expr.val)
        elif isinstance(expr, CastOp):
            return self.visit_affine_expr(expr.val)
        lhs = self.visit_affine_expr(expr.lhs)
        rhs = self.visit_affine_expr(expr.rhs)
        if isinstance(expr, AddOp):
            return lhs + rhs
        elif isinstance(expr, SubOp):
            return lhs - rhs
        elif isinstance(expr, MulOp):
            return lhs * rhs
        elif isinstance(expr, DivOp):
            return AffineExpr.get_floor_div(lhs, rhs)  # or get_ceil_div
        elif isinstance(expr, RemOp):
            return lhs % rhs
        else:
            raise HCLValueError("Not an affine index!")

    def erase_op(self, expr):
        # If expr is a "pass-through" op,
        # i.e. the op is not a real op, its `op` is None,
        # remove its built_op without erasing
        # its built_op. An example is the ConstantOp,
        # whose op can be set to None representing
        # a pass-through op.
        if expr.op is None:
            expr.built_op = None
            return
        expr.built_op.operation.erase()
        expr.built_op = None

    def move_before(self, expr, target_op):
        if expr.built_op is None:
            return
        if "moved" in expr.built_op.attributes:
            return
        if not self.tag_only:
            expr.built_op.move_before(target_op)
        expr.built_op.attributes["moved"] = UnitAttr.get()

    def visit_unary_op(self, expr):
        if self.mode == "build":
            self.visit(expr.val)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.val)
        elif self.mode == "move_before":
            self.visit(expr.val)
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)
            self.visit(expr.val)

    def visit_binary_op(self, expr):
        if self.mode == "build":
            self.visit(expr.lhs)
            self.visit(expr.rhs)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.lhs)
            self.visit(expr.rhs)
        elif self.mode == "move_before":
            self.visit(expr.lhs)
            self.visit(expr.rhs)
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)
            self.visit(expr.rhs)
            self.visit(expr.lhs)

    def visit_ternary_op(self, expr):
        if self.mode == "build":
            # condition
            if is_unsigned_type(expr.dtype):
                dtype = IntegerType.get_signless(expr.dtype.width)
            else:
                dtype = expr.dtype
            if_op = make_if(
                expr.cond,
                ip=GlobalInsertionPoint.get(),
                hasElse=True,
                resultType=[dtype],
                yieldOp=False,
            )
            if is_unsigned_type(expr.dtype):
                if_op.attributes["unsigned"] = UnitAttr.get()
            # true branch
            GlobalInsertionPoint.save(if_op.then_block)
            true_val = self.visit(expr.true_val).result
            if isinstance(if_op, affine.AffineIfOp):
                affine.AffineYieldOp([true_val], ip=GlobalInsertionPoint.get())
            else:  # scf.IfOp
                scf.YieldOp([true_val], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            # false branch
            GlobalInsertionPoint.save(if_op.else_block)
            false_val = self.visit(expr.false_val).result
            if isinstance(if_op, affine.AffineIfOp):
                affine.AffineYieldOp(
                    [false_val], ip=GlobalInsertionPoint.get())
            else:  # scf.IfOp
                scf.YieldOp([false_val], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            expr.built_op = if_op
            return if_op
        elif self.mode == "profile":
            self.visit(expr.cond)
            self.visit(expr.true_val)
            self.visit(expr.false_val)
        elif self.mode == "move_before":
            self.visit(expr.cond)
            self.visit(expr.true_val)
            self.visit(expr.false_val)
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)
            self.visit(expr.false_val)
            self.visit(expr.true_val)
            self.visit(expr.cond)

    def visit_struct_op(self, expr):
        if self.mode == "build":
            fields = [self.visit(e) for e in expr]
            op = StructConstructOp(fields)
            return op.build()
        elif self.mode == "profile":
            fields = [self.visit(e) for e in expr]
        elif self.mode == "move_before":
            fields = [self.visit(e) for e in expr]
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)
        return

    def visit_struct_get_op(self, expr):
        if self.mode == "build":
            self.visit(expr.struct)
            return expr.build()
        elif self.mode == "profile":
            self.visit(expr.struct)
        elif self.mode == "move_before":
            self.visit(expr.struct)
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)
        return

    def visit_load_op(self, expr):
        if self.mode == "remove":
            self.erase_op(expr)
            return
        elif self.mode == "profile":
            self.load.append(expr)
            return
        elif self.mode == "move_before":
            self.move_before(expr, self.target_op)
            return
        else:
            return expr.build()

    def visit_store_op(self, expr):
        if self.mode == "remove":
            self.erase_op(expr)
            return
        elif self.mode == "profile":
            self.store.append(expr)
            return
        elif self.mode == "move_before":
            self.move_before(expr, self.target_op)
            return
        else:
            return expr.build()

    def visit_cast_op(self, expr):
        if self.mode == "move_before":
            self.visit(expr.val)
            self.move_before(expr, self.target_op)
            return
        if self.mode == "remove":
            self.erase_op(expr)
        self.visit(expr.val)
        if self.mode == "build":
            return expr.build()

    def visit_getbit_op(self, expr):
        if self.mode == "build":
            self.visit(expr.num)
            self.visit(expr.index)
            return expr.build()
        elif self.mode == "move_before":
            self.visit(expr.num)
            self.visit(expr.index)
            self.move_before(expr, self.target_op)
        elif self.mode == "remove":
            self.erase_op(expr)
            self.visit(expr.index)
            self.visit(expr.num)
        else:
            self.visit(expr.num)
            self.visit(expr.index)

    def visit_getslice_op(self, expr):
        if self.mode == "build":
            self.visit(expr.num)
            self.visit(expr.hi)
            self.visit(expr.lo)
            return expr.build()
        elif self.mode == "move_before":
            self.visit(expr.num)
            self.visit(expr.hi)
            self.visit(expr.lo)
            self.move_before(expr, self.target_op)
        elif self.mode == "remove":
            self.erase_op(expr)
            self.visit(expr.lo)
            self.visit(expr.hi)
            self.visit(expr.num)
        else:
            self.visit(expr.num)
            self.visit(expr.hi)
            self.visit(expr.lo)

    def visit_setbit_op(self, expr):
        if self.mode == "move_before":
            self.visit(expr.num)
            self.visit(expr.index)
            self.visit(expr.val)
            self.move_before(expr, self.target_op)
            return
        if self.mode == "remove":
            self.erase_op(expr)
        self.visit(expr.num)
        self.visit(expr.index)
        self.visit(expr.val)
        if self.mode == "build":
            return expr.build()

    def visit_setslice_op(self, expr):
        if self.mode == "move_before":
            self.visit(expr.num)
            self.visit(expr.hi)
            self.visit(expr.lo)
            self.visit(expr.val)
            self.move_before(expr, self.target_op)
            return
        if self.mode == "remove":
            self.erase_op(expr)
        self.visit(expr.num)
        self.visit(expr.hi)
        self.visit(expr.lo)
        self.visit(expr.val)
        if self.mode == "build":
            return expr.build()

    def visit_constant_op(self, expr):
        if self.mode == "build":
            return expr.build()
        elif self.mode == "profile":
            pass
        elif self.mode == "move_before":
            self.move_before(expr, self.target_op)
        else:
            self.erase_op(expr)

    def visit_reduce_op(self, expr):
        if self.mode == "remove":
            raise APIError("Cannot remove ReduceOp")
        elif self.mode == "move_before":
            raise APIError("Moving reduce op is not supported")
        elif self.mode == "profile":
            return
        # save insetion point
        save_ip = GlobalInsertionPoint.get()

        # create a single-element register for reduction
        dtype = expr.dtype
        if is_unsigned_type(dtype):
            dtype = IntegerType.get_signless(dtype.width)
        memref_type = MemRefType.get((1,), dtype)
        rv = memref.AllocOp(memref_type, [], [],
                            ip=GlobalInsertionPoint.get())
        prefix = expr.prefix
        init_val = expr.init_val
        reduce_op = expr.reduce_op
        rv.attributes["name"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            rv.attributes["unsigned"] = UnitAttr.get()
        # initialize the single-element register
        zero_idx = arith.ConstantOp(
            IndexType.get(),
            IntegerAttr.get(IndexType.get(), 0),
            ip=GlobalInsertionPoint.get(),
        )
        # initialize the original value of the reducer
        if is_floating_point_type(dtype):
            zero_value = arith.ConstantOp(
                dtype, FloatAttr.get(dtype, init_val), ip=GlobalInsertionPoint.get()
            )
        elif is_integer_type(dtype):
            zero_value = arith.ConstantOp(
                dtype, IntegerAttr.get(dtype, init_val), ip=GlobalInsertionPoint.get()
            )
        elif is_fixed_type(dtype):
            value_attr = IntegerAttr.get(
                IntegerType.get_signless(32), init_val)
            zero_value = arith.ConstantOp(
                IntegerType.get_signless(32), value_attr, ip=GlobalInsertionPoint.get()
            )
            zero_value = hcl_d.IntToFixedOp(
                dtype, zero_value.result, ip=GlobalInsertionPoint.get()
            )
        else:
            raise DTypeError(
                "Unrecognized data type in reduction op: {}".format(dtype))
        if is_unsigned_type(expr.dtype):
            zero_value.attributes["unsigned"] = UnitAttr.get()

        store = affine.AffineStoreOp(
            zero_value.result,
            rv.result,
            [zero_idx.result],
            ip=GlobalInsertionPoint.get(),
        )
        store.attributes["to"] = StringAttr.get("{}_rv".format(prefix))

        # create reduction loop
        if not isinstance(expr.axis, list):
            new_axes = [expr.axis]
        else:
            new_axes = expr.axis
        body_ip = GlobalInsertionPoint.get()
        for axis in new_axes:
            reduction_loop = make_for(
                axis.lower_bound,
                axis.upper_bound,
                step=1,
                reduction=True,
                name=axis.name,
                ip=body_ip,
            )
            body_ip = InsertionPoint(reduction_loop.body.operations[0])

            # update reduction variable
            axis.update_op(reduction_loop.induction_variable)

            # update insertion point
            GlobalInsertionPoint.save(body_ip)

        # visit subexpressions
        data = self.visit(expr.op)

        # load register value and reduce
        load = affine.AffineLoadOp(
            rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        load.attributes["from"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            load.attributes["unsigned"] = UnitAttr.get()
        if is_floating_point_type(dtype):
            reduce_op = reduce_op["float"]
        elif is_integer_type(dtype):
            if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
                reduce_op = reduce_op["si"]
            else:  # unsigned
                reduce_op = reduce_op["ui"]
        elif is_fixed_type(dtype):
            reduce_op = reduce_op["fixed"]
        else:
            raise DTypeError("Unsupported type: {}".format(dtype))
        data_type = get_concrete_type(data.result.type)
        if "unsigned" in data.attributes:
            data_type = IntegerType.get_unsigned(data_type.width)
        if dtype != data_type:
            DTypeWarning(
                "Reduction variable should have the same type with the data. Got {0} and {1}. Do type casting from {1} to {0}".format(
                    dtype, data_type
                )
            ).warn()
            placeholder = ExprOp(None, dtype=data_type)
            placeholder.built_op = data
            data = CastOp(placeholder, dtype)
            data.build()
        iter_reduction = reduce_op(
            data.result, load.result, ip=GlobalInsertionPoint.get()
        )
        if is_unsigned_type(expr.dtype):
            iter_reduction.attributes["unsigned"] = UnitAttr.get()

        # store the result back to register
        store_reg = affine.AffineStoreOp(
            iter_reduction.result,
            rv.result,
            [zero_idx.result],
            ip=GlobalInsertionPoint.get(),
        )
        store_reg.attributes["to"] = StringAttr.get("{}_rv".format(prefix))

        # set terminator
        for axis in new_axes:
            # restore insertion point
            GlobalInsertionPoint.restore()

        zero_idx = arith.ConstantOp(
            IndexType.get(),
            IntegerAttr.get(IndexType.get(), 0),
            ip=GlobalInsertionPoint.get(),
        )
        value = affine.AffineLoadOp(
            rv.result, [zero_idx.result], ip=GlobalInsertionPoint.get()
        )
        value.attributes["from"] = StringAttr.get("{}_rv".format(prefix))
        if is_unsigned_type(expr.dtype):
            value.attributes["unsigned"] = UnitAttr.get()
        expr.built_op = value
        return value


def make_for(lb, ub, step=1, name="", stage="", reduction=False, ip=None, loc=None):
    # TODO: need to test if lb, ub, step are all affine
    # Construct step
    if not isinstance(step, int):
        raise HCLNotImplementedError("int type step is not supported")
    if step < 0:  # need to also change induction variable
        lb, ub = ub + 1, lb + 1  # swap
        step = -step
    # Construct lower bound
    const_flag = True
    if isinstance(lb, int):
        lbCst = AffineConstantExpr.get(lb)
        lbMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[lbCst])
        lbMapAttr = AffineMapAttr.get(lbMap)
        lb_expr = None
    else:
        const_flag = False

    # Construct upper bound
    if isinstance(ub, int):
        ubCst = AffineConstantExpr.get(ub)
        ubMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[ubCst])
        ubMapAttr = AffineMapAttr.get(ubMap)
        ub_expr = None
    else:
        const_flag = False

    if const_flag:
        step = IntegerAttr.get(IntegerType.get_signless(32), step)
        # Create AffineForOp
        forOp = affine.AffineForOp(
            lb_expr,
            ub_expr,
            step,
            lbMapAttr,
            ubMapAttr,
            name=(StringAttr.get("") if name in [
                  "", None] else StringAttr.get(name)),
            stage=("" if stage == "" else StringAttr.get(stage)),
            reduction=(UnitAttr.get() if reduction else None),
            ip=ip,
            loc=loc,
        )
        affine.AffineYieldOp([], ip=InsertionPoint(forOp.body))
    else:
        lb_expr = CastOp(lb, IndexType.get())
        lb_expr.build()
        lb_expr = lb_expr.result
        ub_expr = CastOp(ub, IndexType.get())
        ub_expr.build()
        ub_expr = ub_expr.result
        step = CastOp(step, IndexType.get())
        step.build()
        step = step.result
        forOp = scf.ForOp(
            lb_expr,
            ub_expr,
            step,
            name=(StringAttr.get("") if name in [
                  "", None] else StringAttr.get(name)),
            stage=("" if stage == "" else StringAttr.get(stage)),
            reduction=(UnitAttr.get() if reduction else None),
            ip=ip,
            loc=loc,
        )
        scf.YieldOp([], ip=InsertionPoint(forOp.body))

    return forOp


def make_if(cond, ip=None, hasElse=False, resultType=[], yieldOp=True, cond_pos=None):
    # suppose in a imperative context (build in-place)
    if not isinstance(cond, (CmpOp, LogicalAndOp)):
        raise HCLValueError("`if` operation condition should be CmpOp")
    visitor = ASTVisitor(mode="profile")
    if isinstance(cond, LogicalAndOp):
        lst = cond.cond_lst
        for single_cond in lst:
            visitor.visit(single_cond)
    else:
        visitor.visit(cond)
    if visitor.scf_cnt > 0 or len(visitor.load) != 0 or len(visitor.store) != 0:
        cond_expr = None
        if isinstance(cond, LogicalAndOp):
            lst = cond.cond_lst
            res = lst[0]
            for i, single_cond in enumerate(lst):
                if i == 0:
                    continue
                res = AndOp(res, single_cond)
                res.build()
            cond_result = res.result
            cond_expr = res
        else:
            cond_result = cond.result
            cond_expr = cond
        if_op = scf.IfOp(cond_result, hasElse=hasElse,
                         results_=resultType, ip=ip)
        if cond_pos is None: # top-level if
            mover = ASTVisitor(mode="move_before", tag_only=True)
            mover.visit(cond_expr)
        elif cond_expr.built_op is not None: # nested if (elif branch)
            mover = ASTVisitor(mode="move_before", op=cond_pos)
            mover.visit(cond_expr)
        if yieldOp:
            scf.YieldOp([], ip=InsertionPoint(if_op.then_block))
            if hasElse:
                scf.YieldOp([], ip=InsertionPoint(if_op.else_block))
    else:  # Affine expression
        eq_flags = []
        new_conds = []
        built_flag = True

        def build_single_cond(cond, eq_flag, new_conds):
            nonlocal built_flag
            if not isinstance(
                cond.lhs.dtype, (IntegerType, IndexType)
            ) or not isinstance(cond.rhs.dtype, (IntegerType, IndexType)):
                raise HCLValueError(
                    "`affine.if` can only support integer comparison")
            # only support affine expressions now (i.e. calculations on iteration variables)
            if cond.arg == 0:  # eq
                # lhs==rhs
                eq_flags.append(True)
                new_conds.append(cond.lhs - cond.rhs)
            elif cond.arg == 1:  # ne
                # lhs>rhs and lhs<rhs
                raise MLIRLimitationError(
                    "ne is not supported for `affine.if`")
            elif cond.arg == 2:  # slt
                # lhs<rhs -> rhs-lhs>0 -> rhs-lhs>=1 -> rhs-lhs-1>=0
                eq_flags.append(False)
                new_conds.append(cond.rhs - cond.lhs -
                                 ConstantOp(cond.lhs.dtype, 1))
            elif cond.arg == 3:  # sle
                # lhs<=rhs -> rhs-lhs>=0
                eq_flags.append(False)
                new_conds.append(cond.rhs - cond.lhs)
            elif cond.arg == 4:  # sgt
                # lhs>rhs -> lhs-rhs-1>=0
                eq_flags.append(False)
                new_conds.append(cond.lhs - cond.rhs -
                                 ConstantOp(cond.lhs.dtype, 1))
            elif cond.arg == 5:  # sge
                # lhs>=rhs -> lhs-rhs>=0
                eq_flags.append(False)
                new_conds.append(cond.lhs - cond.rhs)
            else:
                raise HCLValueError(
                    "Unknown predicate of CmpOp: {}".format(cond.arg))

            if cond.built_op is not None:
                cond.built_op.operation.erase()
                cond.built_op = None
            else:
                built_flag = False

        if isinstance(cond, LogicalAndOp):
            lst = cond.cond_lst
            for single_cond in lst:
                build_single_cond(single_cond, eq_flags, new_conds)
        else:
            build_single_cond(cond, eq_flags, new_conds)

        exprs = []
        # make sure all the AffineExpr are referenced in one visitor
        builder = ASTVisitor(mode="build")
        for new_cond in new_conds:
            if built_flag:
                remover = ASTVisitor(mode="remove")
                remover.visit(new_cond)
            # rebuild condition
            exprs.append(builder.visit_affine_expr(new_cond))
        if_cond_set = IntegerSet.get(len(builder.iv), 0, exprs, eq_flags)
        attr = hcl_d.IntegerSetAttr.get(if_cond_set)

        if_op = affine.AffineIfOp(
            attr, builder.iv, ip=ip, hasElse=hasElse, results_=resultType
        )
        if yieldOp:
            affine.AffineYieldOp([], ip=InsertionPoint(if_op.then_block))
            if hasElse:
                affine.AffineYieldOp([], ip=InsertionPoint(if_op.else_block))

    return if_op


def make_while(cond, ip=None):
    # suppose in a imperative context (build in-place)
    if not isinstance(cond, (CmpOp, LogicalAndOp, LogicalOrOp)):
        raise HCLValueError("`if` operation condition should be CmpOp, LogicalAndOp, or LogicalOrOp")

    while_op = scf.WhileOp([], [], ip=ip)
    while_op.before.blocks.append(*[])
    while_op.after.blocks.append(*[])
    GlobalInsertionPoint.save(while_op.before.blocks[0])
    if not isinstance(cond, LogicalAndOp):
        builder = ASTVisitor(mode="build")
        builder.visit(cond)
        cond_result = cond.result
    else:
        # fallback: build the conditions as arith.AndOp
        lst = cond.cond_lst
        res = lst[0]
        for i, single_cond in enumerate(lst):
            if i == 0:
                continue
            res = AndOp(res, single_cond)
            res.build()
        cond_result = res.result
    cond_op = scf.ConditionOp(cond_result, [], ip=GlobalInsertionPoint.get())
    if cond.built_op is not None:
        mover = ASTVisitor(mode="move_before", op=cond_op)
        mover.visit(cond)
    GlobalInsertionPoint.restore()
    return while_op


def get_affine_loop_nests(func):
    results = []
    for op in func.entry_block.operations:
        if isinstance(op, affine.AffineForOp):  # outer-most
            band = []
            loop = op
            while True:
                band.append(
                    {"name": loop.attributes["loop_name"], "body": loop})
                for loop in loop.body.operations:
                    if isinstance(loop, affine.AffineForOp):
                        break
                else:
                    break
            results.append(band)
    return results
