from typing import overload
from . import ir

class ConstantOp(ir.OpState): ...

class AddFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class AddIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class AndIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class BitcastOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class CmpFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, pred: int, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class CmpIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, pred: int, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class ConstantFloatOp(ConstantOp):
    @overload
    def __init__(
        self, builder: ir.AlloOpBuilder, type: ir.F16Type, value: float
    ) -> None: ...
    @overload
    def __init__(
        self, builder: ir.AlloOpBuilder, type: ir.F32Type, value: float
    ) -> None: ...
    @overload
    def __init__(
        self, builder: ir.AlloOpBuilder, type: ir.F64Type, value: float
    ) -> None: ...
    @overload
    def __init__(
        self, builder: ir.AlloOpBuilder, type: ir.BF16Type, value: float
    ) -> None: ...

class ConstantIndexOp(ConstantOp):
    def __init__(self, builder: ir.AlloOpBuilder, value: int) -> None: ...

class ConstantIntOp(ConstantOp):
    def __init__(
        self, builder: ir.AlloOpBuilder, type: ir.IntegerType, value: int
    ) -> None: ...

class DivFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class DivSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class DivUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class ExtFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class ExtSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class ExtUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class FPToSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class FPToUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class FmaOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, a: ir.Value, b: ir.Value, c: ir.Value
    ) -> None: ...

class IndexCastOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        dst_type: ir.Type,
        src: ir.Value,
    ) -> None: ...

class MaxNumFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MaxSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MaxUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MaximumFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MinNumFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MinSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MinUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MinimumFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MulFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class MulIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class NegFOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, val: ir.Value) -> None: ...

class OrIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class RemFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class RemSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class RemUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class SIToFPOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class SelectOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        condition: ir.Value,
        true_value: ir.Value,
        false_value: ir.Value,
    ) -> None: ...

class ShLIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class ShRSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class ShRUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class SubFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class SubIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class TruncFOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class TruncIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class UIToFPOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, src: ir.Value, dst_type: ir.Type
    ) -> None: ...

class XOrIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class CeilDivSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class CeilDivUIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...

class FloorDivSIOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, lhs: ir.Value, rhs: ir.Value
    ) -> None: ...
