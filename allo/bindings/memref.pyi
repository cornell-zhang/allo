from typing import Sequence
from . import ir

class LoadOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        memref: ir.Value,
        indices: Sequence[ir.Value],
    ) -> None: ...

class StoreOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        memref: ir.Value,
        indices: Sequence[ir.Value],
    ) -> None: ...

class AllocOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, type: ir.MemRefType) -> None: ...

class SubViewOp(ir.OpState):
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        memref: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        memref: ir.Value,
        offsets: Sequence[int],
        sizes: Sequence[int],
        strides: Sequence[int],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_type: ir.Type,
        memref: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
        static_offsets: Sequence[int],
        static_sizes: Sequence[int],
        static_strides: Sequence[int],
    ) -> None: ...

class CopyOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        src: ir.Value,
        dst: ir.Value,
    ) -> None: ...

class GlobalOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        sym_name: str,
        visibility: str,
        res_type: ir.MemRefType,
        init_val: ir.Attribute,
        is_const: bool,
        alignment: int,
    ) -> None: ...

class GetGlobalOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_type: ir.Type,
        sym_name: str,
    ) -> None: ...

class ReshapeOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_type: ir.Type,
        memref: ir.Value,
        shape: ir.Value,
    ) -> None: ...

class TransposeOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, memref: ir.Value, perm: ir.AffineMap
    ) -> None: ...
