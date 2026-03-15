from typing import Sequence, overload
from . import ir

class ExtractOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        tensor: ir.Value,
        indices: Sequence[ir.Value],
    ) -> None: ...

class InsertOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        tensor: ir.Value,
        indices: Sequence[ir.Value],
    ) -> None: ...

class SplatOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        shape: Sequence[int],
    ) -> None: ...

class CastOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, input: ir.Value, dst_type: ir.Type
    ) -> None: ...

class EmptyOp(ir.OpState):
    @overload
    def __init__(self, builder: ir.AlloOpBuilder, type: ir.Type) -> None: ...
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        shape: Sequence[int],
        element_type: ir.Type,
    ) -> None: ...

class ExtractSliceOp(ir.OpState):
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        tensor: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_type: ir.Type,
        tensor: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
        static_offsets: Sequence[int],
        static_sizes: Sequence[int],
        static_strides: Sequence[int],
    ) -> None: ...

class InsertSliceOp(ir.OpState):
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        tensor: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        tensor: ir.Value,
        offsets: Sequence[ir.Value],
        sizes: Sequence[ir.Value],
        strides: Sequence[ir.Value],
        static_offsets: Sequence[int],
        static_sizes: Sequence[int],
        static_strides: Sequence[int],
    ) -> None: ...

class GatherOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        res_type: ir.Type,
        tensor: ir.Value,
        indices: ir.Value,
        dims: Sequence[int],
        unique: bool = False,
    ) -> None: ...

class ScatterOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        value: ir.Value,
        tensor: ir.Value,
        indices: ir.Value,
        dims: Sequence[int],
        unique: bool = False,
    ) -> None: ...
