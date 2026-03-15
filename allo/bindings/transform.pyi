from enum import Enum
from typing import Sequence

from . import ir

class OperationType(ir.Type):
    @staticmethod
    def get(context: ir.Context, op_name: str) -> OperationType: ...

class ParamType(ir.Type):
    @staticmethod
    def get(context: ir.Context, type: ir.Type) -> ParamType: ...

class AnyOpType(ir.Type):
    @staticmethod
    def get(context: ir.Context) -> AnyOpType: ...

class AnyParamType(ir.Type):
    @staticmethod
    def get(context: ir.Context) -> AnyParamType: ...

class AnnotateOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        name: str,
        value: ir.Attribute,
    ) -> None: ...

class GetDefiningOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class NamedSequenceOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        name: str,
        root_type: ir.Type,
        res_types: Sequence[ir.Type],
    ) -> None: ...
    def get_entry_block(self) -> ir.Block: ...
    def get_arg_at(self, idx: int) -> ir.BlockArgument: ...

class YieldOp(ir.OpState):
    def __init__(
        self, builder: ir.AlloOpBuilder, operands: Sequence[ir.Value]
    ) -> None: ...

class ApplyCSEOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class ApplyDCEOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class ApplyCanonicalizationOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder) -> None: ...

class ApplyLICMOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class ApplyPatternsOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...
    def get_body(self) -> ir.Block: ...

class ApplyRegisteredPassOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        pass_name: str,
        pass_options: ir.DictionaryAttr,
        dynamic_args: Sequence[ir.Value],
    ) -> None: ...

class MatchOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        res_type: ir.Type,
        op_names: Sequence[str],
        op_attrs: ir.DictionaryAttr | None = None,
    ) -> None: ...

class PartitionKind(Enum):
    Complete = 0
    Block = 1
    Cyclic = 2

Complete = PartitionKind.Complete
Block = PartitionKind.Block
Cyclic = PartitionKind.Cyclic

class PartitionAttr(ir.Attribute):
    @staticmethod
    def get(
        context: ir.Context,
        sub_partitions: Sequence[tuple[int, PartitionKind, int]],
    ) -> PartitionAttr: ...

class LoopUnrollOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> None: ...

class MergeHandlesOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        handles: Sequence[ir.Value],
        deduplicate: bool = True,
    ) -> None: ...

class SplitHandleOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        handle: ir.Value,
        num_splits: int,
    ) -> None: ...

class RenameOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        name: str,
    ) -> None: ...

class RaiseToAffineOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class OutlineOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        kernel_name: str,
    ) -> None: ...

class TagPipelineOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        ii: int,
    ) -> None: ...

class TagUnrollOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> None: ...

class LoopReorderOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        order: Sequence[int],
    ) -> None: ...

class LoopSplitOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> None: ...

class LoopTileOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factors: Sequence[int],
    ) -> None: ...

class LoopFlattenOp(ir.OpState):
    def __init__(self, builder: ir.AlloOpBuilder, target: ir.Value) -> None: ...

class ReuseAtOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        axis: ir.Value,
        ring: bool = False,
    ) -> None: ...

class ComputeAtOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        producer: ir.Value,
        consumer_loop: ir.Value,
    ) -> None: ...

class BufferAtOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        axis: ir.Value,
    ) -> None: ...

class MatchValueOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        index: int,
        source_kind: int = 0,
    ) -> None: ...

class PartitionOp(ir.OpState):
    def __init__(
        self,
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        partition: PartitionAttr,
    ) -> None: ...

def apply_transforms(
    payload: ir.Operation,
    transform_root: ir.Operation,
    transform_module: ir.ModuleOp,
) -> tuple[bool, str]: ...
