from enum import Enum
from typing import Sequence

from . import ir

class PartitionKind(Enum):
    Complete = ...
    Block = ...
    Cyclic = ...

Complete = PartitionKind.Complete
Block = PartitionKind.Block
Cyclic = PartitionKind.Cyclic

class PartitionAttr(ir.Attribute):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def get(
        context: ir.Context,
        sub_partitions: Sequence[tuple[int, int, int]],
    ) -> PartitionAttr: ...

class RenameOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        name: str,
    ) -> RenameOp: ...

class RaiseToAffineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> RaiseToAffineOp: ...

class OutlineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        kernel_name: str,
    ) -> OutlineOp: ...

class TagPipelineOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        ii: int,
    ) -> TagPipelineOp: ...

class TagUnrollOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> TagUnrollOp: ...

class LoopReorderOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        order: Sequence[int],
    ) -> LoopReorderOp: ...

class LoopSplitOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> LoopSplitOp: ...

class LoopTileOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factors: Sequence[int],
    ) -> LoopTileOp: ...

class LoopFlattenOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> LoopFlattenOp: ...

class ReuseAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        axis: ir.Value,
    ) -> ReuseAtOp: ...

class ComputeAtOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        producer: ir.Value,
        consumer_loop: ir.Value,
    ) -> ComputeAtOp: ...

class MatchValueOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        index: int,
        source_kind: int = 0,
    ) -> ir.Value: ...

class PartitionOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        partition: PartitionAttr,
    ) -> PartitionOp: ...
