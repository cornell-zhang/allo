from typing import Sequence

from . import ir

class OperationType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context, op_name: str) -> OperationType: ...

class ParamType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context, type: ir.Type) -> ParamType: ...

class AnyOpType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context) -> AnyOpType: ...

class AnyParamType(ir.Type):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def get(context: ir.Context) -> AnyParamType: ...

class NamedSequenceOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        name: str,
        root_type: ir.Type,
        result_types: Sequence[ir.Type],
    ) -> NamedSequenceOp: ...
    @property
    def entry_block(self) -> ir.Block: ...
    def get_arg_at(self, index: int) -> ir.BlockArgument: ...

class YieldOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, operands: Sequence[ir.Value]) -> YieldOp: ...

class ApplyCSEOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyCSEOp: ...

class ApplyDCEOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyDCEOp: ...

class ApplyCanonicalizationOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder) -> ApplyCanonicalizationOp: ...

class ApplyLICMOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyLICMOp: ...

class ApplyPatternsOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(builder: ir.AlloOpBuilder, target: ir.Value) -> ApplyPatternsOp: ...
    @property
    def body(self) -> ir.Block: ...

class ApplyRegisteredPassOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        pass_name: str,
        pass_options: ir.DictionaryAttr,
        dynamic_args: Sequence[ir.Value],
    ) -> ApplyRegisteredPassOp: ...

class MatchOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        res_type: ir.Type,
        op_names: Sequence[str],
        op_attrs: ir.DictionaryAttr | None = None,
    ) -> ir.Value: ...

class LoopUnrollOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        target: ir.Value,
        factor: int,
    ) -> LoopUnrollOp: ...

class MergeHandlesOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        handles: Sequence[ir.Value],
        deduplicate: bool = True,
    ) -> MergeHandlesOp: ...

class SplitHandleOp(ir.OpState):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        handle: ir.Value,
        num_splits: int,
    ) -> SplitHandleOp: ...

def apply_transforms(
    payload: ir.Operation,
    transform_root: ir.Operation,
    transform_module: ir.ModuleOp,
) -> tuple[bool, str]: ...
