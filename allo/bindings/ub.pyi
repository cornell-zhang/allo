from . import ir

class PoisonOp(ir.OpState):
    def __init__(*args, **kwargs): ...
    @staticmethod
    def create(
        builder: ir.AlloOpBuilder,
        type: ir.Type,
    ) -> ir.Value: ...
