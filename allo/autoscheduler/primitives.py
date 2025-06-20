# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from allo.customize import Schedule
from allo.ir.utils import MockBuffer

KINDS = ["reorder", "pipeline", "to"]


class SchedulePrimitive:
    def __init__(
        self, kind: str, args: list[Any] = None, kwargs: dict[str, Any] = None
    ):
        assert kind in KINDS
        self.kind = kind
        self.args = args or []
        self.kwargs = kwargs or {}

    def __repr__(self):
        match self.kind:
            case "reorder":
                return f"Reorder(order={self.args[0]})"
            case "pipeline":
                return f"Pipeline(axis={self.args[0]}, ii={self.kwargs['initiation_interval']})"
            case "to":
                return f"BufferToFifo(target={self.args[0]}, dst={self.args[1]})"

    def applyTo(self, schedule: Schedule):
        getattr(schedule, self.kind)(*self.args, **self.kwargs)

    @staticmethod
    def reorder(order: list[str]):
        return SchedulePrimitive("reorder", order)

    @staticmethod
    def pipeline(axis: str, ii: int):
        return SchedulePrimitive("pipeline", [axis], {"initiation_interval": ii})

    @staticmethod
    def buffer_to_fifo(target: MockBuffer, dst: str):
        return SchedulePrimitive("to", [target, dst])
    

class UnresolvedFIFOPrimitive():
    def __init__(self, buffer_name: str, node_id: int):
        self.buffer_name = buffer_name
        self.node_id = node_id
    
    def resolve(self, top_fn_name: str, node_to_fn: dict[int, str]) -> SchedulePrimitive:
        """Resolve the unresolved FIFO primitive to a concrete SchedulePrimitive."""
        return SchedulePrimitive.buffer_to_fifo(MockBuffer(top_fn_name, self.buffer_name), node_to_fn[self.node_id])

    
    
