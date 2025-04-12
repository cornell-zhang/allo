# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from allo.customize import Schedule
from allo.ir.utils import MockBuffer

class SchedulePrimitive:
    def __init__(self, args: list[Any]=None, kwargs: dict[str, Any]=None):
        self.args = args or []
        self.kwargs = kwargs or {}
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.args}"
    
    def applyTo(self, _schedule: Schedule):
        pass

class Reorder(SchedulePrimitive):
    def __init__(self, order: list[str]):
        super().__init__(order)
    def applyTo(self, schedule: Schedule):
        schedule.reorder(*self.args)

class Pipeline(SchedulePrimitive):
    def __init__(self, axis: str, ii: int):
        super().__init__([axis], {"initiation_interval": ii})
    def applyTo(self, schedule: Schedule):
        schedule.pipeline(*self.args, **self.kwargs)

class BufferToFifo(SchedulePrimitive):
    def __init__(self, target: MockBuffer, dst: str):
        super().__init__([target, dst])
    def applyTo(self, schedule: Schedule):
        schedule.to(*self.args)