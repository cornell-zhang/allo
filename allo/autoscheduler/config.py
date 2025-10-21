# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutoschedulerConfig:
    dsp_factors: dict[str, int] = field(
        default_factory=lambda: {
            "arith.mulf": 3,
            "arith.addf": 1,
            "arith.subf": 1,
            "arith.divf": 14,
            "arith.remf": 14,
            "arith.muli": 1,
            "arith.addi": 0,
            "arith.subi": 0,
            "arith.divi": 8,
            "arith.divu": 8,
            "arith.remi": 8,
            "arith.remu": 8,
        }
    )
    dsp_limit: int = 2560
    tiling_limit: int = 4
    mem_w_ports: Optional[int] = None
    mem_r_ports: Optional[int] = None
    kind: Optional[str] = None
    debug_point: Optional[str] = None
    debug_lp: bool = False
    verify: bool = False
    verbose: bool = True

    @staticmethod
    def builder():
        """Start a new builder chain."""
        return AutoschedulerConfig()

    def with_dsp_factors(self, factors: dict[str, int]):
        self.dsp_factors = factors
        return self

    def with_dsp_limit(self, limit: int):
        self.dsp_limit = limit
        return self

    def with_tiling_limit(self, limit: int):
        self.tiling_limit = limit
        return self

    def with_mem_w_ports(self, ports: int):
        self.mem_w_ports = ports
        return self

    def with_mem_r_ports(self, ports: int):
        self.mem_r_ports = ports
        return self

    def with_kind(self, kind: str):
        self.kind = kind
        return self

    def with_debug_point(self, point: str):
        self.debug_point = point
        return self

    def enable_debug_lp(self):
        self.debug_lp = True
        return self

    def disable_debug_lp(self):
        self.debug_lp = False
        return self

    def enable_verify(self):
        self.verify = True
        return self

    def disable_verify(self):
        self.verify = False
        return self

    def with_verbose(self, verbose: bool = True):
        self.verbose = verbose
        return self
