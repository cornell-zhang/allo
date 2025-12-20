# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import Int, int32, float32, stateful


def acc_stateless(x: Int(4), _sum: Int(4), rst: bool) -> Int(4):
    if rst:
        return 0
    else:
        return x + _sum


def acc_stateful(x: Int(4), rst: bool) -> Int(4):
    _sum: stateful(Int(4))
    if rst:
        _sum = 0
    else:
        _sum = _sum + x
    return _sum


s_stateless = allo.customize(acc_stateless)
s_stateful = allo.customize(acc_stateful)


mod_stateless = s_stateless.build(target="vitis_hls")
mod_stateful = s_stateful.build(target="vitis_hls")

print(mod_stateless)
print(mod_stateful)
