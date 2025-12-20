# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, stateful


def acc[T_in](x: "T_in") -> "T_in":
    state: stateful(T_in) = 0
    state = state + x
    return state


def top(x: int32, y: int32) -> int32:
    res1 = acc[int32, "A1"](x)
    res2 = acc[int32, "A2"](y)
    return res1 + res2


s1 = allo.customize(acc, instantiate=[int32])
s2 = allo.customize(acc, instantiate=[int32])
s = allo.customize(top)
s.compose(s1, id="A1")
s.compose(s2, id="A2")

print(s.module)

code = s.build(target="vhls")
print(code)
