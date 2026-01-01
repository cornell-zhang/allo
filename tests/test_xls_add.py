# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32


def add(a: int32, b: int32) -> int32:
    return a + b


def test_xls_add():
    s = allo.customize(add)
    print("=" * 60)
    print("MLIR Module:")
    print("=" * 60)
    print(s.module)

    # Note: This is a combinational function (no arrays), so use_memory has no effect
    # Both modes will produce the same output
    print("\n" + "=" * 60)
    print("COMBINATIONAL MODE (no arrays - use_memory has no effect):")
    print("=" * 60)
    mod_register = s.build(target="xls", project="add_register.prj", use_memory=False)
    print(mod_register)

    print("\n" + "=" * 60)
    print("COMBINATIONAL MODE with use_memory=True (same output):")
    print("=" * 60)
    mod_memory = s.build(target="xls", project="add_memory.prj", use_memory=True)
    print(mod_memory)


if __name__ == "__main__":
    test_xls_add()
