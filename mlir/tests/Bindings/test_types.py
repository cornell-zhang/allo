# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

import allo_mlir
from allo_mlir.ir import *
from allo_mlir.dialects import allo as allo_d


def test_fixed():
    with Context() as ctx, Location.unknown() as loc:
        allo_d.register_dialect(ctx)  # need to first register dialect
        fixed_type = allo_d.FixedType.get(12, 6)
        ufixed_type = allo_d.UFixedType.get(20, 12)
        print(fixed_type, ufixed_type)


def test_print():
    with Context() as ctx, Location.unknown() as loc:
        allo_d.register_dialect(ctx)
        print(allo_mlir.print_mlir_type(IntegerType.get_signless(1)))
        print(allo_mlir.print_mlir_type(IntegerType.get_signless(8)))
        print(allo_mlir.print_mlir_type(IntegerType.get_unsigned(16)))
        print(allo_mlir.print_mlir_type(IntegerType.get_unsigned(12)))


if __name__ == "__main__":
    test_fixed()
    test_print()
