# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s

import hcl_mlir
from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d


def test_fixed():
    with Context() as ctx, Location.unknown() as loc:
        hcl_d.register_dialect(ctx)  # need to first register dialect
        fixed_type = hcl_d.FixedType.get(12, 6)
        ufixed_type = hcl_d.UFixedType.get(20, 12)
        print(fixed_type, ufixed_type)


def test_print():
    with Context() as ctx, Location.unknown() as loc:
        hcl_d.register_dialect(ctx)
        print(hcl_mlir.print_mlir_type(IntegerType.get_signless(1)))
        print(hcl_mlir.print_mlir_type(IntegerType.get_signless(8)))
        print(hcl_mlir.print_mlir_type(IntegerType.get_unsigned(16)))
        print(hcl_mlir.print_mlir_type(IntegerType.get_unsigned(12)))


if __name__ == "__main__":
    test_fixed()
    test_print()
