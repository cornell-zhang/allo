# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, no-name-in-module

from hcl_mlir.ir import IntegerType, IndexType, F16Type, F32Type, F64Type
from hcl_mlir.dialects import hcl as hcl_d


class AlloType:
    def __init__(self, bits, name):
        self.bits = bits
        self.name = name

    def build(self):
        # Required a MLIR context outside
        raise NotImplementedError

    def __getitem__(self, sizes):
        # Placeholder for type[*sizes]
        pass

    def __repr__(self):
        return self.name


class Index(AlloType):
    def __init__(self):
        super().__init__(32, "index")

    def build(self):
        return IndexType.get()


class Int(AlloType):
    def build(self):
        return IntegerType.get_signless(self.bits)


class UInt(AlloType):
    def build(self):
        return IntegerType.get_unsigned(self.bits)


class Float(AlloType):
    def __init__(self, bits, name):
        if bits not in [16, 32, 64]:
            raise RuntimeError("Unsupported floating point type")
        super().__init__(bits, name)

    # pylint: disable=inconsistent-return-statements
    def build(self):
        if self.bits == 16:
            return F16Type.get()
        if self.bits == 32:
            return F32Type.get()
        if self.bits == 64:
            return F64Type.get()


class Fixed(AlloType):
    def __init__(self, bits, frac, name):
        super().__init__(bits, name)
        self.frac = frac

    def build(self):
        raise hcl_d.FixedType.get(self.bits, self.frac)


class UFixed(AlloType):
    def __init__(self, bits, frac, name):
        super().__init__(bits, name)
        self.frac = frac

    def build(self):
        raise hcl_d.UFixedType.get(self.bits, self.frac)


bool = Int(1, "bool")
int1 = Int(1, "int1")
int8 = Int(8, "int8")
int16 = Int(16, "int16")
int32 = Int(32, "int32")
int64 = Int(64, "int64")
index = Index()
float16 = Float(16, "float16")
float32 = Float(32, "float32")
float64 = Float(64, "float64")
