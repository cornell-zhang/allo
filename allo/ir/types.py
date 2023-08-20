# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, no-name-in-module

import numbers
from collections import OrderedDict

from hcl_mlir.ir import IntegerType, IndexType, F16Type, F32Type, F64Type
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.exceptions import DTypeError


class AlloType:
    def __init__(self, bits, fracs, name):
        if not isinstance(bits, numbers.Integral):
            raise DTypeError("Bitwidth must be an integer.")
        if not isinstance(fracs, numbers.Integral):
            raise DTypeError("Number of fractional bits must be an integer.")
        if bits > 2047:
            raise DTypeError("The maximum supported total bitwidth is 2047 bits.")
        if fracs > 255:
            raise DTypeError("The maximum supported fractional bitwidth is 255 bits.")
        self.bits = bits
        self.fracs = fracs
        self.name = name

    def build(self):
        # Required a MLIR context outside
        raise NotImplementedError

    def __getitem__(self, sizes):
        # Placeholder for type[*sizes]
        pass

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if other is None or not isinstance(other, AlloType):
            return False
        return self.name == other.name


class Index(AlloType):
    def __init__(self):
        super().__init__(32, 0, "index")

    def build(self):
        return IndexType.get()


class Int(AlloType):
    def __init__(self, bits):
        super().__init__(bits, 0, f"int{bits}")

    def build(self):
        return IntegerType.get_signless(self.bits)


class UInt(AlloType):
    def __init__(self, bits):
        super().__init__(bits, 0, f"uint{bits}")

    def build(self):
        return IntegerType.get_unsigned(self.bits)


class Float(AlloType):
    def __init__(self, bits):
        if bits == 16:
            super().__init__(16, 10, f"float{bits}")
            self.exponent = 5
        elif bits == 32:
            super().__init__(32, 23, f"float{bits}")
            self.exponent = 8
        elif bits == 64:
            super().__init__(64, 52, f"float{bits}")
            self.exponent = 11
        else:
            raise DTypeError("Only support float16, float32 and float64")

    # pylint: disable=inconsistent-return-statements
    def build(self):
        if self.bits == 16:
            return F16Type.get()
        if self.bits == 32:
            return F32Type.get()
        if self.bits == 64:
            return F64Type.get()


class Fixed(AlloType):
    def __init__(self, bits, fracs):
        super().__init__(bits, fracs, f"fixed({bits}, {fracs})")

    def build(self):
        raise hcl_d.FixedType.get(self.bits, self.fracs)


class UFixed(AlloType):
    def __init__(self, bits, fracs):
        super().__init__(bits, fracs, f"ufixed({bits}, {fracs})")

    def build(self):
        raise hcl_d.UFixedType.get(self.bits, self.fracs)


class Struct(AlloType):
    """A C-like struct

    The struct members are defined with a Python dictionary
    """

    def __init__(self, dtype_dict):
        self.bits = 0
        for name, dtype in dtype_dict.items():
            assert isinstance(dtype, AlloType), "dtype must be an AlloType"
            dtype_dict[name] = dtype
            self.bits += dtype.bits
        self.dtype_dict = OrderedDict(dtype_dict)
        super().__init__(self.bits, "struct")

    def __repr__(self):
        return "Struct(" + str(self.dtype_dict) + ")"

    def __getattr__(self, key):
        try:
            return self.dtype_dict[key]
        except KeyError as exc:
            raise DTypeError(key + " is not in struct") from exc

    def __getitem__(self, key):
        return self.__getattr__(key)

    def build(self):
        raise NotImplementedError("TODO")


bool = Int(1)
int1 = Int(1)
int8 = Int(8)
int16 = Int(16)
int32 = Int(32)
int64 = Int(64)
index = Index()
float16 = Float(16)
float32 = Float(32)
float64 = Float(64)
