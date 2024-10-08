# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, no-name-in-module

import numbers
from collections import OrderedDict

from .._mlir.ir import (
    IntegerType,
    IndexType,
    F16Type,
    F32Type,
    F64Type,
    MemRefType,
    StringAttr,
)
from .._mlir.dialects import allo as allo_d
from .._mlir.exceptions import DTypeError, DTypeWarning


class AlloType:
    def __init__(self, bits, fracs, name):
        if not isinstance(bits, numbers.Integral):
            raise DTypeError("Bitwidth must be an integer.")
        if not isinstance(fracs, numbers.Integral):
            raise DTypeError("Number of fractional bits must be an integer.")
        if bits > 512:
            DTypeWarning(
                "NumPy does not support bitwidths larger than 512, so you cannot use NumPy for simulation, but you can still generate the corresponding MLIR module."
            ).warn()
        if fracs > 255:
            raise DTypeError("The maximum supported fractional bitwidth is 255 bits.")
        self.bits = bits
        self.fracs = fracs
        self.name = name

    def build(self):
        # Required a MLIR context outside
        raise NotImplementedError

    @staticmethod
    def isinstance(other):
        return isinstance(other, (AlloType, numbers.Number))

    def __getitem__(self, sizes):
        # Placeholder for type[*sizes]
        pass

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if other is None or not isinstance(other, AlloType):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Index(AlloType):
    """
    An array index.
    """

    def __init__(self):
        super().__init__(32, 0, "index")

    def build(self):
        return IndexType.get()

    @staticmethod
    def isinstance(other):
        return isinstance(other, (Index, int))


class Int(AlloType):
    """
    An integer of variable bitwidth.
    """

    def __init__(self, bits):
        """
        Constructs an integer.

        Parameters
        ----------
        bits: The bitwidth of the integer.
        """
        super().__init__(bits, 0, f"i{bits}")

    def build(self):
        return IntegerType.get_signless(self.bits)

    @staticmethod
    def isinstance(other):
        return isinstance(other, (Int, int))


class UInt(AlloType):
    """
    An unsigned integer of variable bitwidth.
    """

    def __init__(self, bits):
        """
        Constructs an unsigned integer.

        Parameters
        ----------
        bits: The bitwidth of the integer.
        """
        super().__init__(bits, 0, f"ui{bits}")

    def build(self):
        # A bit hacky here: Since the MLIR code dialect does not support
        # unsigned integers as arguments, we use the signless integer type,
        # label it in the IR with attributes, and then cast it to unsigned
        # in the codegen.
        return IntegerType.get_signless(self.bits)

    @staticmethod
    def isinstance(other):
        return isinstance(other, UInt) or (isinstance(other, int) and other >= 0)


class Float(AlloType):
    """
    A floating point decimal number.
    """

    def __init__(self, bits):
        """
        Constructs a floating point decimal number.

        Parameters
        ----------
        bits: The bitwidth of the float. This must be either 16, 32, or 64.
        """
        if bits == 16:
            super().__init__(16, 10, f"f{bits}")
            self.exponent = 5
        elif bits == 32:
            super().__init__(32, 23, f"f{bits}")
            self.exponent = 8
        elif bits == 64:
            super().__init__(64, 52, f"f{bits}")
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

    @staticmethod
    def isinstance(other):
        return isinstance(other, (Float, float))


class Fixed(AlloType):
    """
    A fixed point decimal.
    """

    def __init__(self, bits, fracs):
        """
        Constructs a fixed point decimal.

        Parameters
        ----------
        bits: The bitwidth of the number.

        frac: The number of fraction bits in the decimal.
        """

        super().__init__(bits, fracs, f"fixed({bits}, {fracs})")

    def build(self):
        return allo_d.FixedType.get(self.bits, self.fracs)


class UFixed(AlloType):
    """
    An unsigned fixed point decimal.
    """

    def __init__(self, bits, fracs):
        """
        Constructs an unsigned fixed point decimal.

        Parameters
        ----------
        bits: The bitwidth of the number.

        frac: The number of fraction bits in the decimal.
        """
        super().__init__(bits, fracs, f"ufixed({bits}, {fracs})")

    def build(self):
        return allo_d.UFixedType.get(self.bits, self.fracs)


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
        super().__init__(self.bits, 0, "struct")

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


class Stream(AlloType):
    """
    A FIFO type. Schedules using the `dataflow` schedule may find using this improves parallelism.
    """

    def __init__(self, dtype, shape, depth=2, size=1):
        assert isinstance(dtype, AlloType), f"dtype must be an AlloType, got {dtype}"
        self.dtype = dtype
        self.shape = shape
        self.depth = depth
        self.size = size
        super().__init__(0, 0, f"stream<{dtype}>")

    def build(self):
        return MemRefType.get(
            self.shape if len(self.shape) > 0 else (1,),
            self.dtype.build(),
            None,
            StringAttr.get(f"stream:{self.depth}"),
        )

    def __repr__(self):
        shape = ", ".join(str(s) for s in self.shape)
        return f"Stream({self.dtype}[{shape}])"


bool = Int(1)
int1 = Int(1)
int8 = Int(8)
int16 = Int(16)
int32 = Int(32)
int64 = Int(64)
int128 = Int(128)
int256 = Int(256)
int512 = Int(512)
int2 = Int(2)
int3 = Int(3)
int4 = Int(4)
int5 = Int(5)
int6 = Int(6)
int7 = Int(7)
int9 = Int(9)
int10 = Int(10)
int11 = Int(11)
int12 = Int(12)
int13 = Int(13)
int14 = Int(14)
int15 = Int(15)
uint1 = UInt(1)
uint8 = UInt(8)
uint16 = UInt(16)
uint32 = UInt(32)
uint64 = UInt(64)
uint128 = UInt(128)
uint256 = UInt(256)
uint512 = UInt(512)
uint2 = UInt(2)
uint3 = UInt(3)
uint4 = UInt(4)
uint5 = UInt(5)
uint6 = UInt(6)
uint7 = UInt(7)
uint9 = UInt(9)
uint10 = UInt(10)
uint11 = UInt(11)
uint12 = UInt(12)
uint13 = UInt(13)
uint14 = UInt(14)
uint15 = UInt(15)
index = Index()
float16 = Float(16)
float32 = Float(32)
float64 = Float(64)
