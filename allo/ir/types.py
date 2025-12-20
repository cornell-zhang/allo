# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, no-name-in-module

import re
import numbers
from collections import OrderedDict

from .._mlir.ir import (
    IntegerType,
    IndexType,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    MemRefType,
)
from .._mlir.dialects import allo as allo_d
from .._mlir.exceptions import DTypeError, DTypeWarning


class TypeAnnotation:
    """
    A wrapper class for type annotations that supports the @ operator.

    This allows annotations like: a: int32[32] @ Memory(impl="URAM")

    The @ operator is parsed at the AST level, but Python still tries to
    evaluate it at runtime. This class makes the runtime evaluation succeed
    by providing a __matmul__ method.
    """

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __matmul__(self, other):
        """Support the @ operator for memory/layout annotations."""
        # Return self to allow chaining, the actual spec is extracted from AST
        return self

    def __repr__(self):
        return f"{self.dtype}[{', '.join(str(s) for s in self.shape)}]"


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
        self.stateful = False

    def build(self):
        # Required a MLIR context outside
        raise NotImplementedError

    @staticmethod
    def isinstance(other):
        return isinstance(other, (AlloType, numbers.Number))

    def __getitem__(self, sizes):
        # Return a TypeAnnotation that supports the @ operator
        if isinstance(sizes, tuple):
            shape = sizes
        else:
            shape = (sizes,)
        return TypeAnnotation(self, shape)

    def __repr__(self):
        prefix = "stateful(" if self.stateful else ""
        suffix = ")" if self.stateful else ""
        return f"{prefix}{self.name}{suffix}"

    def __eq__(self, other):
        if other is None or not isinstance(other, AlloType):
            return False
        return self.name == other.name and self.stateful == other.stateful

    def __hash__(self):
        return hash((self.name, self.stateful))


def stateful(dtype: AlloType):
    """
    Marks a type as stateful, making it persistent across kernel invocations.

    Usage:
        state: stateful(int32)           # Stateful scalar
        counter: stateful(int32[10])     # Stateful array

    Parameters
    ----------
    dtype : AlloType
        The underlying type to mark as stateful

    Returns
    -------
    tuple
        A marker tuple that will be processed during type inference
    """
    return ("stateful", dtype)


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

    def __init__(self, bits, fracs, name="float"):
        """
        Constructs a floating point decimal number.

        Parameters
        ----------
        bits: The bitwidth of the float. This must be either 16, 32, or 64.
        |--------+------+----------+----------|
        | Format | Bits | Exponent | Fraction |
        |--------+------+----------+----------|
        | FP64   |   64 |       11 |       52 |
        | FP32   |   32 |        8 |       23 |
        | FP16   |   16 |        5 |       10 |
        | BF16   |   16 |        8 |        7 |
        |--------+------+----------+----------|
        """
        assert bits in {
            16,
            32,
            64,
        }, "Only support bfloat16, float16, float32 and float64"
        super().__init__(bits, fracs, name)
        self.exponent = bits - fracs - 1

    # pylint: disable=inconsistent-return-statements
    def build(self):
        if self.bits == 16:
            if self.exponent == 8:
                return BF16Type.get()
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
        prefix = "Stateful[" if self.stateful else ""
        suffix = "]" if self.stateful else ""
        return f"{prefix}Struct({self.dtype_dict}){suffix}"

    def __getattr__(self, key):
        try:
            return self.dtype_dict[key]
        except KeyError as exc:
            raise DTypeError(key + " is not in struct") from exc

    def __getitem__(self, key):
        return self.__getattr__(key)

    def build(self):
        types = []
        for _, dtype in self.dtype_dict.items():
            types.append(dtype.build())
        return allo_d.StructType.get(types)


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
        if len(self.shape) > 0:
            return MemRefType.get(self.shape, self.dtype.build())
        return self.dtype.build()

    def __repr__(self):
        shape = ", ".join(str(s) for s in self.shape)
        prefix = "Stateful[" if self.stateful else ""
        suffix = "]" if self.stateful else ""
        return f"{prefix}Stream({self.dtype}[{shape}]){suffix}"


def allo_type_from_mlir_type(mlir_type):
    """
    Reconstruct an Allo Type from an MLIR type.
    """
    # index
    if isinstance(mlir_type, IndexType):
        return Index()
    # integer
    if isinstance(mlir_type, IntegerType):
        # MLIR integer reconstructed as Int
        return Int(bits=mlir_type.width)
    # float
    if isinstance(mlir_type, BF16Type):
        return Float(16, 7)
    if isinstance(mlir_type, F16Type):
        return Float(16, 10)
    if isinstance(mlir_type, F32Type):
        return Float(32, 23)
    if isinstance(mlir_type, F64Type):
        return Float(64, 52)
    # fixme (Shihan): avoid using string matching and parsing
    pattern = re.compile(r"!allo\.(U)?Fixed<\s*(\d+)\s*,\s*(\d+)\s*>")
    m = pattern.fullmatch(str(mlir_type))
    if m:
        is_unsigned = m.group(1) is not None
        bits = int(m.group(2))
        fracs = int(m.group(3))
        if is_unsigned:
            return UFixed(bits, fracs)
        return Fixed(bits, fracs)
    raise TypeError(f"Cannot reconstruct Allo Type from MLIR type: {mlir_type}")


# boolean type should not be used as i1!
bool = UInt(1)
# signed integer types
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
# unsigned integer types
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
# index type
index = Index()
# floating point types
float16 = Float(16, 10, "f16")
float32 = Float(32, 23, "f32")
float64 = Float(64, 52, "f64")
# brain floating point
bfloat16 = Float(16, 7, "bf16")
