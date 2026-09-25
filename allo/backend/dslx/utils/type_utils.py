"""Type utilities for DSLX code generation.

Provides type conversion, float handling, and literal generation for DSLX.
"""

import struct
from ....utils import get_bitwidth_from_type

# Float type parameters: (exp_bits, frac_bits)
FLOAT_PARAMS = {
    "f16": (5, 10),
    "f32": (8, 23),
    "f64": (11, 52),
    "bf16": (8, 7),
}


def is_float_type(dtype: str) -> bool:
    """Check if dtype is a float type."""
    return dtype in FLOAT_PARAMS


def get_float_params(dtype: str):
    """Get float params (exp_bits, frac_bits) for a dtype."""
    return FLOAT_PARAMS.get(dtype)


def float_dtype_to_dslx(dtype: str) -> str:
    """Get DSLX type name for float dtype."""
    return dtype.upper()


def float_to_dslx_literal(value: float, dtype: str) -> str:
    """Convert Python float to DSLX literal for given dtype."""
    params = FLOAT_PARAMS.get(dtype)
    if not params:
        raise ValueError(f"unknown float type: {dtype}")
    exp_bits, frac_bits = params
    total_bits = 1 + exp_bits + frac_bits
    dslx_name = dtype.upper()

    # Convert to bits for the target format
    if dtype == "f64":
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    elif dtype == "f32":
        bits = struct.unpack(">I", struct.pack(">f", value))[0]
    elif dtype == "bf16":
        # bf16 is truncated f32
        bits = struct.unpack(">I", struct.pack(">f", value))[0] >> 16
    elif dtype == "f16":
        import numpy as np

        bits = int(np.float16(value).view(np.uint16))
    else:
        # For arbitrary floats, convert via f64 and manually repack
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]

    return f"apfloat::unflatten<{dslx_name}_EXP_SZ, {dslx_name}_FRAC_SZ>(u{total_bits}:{bits})"


def allo_dtype_to_dslx_type(dtype: str, shape: tuple = None) -> str:
    """Convert Allo dtype to DSLX type string.
    
    Args:
        dtype: Allo dtype string (e.g., "i32", "ui8", "f32", "index")
        shape: Optional tuple of dimensions
        
    Returns:
        DSLX type string (e.g., "s32", "u8", "F32", "s32[4][4]")
    """
    # Handle index type (used for loop indices) - treat as signed 32-bit
    if dtype == "index":
        base = "s32"
    # Handle float types
    elif dtype in FLOAT_PARAMS:
        base = float_dtype_to_dslx(dtype)
    else:
        bw = get_bitwidth_from_type(dtype)
        # Signed int
        if dtype.startswith("i"):
            base = f"s{bw}" if bw <= 64 else f"sN[{bw}]"
        # Unsigned int
        elif dtype.startswith("ui"):
            base = f"u{bw}" if bw <= 64 else f"uN[{bw}]"
        else:
            raise NotImplementedError(f"unsupported type: {dtype}")

    # Add shape suffix (reversed for DSLX row-major indexing)
    if shape:
        shape_suffix = "".join(f"[{d}]" for d in reversed(shape))
        return base + shape_suffix
    return base


def parse_dslx_dims_reversed(dslx_type: str) -> list:
    """Parse dimensions from DSLX type string (reversed)."""
    if "[" not in dslx_type:
        return []
    return list(reversed([int(x[:-1]) for x in dslx_type.split("[")[1:]]))


def format_shape_suffix(shape) -> str:
    """Format shape as DSLX array suffix."""
    return "".join(f"[{d}]" for d in reversed(shape))


def emit_float_defs(float_types: set) -> str:
    """Generate DSLX type definitions for a set of float types."""
    if not float_types:
        return ""
    lines = ["import apfloat;", ""]
    for dtype in sorted(float_types):
        params = FLOAT_PARAMS.get(dtype)
        if not params:
            continue
        exp, frac = params
        name = float_dtype_to_dslx(dtype)
        lines.append(f"pub const {name}_EXP_SZ = u32:{exp};")
        lines.append(f"pub const {name}_FRAC_SZ = u32:{frac};")
        lines.append(f"pub type {name} = apfloat::APFloat<{name}_EXP_SZ, {name}_FRAC_SZ>;")
        lines.append("")
    return "\n".join(lines)


def get_zero_literal(dslx_type: str) -> str:
    """Get zero literal for a DSLX type."""
    if "[" not in dslx_type:
        return f"{dslx_type}:0"
    
    base = dslx_type.split("[")[0]
    dims = list(reversed([int(x[:-1]) for x in dslx_type.split("[")[1:]]))
    leaf = f"{base}:0"

    def nest(dim_list):
        if len(dim_list) == 1:
            return "[" + ", ".join([leaf] * dim_list[0]) + "]"
        inner = nest(dim_list[1:])
        return "[" + ", ".join([inner] * dim_list[0]) + "]"

    return nest(dims)
