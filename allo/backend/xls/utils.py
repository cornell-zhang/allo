# allo/allo/backend/xls/utils.py

import struct
from ...utils import get_bitwidth_from_type

# float type parameters: (exp_bits, frac_bits)
FLOAT_PARAMS = {
  "f16":  (5, 10),
  "f32":  (8, 23),
  "f64":  (11, 52),
  "bf16": (8, 7),
}

# check if dtype is a float type
def is_float_type(dtype: str) -> bool:
  return dtype in FLOAT_PARAMS

# get float params (exp_bits, frac_bits) for a dtype
def get_float_params(dtype: str):
  return FLOAT_PARAMS.get(dtype)

# get dslx type name for float dtype
def float_dtype_to_dslx(dtype: str) -> str:
  return dtype.upper()

# convert python float to dslx literal for given dtype
def float_to_dslx_literal(value: float, dtype: str) -> str:
  params = FLOAT_PARAMS.get(dtype)
  if not params:
    raise ValueError(f"unknown float type: {dtype}")
  exp_bits, frac_bits = params
  total_bits = 1 + exp_bits + frac_bits
  dslx_name = dtype.upper()
  # convert to f64 bits, then reinterpret for the target format
  if dtype == "f64":
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
  elif dtype == "f32":
    bits = struct.unpack('>I', struct.pack('>f', value))[0]
  elif dtype == "bf16":
    # bf16 is truncated f32
    bits = struct.unpack('>I', struct.pack('>f', value))[0] >> 16
  elif dtype == "f16":
    import numpy as np
    bits = int(np.float16(value).view(np.uint16))
  else:
    # for arbitrary floats, convert via f64 and manually repack
    # this is a placeholder - proper arbitrary float support needs more work
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
  return f"apfloat::unflatten<{dslx_name}_EXP_SZ, {dslx_name}_FRAC_SZ>(u{total_bits}:{bits})"

def allo_dtype_to_dslx_type(dtype: str, shape: str = None) -> str:
  # handle index type (used for loop indices) - treat as signed 32-bit
  if dtype == "index":
    return "s32"

  shape_suffix = "".join(f"[{d}]" for d in reversed(shape)) if shape else ""
  
  # handle float types
  if dtype in FLOAT_PARAMS:
    mapped = float_dtype_to_dslx(dtype)
    return mapped + shape_suffix if shape else mapped

  bw = get_bitwidth_from_type(dtype)

  # signed int
  if dtype.startswith("i"):
    mapped = f"s{bw}" if bw <= 64 else f"sN[{bw}]"
    return mapped + shape_suffix if shape else mapped

  # unsigned int
  if dtype.startswith("ui"):
    mapped = f"u{bw}" if bw <= 64 else f"uN[{bw}]"
    return mapped + shape_suffix if shape else mapped

  raise NotImplementedError(f"unsupported type: {dtype}")


def parse_dslx_dims_reversed(dslx_type: str) -> list:
  if '[' not in dslx_type:
    return []
  return list(reversed([int(x[:-1]) for x in dslx_type.split('[')[1:]]))


def format_shape_suffix(shape) -> str:
  return "".join(f"[{d}]" for d in reversed(shape))

# generate dslx type definitions for a set of float types
def emit_float_defs(float_types: set) -> str:
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