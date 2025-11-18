from ...utils import get_bitwidth_from_type


def allo_dtype_to_dslx_type(dtype: str) -> str:
  bw = get_bitwidth_from_type(dtype)

  # signed int
  if dtype.startswith("i"):
    return f"s{bw}"

  # unsigned int
  if dtype.startswith("ui"):
    return f"u{bw}"
  
  raise NotImplementedError("only support integers for now")