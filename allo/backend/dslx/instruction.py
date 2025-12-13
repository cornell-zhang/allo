# allo/allo/backend/xls/instruction.py
# emit dslx code for individual mlir operations.

from allo._mlir.ir import Operation, IndexType, F16Type, F32Type, F64Type, BF16Type
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.dialects import affine as affine_d
from .utils import allo_dtype_to_dslx_type, float_to_dslx_literal, float_dtype_to_dslx

# mlir float type to dtype string mapping
MLIR_FLOAT_MAP = {
  F16Type: "f16", F32Type: "f32", F64Type: "f64", BF16Type: "bf16"
}

# check if mlir type is a float type and return dtype string
def get_float_dtype(mlir_type):
  for ftype, dtype in MLIR_FLOAT_MAP.items():
    if ftype.isinstance(mlir_type):
      return dtype
  return None

# emits dslx code for mlir arithmetic and control operations
class InstructionEmitter:
  def __init__(self, parent):
    self.p = parent
    self.float_types_used = set()

  # convert mlir type to dslx type string
  def dslx_type(self, op):
    result_type = op.result.type
    if isinstance(result_type, IndexType):
      return "s32"
    # handle float types
    dtype = get_float_dtype(result_type)
    if dtype:
      self.float_types_used.add(dtype)
      return float_dtype_to_dslx(dtype)
    sgn = "u" if any(a.name == "unsigned" for a in op.attributes) else "s"
    bw = result_type.width
    return f"{sgn}{bw}" if (bw <= 64) else f"{sgn}N[{bw}]"

  # emit dslx code for an mlir operation
  def emit(self, op: Operation) -> list[str]:
    if isinstance(op, arith_d.ConstantOp):
      return self._emit_constant(op)
    # integer ops
    elif isinstance(op, arith_d.AddIOp):
      return self._emit_add(op)
    elif isinstance(op, arith_d.SubIOp):
      return self._emit_sub(op)
    elif isinstance(op, arith_d.OrIOp):
      return self._emit_or(op)
    elif isinstance(op, arith_d.AndIOp):
      return self._emit_and(op)
    elif isinstance(op, arith_d.XOrIOp):
      return self._emit_xor(op)
    elif isinstance(op, arith_d.MulIOp):
      return self._emit_mul(op)
    elif isinstance(op, arith_d.RemSIOp):
      return self._emit_rem(op)
    elif isinstance(op, arith_d.FloorDivSIOp):
      return self._emit_div(op)
    elif isinstance(op, arith_d.CmpIOp):
      return self._emit_cmp(op)
    # float ops
    elif isinstance(op, arith_d.AddFOp):
      return self._emit_addf(op)
    elif isinstance(op, arith_d.SubFOp):
      return self._emit_subf(op)
    elif isinstance(op, arith_d.MulFOp):
      return self._emit_mulf(op)
    elif isinstance(op, arith_d.DivFOp):
      return self._emit_divf(op)
    elif isinstance(op, arith_d.CmpFOp):
      return self._emit_cmpf(op)
    elif isinstance(op, arith_d.NegFOp):
      return self._emit_negf(op)
    # common ops
    elif isinstance(op, arith_d.SelectOp):
      return self._emit_sel(op)
    elif isinstance(op, arith_d.ExtUIOp):
      return self._emit_extui(op)
    elif isinstance(op, arith_d.ExtSIOp):
      return self._emit_extsi(op)
    elif isinstance(op, arith_d.TruncIOp):
      return self._emit_trunc(op)
    elif isinstance(op, arith_d.IndexCastOp):
      return self._emit_index_cast(op)
    elif isinstance(op, scf_d.YieldOp):
      return self._emit_yield(op)
    elif isinstance(op, func_d.ReturnOp):
      return self._emit_return(op)
    elif isinstance(op, memref_d.AllocOp):
      return self._emit_alloc(op)
    elif isinstance(op, affine_d.AffineStoreOp):
      return self._emit_affine_store(op)
    elif isinstance(op, affine_d.AffineLoadOp):
      return self._emit_affine_load(op)

    raise NotImplementedError(f"not implemented: {repr(op)}")
  
  # Emit binary operation with given opcode.
  def _binary_op(self, op, opcode) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({lhs} {opcode} {rhs});"]
  
  # Emit precision cast operation.
  def _emit_prec(self, op) -> list[str]:
    src   = self.p.lookup(op.operands[0])
    tmp   = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({src} as {self.dslx_type(op)});"]
  
  def _emit_constant(self, op: arith_d.ConstantOp) -> list[str]:
    result_type = op.result.type
    # handle float constants
    dtype = get_float_dtype(result_type)
    if dtype:
      self.float_types_used.add(dtype)
      value_str = str(op.value).split(":")[0].strip()
      self.p.constant(op.result, float_to_dslx_literal(float(value_str), dtype))
      return []
    value = str(op.value).split(":", 1)[0].strip()
    dslx_prefix = allo_dtype_to_dslx_type(str(op.result.type))
    self.p.constant(op.result, f"{dslx_prefix}:{value}")
    return []

  def _emit_add(self, op: arith_d.AddIOp) -> list[str]:
    return self._binary_op(op, "+")
  
  def _emit_sub(self, op: arith_d.SubIOp) -> list[str]:
    return self._binary_op(op, "-")
  
  def _emit_and(self, op: arith_d.AndIOp) -> list[str]:
    return self._binary_op(op, "&")
  
  def _emit_or(self, op: arith_d.OrIOp) -> list[str]:
    return self._binary_op(op, "|")
  
  def _emit_xor(self, op: arith_d.XOrIOp) -> list[str]:
    return self._binary_op(op, "^")

  def _emit_mul(self, op: arith_d.MulIOp) -> list[str]:
    return self._binary_op(op, "*")
  
  def _emit_rem(self, op: arith_d.RemSIOp) -> list[str]:
    return self._binary_op(op, "%")
  
  def _emit_div(self, op: arith_d.FloorDivSIOp) -> list[str]:
    return self._binary_op(op, "/")

  def _emit_cmp(self, op: arith_d.CmpIOp):
    opcode = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", 
              ">", ">="][op.predicate.value]
    return self._binary_op(op, opcode)

  # float binary op using apfloat function
  def _float_binary_op(self, op, func_name) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    dtype = get_float_dtype(op.result.type)
    if dtype:
      self.float_types_used.add(dtype)
    return [f"    let {tmp} = apfloat::{func_name}({lhs}, {rhs});"]

  def _emit_addf(self, op: arith_d.AddFOp) -> list[str]:
    return self._float_binary_op(op, "add")

  def _emit_subf(self, op: arith_d.SubFOp) -> list[str]:
    return self._float_binary_op(op, "sub")

  def _emit_mulf(self, op: arith_d.MulFOp) -> list[str]:
    return self._float_binary_op(op, "mul")

  def _emit_divf(self, op: arith_d.DivFOp) -> list[str]:
    # apfloat doesn't have div, so we need to implement it differently
    # for now, raise not implemented
    raise NotImplementedError("float division not yet supported in xls backend")

  def _emit_negf(self, op: arith_d.NegFOp) -> list[str]:
    src = self.p.lookup(op.operands[0])
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    dtype = get_float_dtype(op.result.type)
    if dtype:
      self.float_types_used.add(dtype)
    return [f"    let {tmp} = apfloat::negate({src});"]

  def _emit_cmpf(self, op: arith_d.CmpFOp) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    dtype = get_float_dtype(op.lhs.type)
    if dtype:
      self.float_types_used.add(dtype)
    # cmpf predicates: 0=false, 1=oeq, 2=ogt, 3=oge, 4=olt, 5=ole, 6=one, 7=ord, etc.
    cmp_funcs = {1: "eq_2", 2: "gt_2", 3: "gte_2", 4: "lt_2", 5: "lte_2"}
    if op.predicate.value in cmp_funcs:
      return [f"    let {tmp} = apfloat::{cmp_funcs[op.predicate.value]}({lhs}, {rhs});"]
    raise NotImplementedError(f"float comparison predicate {op.predicate.value} not supported")
  
  def _emit_sel(self, op: arith_d.SelectOp):
    sel = self.p.lookup(op.operands[0])
    lhs = self.p.lookup(op.operands[1])
    rhs = self.p.lookup(op.operands[2])
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = if ({sel}) {{ {lhs} }} else {{ {rhs} }};"]

  def _emit_extui(self, op: arith_d.ExtUIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_extsi(self, op: arith_d.ExtSIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_trunc(self, op: arith_d.TruncIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_index_cast(self, op: arith_d.IndexCastOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_yield(self, op: scf_d.YieldOp) -> list[str]:
    return []
  
  # Track temporary memref allocation (value set by store).
  def _emit_alloc(self, op: memref_d.AllocOp) -> list[str]:
    return []
  
  # Register memref with stored value for later loads.
  def _emit_affine_store(self, op: affine_d.AffineStoreOp) -> list[str]:
    stored_val = self.p.lookup(op.operands[0])
    memref = op.operands[1]
    self.p.register(memref, stored_val)
    return []
  
  # Load value from memref by looking up registered value.
  def _emit_affine_load(self, op: affine_d.AffineLoadOp) -> list[str]:
    memref = op.operands[0]
    stored_val = self.p.lookup(memref)
    self.p.register(op.result, stored_val)
    return []
  
  def _emit_return(self, op: func_d.ReturnOp) -> list[str]:
    if (self.p.tok_counter > 1):
      lines = [f"    let tok = join({", ".join([f"tok{i}" for i in range(self.p.tok_counter)])});"]
      token = "tok"
    else:
      lines = []
      token = "tok0"
    
    for idx, operand in enumerate(op.operands):
      src = self.p.lookup(operand)
      out_chan = self.p.outputs[idx]
      lines.append(f"    send({token}, {out_chan}, {src});")
    return lines
