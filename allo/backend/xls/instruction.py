# allo/allo/backend/xls/instruction.py

from allo._mlir.ir import Operation, IndexType
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.dialects import affine as affine_d
from .utils import allo_dtype_to_dslx_type

class InstructionEmitter:
  def __init__(self, parent):
    self.p = parent

  def dslx_type(self, op):
    result_type = op.result.type
    # Handle IndexType (used for loop indices) - treat as signed 32-bit
    if isinstance(result_type, IndexType):
      return "s32"
    sgn =  "u" if any(a.name == "unsigned" for a in op.attributes) else "s"
    bw = result_type.width
    return f"{sgn}{bw}" if (bw <= 64) else f"{sgn}N[{bw}]"

  def emit(self, op: Operation) -> list[str]:
    if isinstance(op, arith_d.ConstantOp):
      return self._emit_constant(op)
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
  
  # ---------------------------------------------------------------------
  # helpers
  # ---------------------------------------------------------------------

  def _binary_op(self, op, opcode) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({lhs} {opcode} {rhs});"]
  
  def _emit_prec(self, op) -> list[str]:
    src   = self.p.lookup(op.operands[0])
    tmp   = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({src} as {self.dslx_type(op)});"]
  
  # ---------------------------------------------------------------------
  # instruction
  # ---------------------------------------------------------------------

  def _emit_constant(self, op: arith_d.ConstantOp) -> list[str]:
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
  
  def _emit_alloc(self, op: memref_d.AllocOp) -> list[str]:
    # For temporary memrefs inside loops, we just track them
    # The actual value will be set by affine.store
    return []
  
  def _emit_affine_store(self, op: affine_d.AffineStoreOp) -> list[str]:
    # Store the value to the memref - register the memref with the stored value
    stored_val = self.p.lookup(op.operands[0])
    memref = op.operands[1]
    # Register the memref so loads can find the value
    self.p.register(memref, stored_val)
    return []
  
  def _emit_affine_load(self, op: affine_d.AffineLoadOp) -> list[str]:
    # Load from the memref - look up the stored value
    memref = op.operands[0]
    stored_val = self.p.lookup(memref)
    # Register the load result with the same value
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

