# allo/allo/backend/xls/instruction.py

from allo._mlir.ir import Operation
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import func as func_d

class InstructionEmitter:
  def __init__(self, parent):
    self.p = parent

  def dslx_type(self, op):
    sgn =  "u" if any(a.name == "unsigned" for a in op.attributes) else "s"
    bw = op.result.type.width
    return f"{sgn}{bw}" if (bw <= 64) else f"{sgn}N[{bw}]"

  def emit(self, op: Operation) -> list[str]:
    if isinstance(op, arith_d.ConstantOp):
      return self._emit_constant(op)
    elif isinstance(op, arith_d.AddIOp):
      return self._emit_add(op)
    elif isinstance(op, arith_d.MulIOp):
      return self._emit_mul(op)
    elif isinstance(op, arith_d.ExtUIOp):
      return self._emit_extui(op)
    elif isinstance(op, arith_d.ExtSIOp):
      return self._emit_extsi(op)
    elif isinstance(op, arith_d.TruncIOp):
      return self._emit_trunc(op)
    elif isinstance(op, func_d.ReturnOp):
      return self._emit_return(op)

    raise NotImplementedError(f"not implemented: {repr(op)}")

  def _emit_constant(self, op: arith_d.ConstantOp) -> list[str]:
    tmp = self.p.new_tmp()
    value = op.value.value
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = {value};"]

  def _emit_add(self, op: arith_d.AddIOp) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({lhs} + {rhs});"]

  def _emit_mul(self, op: arith_d.MulIOp) -> list[str]:
    lhs = self.p.lookup(op.lhs)
    rhs = self.p.lookup(op.rhs)
    tmp = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({lhs} * {rhs});"]
  
  def _emit_prec(self, op) -> list[str]:
    src   = self.p.lookup(op.operands[0])
    tmp   = self.p.new_tmp()
    self.p.register(op.result, tmp)
    return [f"    let {tmp} = ({src} as {self.dslx_type(op)});"]
  
  def _emit_extui(self, op: arith_d.ExtUIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_extsi(self, op: arith_d.ExtSIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_trunc(self, op: arith_d.TruncIOp) -> list[str]:
    return self._emit_prec(op)
  
  def _emit_return(self, op: func_d.ReturnOp) -> list[str]:
    lines = [f"    let tok = join({", ".join([f"tok{i}" for i in range(self.p.tok_counter)])});"]
    for idx, operand in enumerate(op.operands):
      src = self.p.lookup(operand)
      out_chan = self.p.outputs[idx]
      lines.append(f"    send(tok, {out_chan}, {src});")
    return lines

