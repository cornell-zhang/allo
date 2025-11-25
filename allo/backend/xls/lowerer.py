# allo/allo/backend/xls/lowerer.py

from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d, arith as arith_d
from allo._mlir.ir import BlockArgument
from allo._mlir.ir import Module
from allo._mlir.passmanager import PassManager
from ...utils import get_func_inputs_outputs
from .utils import allo_dtype_to_dslx_type
from .instruction import InstructionEmitter

# base class
class DslxFuncLowererBase:
  def __init__(self, func: func_d.FuncOp):
    self.func = func

    # mappings
    self.value_map = {}  # MLIR SSA -> dslx identifier
    self.const_map = {}  # MLIR SSA -> dslx constant literal

    # input and output channels
    self.inputs    = {}  # mlir_arg_idx -> dslx_channel_name
    self.outputs   = {}  # mlir_ret_idx -> dslx_channel_name
    self.channels  = []  # declaration strings

    # naming variables
    self.tmp_counter = 0
    self.tok_counter = 0

    # instruction emitter
    self.inst_emitter = InstructionEmitter(self)

  # ---------------------------------------------------------------------
  # naming
  # ---------------------------------------------------------------------

  def new_tmp(self) -> str:
    name = f"tmp{self.tmp_counter}"
    self.tmp_counter += 1
    return name

  def new_tok(self) -> str:
    name = f"tok{self.tok_counter}"
    self.tok_counter += 1
    return name

  def register(self, mlir_name, dslx_name):
    self.value_map[mlir_name] = dslx_name

  def constant(self, mlir_name, dslx_name):
    self.const_map[mlir_name] = dslx_name

  def lookup(self, mlir_name):
    if mlir_name in self.const_map:
      return self.const_map[mlir_name]
    return self.value_map[mlir_name]

  # ---------------------------------------------------------------------
  # channels / config
  # ---------------------------------------------------------------------

  def _emit_channels(self) -> str:
    inputs, outputs = get_func_inputs_outputs(self.func)

    # input channels
    for idx, (dtype, shape) in enumerate(inputs):
      assert shape == (), "arrays not currently supported"
      dslx_type = allo_dtype_to_dslx_type(dtype)
      name = f"in{idx}"
      self.inputs[idx] = name
      self.channels.append(f"{name}: chan<{dslx_type}> in")

    # output channels
    for idx, (dtype, shape) in enumerate(outputs):
      assert shape == (), "arrays not currently supported"
      dslx_type = allo_dtype_to_dslx_type(dtype)
      name = f"out{idx}"
      self.outputs[idx] = name
      self.channels.append(f"{name}: chan<{dslx_type}> out")

    return "\n".join([f"  {s};" for s in self.channels])

  def _emit_config(self):
    names = ", ".join(list(self.inputs.values()) + list(self.outputs.values()))
    return f"  config({', '.join(self.channels)}) {{ ({names}) }}"

  # ---------------------------------------------------------------------
  # next{ ... } helpers: recv inputs + body
  # ---------------------------------------------------------------------

  def _emit_inputs(self):
    raise NotImplementedError

  def _emit_body(self):
    raise NotImplementedError

  # ---------------------------------------------------------------------
  # abstract hooks for subclasses
  # ---------------------------------------------------------------------

  def _emit_init(self):
    raise NotImplementedError

  def _emit_next(self):
    raise NotImplementedError

  # ---------------------------------------------------------------------
  # assemble proc
  # ---------------------------------------------------------------------

  def emit_proc(self):
    func = f"pub proc {self.func.name.value} {{\n"
    func += self._emit_channels() + "\n\n"
    func += self._emit_config() + "\n\n"
    func += self._emit_init() + "\n\n"
    func += self._emit_next()
    func += "}"
    return func


# combinational lower
class DslxCombLowerer(DslxFuncLowererBase):
  def _emit_inputs(self):
    lines = []
    for idx, mlir_arg in enumerate(self.func.arguments):
      chan = self.inputs[idx]
      var = self.new_tmp()
      tok = self.new_tok()
      lines.append(f"    let ({tok}, {var}) = recv(join(), {chan});")
      self.register(mlir_arg, var)
    return lines

  def _emit_body(self):
    lines = []
    block = self.func.body.blocks[0]
    for op in block.operations:
      lines.extend(self.inst_emitter.emit(op))
    return lines

  # ---------------------------------------------------------------------
  # init / next
  # ---------------------------------------------------------------------

  def _emit_init(self):
    return "  init { () }"

  def _emit_next(self):
    lines = ["  next(state: ()) {"]
    lines.extend(self._emit_inputs())
    lines.extend(self._emit_body())
    lines.append("  }\n")
    return "\n".join(lines)


# stateful lower
class DslxStatefulLowerer(DslxFuncLowererBase):

  def __init__(self, func):
    super(DslxStatefulLowerer, self).__init__(func)

    self.loop  = None
    self.state = [] 

    self._analyze_state()

  def _analyze_state(self):
    block = self.func.body.blocks[0]

    # find the scf.for loop
    for op in block.operations:
      if isinstance(op, scf_d.ForOp):
        self.loop = op
        break

    body = self.loop.body

    defined = set()
    used = set()

    # collect defs + uses in the loop body
    for op in body.operations:
      for r in op.results:
        defined.add(r)
      for operand in op.operands:
        used.add(operand)

    # used but not defined → candidate state
    candidates = [v for v in used if v not in defined]

    for v in candidates:
      # skip induction variable (block argument)
      if isinstance(v.owner, BlockArgument):
        continue

      # keep only memrefs as state
      vtype = getattr(v, "type", None)
      if vtype is not None and "memref" in str(vtype):
        self.state.append(v)


  # ---------------------------------------------------------------------
  # init / next
  # ---------------------------------------------------------------------

  def _emit_init(self):
    return "  init { /* state here */ }"

  def _emit_next(self):
    return "  next(state) { /* stateful logic */ }\n"


# ================================================================
# Module lowerer
# ================================================================
class DslxModuleLowerer:
  def __init__(self, module: Module, top_func_name: str):
    self.module = module
    self.func_lowerers = []

    for op in module.body.operations:
      if isinstance(op, func_d.FuncOp):

        # detect if function contains scf.for
        has_loop = any(
          inner_op.operation.name == "scf.for"
          for block in op.body.blocks
          for inner_op in block.operations
        )

        if has_loop:
          self.func_lowerers.append(DslxStatefulLowerer(op))
        else:
          self.func_lowerers.append(DslxCombLowerer(op))

  def emit_module(self):
    return "\n\n".join(fl.emit_proc() for fl in self.func_lowerers)


# ================================================================
# MLIR canonicalization + lowering entry
# ================================================================
def clean_mlir(module):
  with module.context:
    pm = PassManager.parse("builtin.module(canonicalize, sccp, cse, symbol-dce)")
    pm.run(module.operation)

def lower_mlir(module: Module, top_func_name: str, **kwargs) -> str:
  clean_mlir(module)
  return DslxModuleLowerer(module, top_func_name).emit_module()
