from allo._mlir.dialects import func as func_d
from allo._mlir.ir import Module
from ...utils import get_func_inputs_outputs
from .utils import allo_dtype_to_dslx_type
from .instruction import InstructionEmitter

class DslxFuncLowerer:
  def __init__(self, func: func_d.FuncOp):
    self.func = func

    # mappings
    self.value_map = {} # MLIR value -> dslx identifier

    # input and output channels
    self.inputs   = {}    # mlir_arg -> (dslx_chan, dslx_type)
    self.outputs  = {}    # mlir_ret -> (dslx_chan, dslx_type)
    self.channels = []    # declaration strings

    # naming variables
    self.tmp_counter = 0
    self.tok_counter = 0

    # instruction emitter
    self.inst_emitter = InstructionEmitter(self)

  # ---------------------------------------------------------------------
  # naming
  # ---------------------------------------------------------------------

  # temp variables used in dslx code
  def new_tmp(self) -> str:
    name = f"tmp{self.tmp_counter}"
    self.tmp_counter += 1
    return name
  
  # token variables used in dslx code
  def new_tok(self) -> str:
    name = f"tok{self.tok_counter}"
    self.tok_counter += 1
    return name
  
  # register a mlir <-> dslx variable mapping
  def register(self, mlir_name: str, dslx_name: str):
    self.value_map[mlir_name] = dslx_name

  # lookup variable mapping
  def lookup(self, mlir_name: str) -> str:
    return self.value_map[mlir_name]
  
  # ---------------------------------------------------------------------
  # channels / config / init
  # ---------------------------------------------------------------------

  def _emit_channels(self) -> str:
    inputs, outputs = get_func_inputs_outputs(self.func)

    # input channels
    for idx, (dtype, shape) in enumerate(inputs):
      assert shape == (), "arrays not currently supported"
      dslx_type = allo_dtype_to_dslx_type(dtype)
      dslx_name = f"in{idx}"

      # track input
      self.inputs[idx] = dslx_name

      # emit declaration
      self.channels.append(f"{dslx_name}: chan<{dslx_type}> in")

    # input channels
    for idx, (dtype, shape) in enumerate(outputs):
      assert shape == (), "arrays not currently supported"
      dslx_type = allo_dtype_to_dslx_type(dtype)
      dslx_name = f"out{idx}"

      # track output
      self.outputs[idx] = dslx_name

      # emit declaration
      self.channels.append(f"{dslx_name}: chan<{dslx_type}> out")

    return "\n".join([f"  {s};" for s in self.channels])
  
  def _emit_config(self):
    names = ", ".join(list(self.inputs.values()) + list(self.outputs.values()))
    return f"  config({", ".join(self.channels)}) {{ ({names}) }}"
  
  def _emit_init(self):
    # no state for now...
    return "  init { () }"
  
  # ---------------------------------------------------------------------
  # next{ ... } = receives + body + sends
  # ---------------------------------------------------------------------
  
  def _emit_inputs(self):
    lines = []

    for idx, mlir_arg in enumerate(self.func.arguments):
      chan_name = self.inputs[idx]
      var_name  = self.new_tmp()
      tok_name  = self.new_tok()

      lines.append(f"    let ({tok_name}, {var_name}) = recv(join(), {chan_name});")
      self.register(mlir_arg, var_name)
    
    return lines
  
  def _emit_body(self):
    # walk ir
    lines = []
    block = self.func.body.blocks[0]

    for op in block.operations:
      lines.extend(self.inst_emitter.emit(op))

    return lines
  
  def _emit_next(self):
    lines = ["  next(state: ()) {"]
    lines.extend(self._emit_inputs())
    lines.extend(self._emit_body())
    lines.append("  }\n")
    return "\n".join(lines)

  def emit_proc(self):
    func = f"pub proc {self.func.name.value} {{\n"

    # add channels
    func += self._emit_channels()
    func += "\n\n"

    # add config
    func += self._emit_config()
    func += "\n\n"

    # add init
    func += self._emit_init()
    func += "\n\n"

    # add next
    func += self._emit_next()

    func += "}"

    return func

class DslxModuleLowerer:
  def __init__(self, module: Module, top_func_name: str):
    self.module = module
    self.func_lowerers = []

    for op in module.body.operations:
      if isinstance(op, func_d.FuncOp):
        self.func_lowerers.append(DslxFuncLowerer(op))

  def emit_module(self) -> str:
    functions = [fl.emit_proc() for fl in self.func_lowerers]
    return "\n\n".join(functions)

def lower_mlir(module: Module, top_func_name: str, **kwargs) -> str:
  lowerer = DslxModuleLowerer(module, top_func_name)
  return lowerer.emit_module()
