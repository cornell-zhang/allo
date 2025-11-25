# allo/allo/backend/xls/lowerer.py

from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d, arith as arith_d, affine as affine_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.ir import BlockArgument
from allo._mlir.ir import Module, MemRefType
from allo._mlir.passmanager import PassManager
from ...utils import get_func_inputs_outputs, get_dtype_and_shape_from_type
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
    self.input_types = []  # dslx types for inputs

    # naming variables
    self.tmp_counter = 0
    self.tok_counter = 0

    # instruction emitter
    self.inst_emitter = InstructionEmitter(self)
    
    # Register all constants upfront
    self._register_all_constants()

  def _register_all_constants(self):
    """Recursively find and register all constants in the function."""
    def visit_block(block):
      for op in block.operations:
        if isinstance(op, arith_d.ConstantOp):
          self.inst_emitter.emit(op)
        # Recurse into nested regions
        for region in op.regions:
          for blk in region.blocks:
            visit_block(blk)
    
    for block in self.func.body.blocks:
      visit_block(block)

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
    if mlir_name in self.value_map:
      return self.value_map[mlir_name]
    raise KeyError(f"Value not found: {mlir_name}")

  # ---------------------------------------------------------------------
  # channels / config
  # ---------------------------------------------------------------------

  def _emit_channels(self) -> str:
    inputs, outputs = get_func_inputs_outputs(self.func)

    # input channels
    self.input_types = []
    for idx, (dtype, shape) in enumerate(inputs):
      assert shape == (), "arrays not currently supported"
      dslx_type = allo_dtype_to_dslx_type(dtype)
      self.input_types.append(dslx_type)
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
    func += self._emit_next() + "\n"
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


# stateful lower - handles FOR loops, WHILE loops, and nested loops
class DslxStatefulLowerer(DslxFuncLowererBase):

  def __init__(self, func):
    super(DslxStatefulLowerer, self).__init__(func)
    self.loop = None
    self.loop_type = None  # 'for' or 'while'
    self.state = []        # List of (memref_value, dslx_type, init_value)
    self.loops = []        # List of nested ForOps from outermost to innermost
    self.return_state_idx = 0  # Which state variable is returned
    self._analyze()

  def _analyze(self):
    block = self.func.body.blocks[0]
    for op in block.operations:
      if isinstance(op, scf_d.ForOp):
        self.loop, self.loop_type = op, 'for'
        break
      elif isinstance(op, scf_d.WhileOp):
        self.loop, self.loop_type = op, 'while'
        break
    if not self.loop:
      return

    # Collect ALL nested for loops (outermost to innermost)
    if self.loop_type == 'for':
      self._collect_nested_loops(self.loop)

    # Collect memrefs defined before loop
    memrefs = {op.result for op in block.operations 
               if isinstance(op, memref_d.AllocOp) and op.result != self.loop}
    
    # Find memrefs used in loop body
    # Note: scf.ForOp.body returns a Block directly, while scf.WhileOp.before/after are Regions
    if self.loop_type == 'while':
      body = list(self.loop.after.blocks)[0]
      used = self._collect_memrefs(body, memrefs)
      used |= self._collect_memrefs(list(self.loop.before.blocks)[0], memrefs)
    else:
      # For scf.ForOp, .body is already a Block, not a Region
      body = self.loop.body
      used = self._collect_memrefs(body, memrefs)

    for m in used:
      if "memref" in str(getattr(m, "type", "")):
        mt = MemRefType(m.type)
        dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
        self.state.append((m, allo_dtype_to_dslx_type(dtype), self._get_init(m)))

    # Find which state variable is returned
    self._find_return_state_idx(block)

  def _find_return_state_idx(self, block):
    """Find which state variable corresponds to the return value."""
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    for op in block.operations:
      if isinstance(op, func_d.ReturnOp):
        if len(op.operands) > 0:
          ret_val = op.operands[0]
          # The return value is typically loaded from a memref
          owner = ret_val.owner.opview if hasattr(ret_val.owner, 'opview') else ret_val.owner
          if isinstance(owner, affine_d.AffineLoadOp):
            memref_key = str(owner.operands[0])
            if memref_key in state_memrefs:
              self.return_state_idx = state_memrefs[memref_key]
              return
        break

  def _collect_nested_loops(self, loop_op):
    """Recursively collect nested for loops from outermost to innermost."""
    self.loops.append(loop_op)
    for op in loop_op.body.operations:
      if isinstance(op, scf_d.ForOp):
        self._collect_nested_loops(op)

  def _collect_memrefs(self, block, memrefs):
    used = set()
    for op in block.operations:
      for operand in op.operands:
        if operand in memrefs:
          used.add(operand)
      if isinstance(op, scf_d.ForOp):
        # For scf.ForOp, .body is already a Block
        used |= self._collect_memrefs(op.body, memrefs)
      elif isinstance(op, scf_d.IfOp):
        # Handle both API styles: then_block (Block) vs thenRegion (Region)
        if hasattr(op, 'then_block') and op.then_block is not None:
          then_blk = op.then_block
        else:
          then_blk = list(op.thenRegion.blocks)[0]
        used |= self._collect_memrefs(then_blk, memrefs)
        # Check for else block
        if hasattr(op, 'else_block') and op.else_block is not None:
          used |= self._collect_memrefs(op.else_block, memrefs)
        elif hasattr(op, 'elseRegion') and len(op.elseRegion.blocks) > 0:
          used |= self._collect_memrefs(list(op.elseRegion.blocks)[0], memrefs)
    return used

  def _get_init(self, memref):
    func_args = list(self.func.arguments)
    for op in self.func.body.blocks[0].operations:
      if isinstance(op, (scf_d.ForOp, scf_d.WhileOp)):
        break
      if isinstance(op, affine_d.AffineStoreOp) and op.operands[1] == memref:
        v = op.operands[0]
        # Check if it's a function argument
        if isinstance(v, BlockArgument):
          return f"__arg{v.arg_number}__"
        # Also check by comparing with function arguments directly
        for i, arg in enumerate(func_args):
          if v == arg:
            return f"__arg{i}__"
        # Check if it's a constant
        owner = v.owner.opview if hasattr(v.owner, 'opview') else v.owner
        if isinstance(owner, arith_d.ConstantOp):
          return str(owner.value).split(":")[0].strip()
    return "0"

  def _emit_body(self, ops, acc_names, lines, temp_memrefs=None):
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    if temp_memrefs is None:
      temp_memrefs = {}  # Track temporary memrefs inside loop body (str -> value)
    updated = {}
    
    for op in ops:
      if isinstance(op, arith_d.ConstantOp):
        pass  # Already registered in _register_all_constants
      elif isinstance(op, memref_d.AllocOp):
        temp_memrefs[str(op.result)] = None
      elif isinstance(op, affine_d.AffineLoadOp):
        mkey = str(op.operands[0])
        if mkey in state_memrefs:
          self.register(op.result, self.lookup(self.state[state_memrefs[mkey]][0]))
        elif mkey in temp_memrefs:
          self.register(op.result, temp_memrefs[mkey])
        else:
          self.inst_emitter.emit(op)
      elif isinstance(op, affine_d.AffineStoreOp):
        val, mkey = self.lookup(op.operands[0]), str(op.operands[1])
        if mkey in state_memrefs:
          i = state_memrefs[mkey]
          tmp = self.new_tmp()
          updated[i] = tmp
          lines.append(f"    let {tmp} = {val};")
          self.register(self.state[i][0], tmp)
        elif mkey in temp_memrefs:
          temp_memrefs[mkey] = val
        else:
          self.inst_emitter.emit(op)
      elif isinstance(op, scf_d.IfOp):
        cond = self.lookup(op.operands[0])
        saved = dict(self.value_map)
        then_lines, else_lines = [], []
        # Handle both API styles: then_block (Block) vs thenRegion (Region)
        if hasattr(op, 'then_block') and op.then_block is not None:
          then_blk = op.then_block
        else:
          then_blk = list(op.thenRegion.blocks)[0]
        then_upd = self._emit_body(then_blk.operations, acc_names, then_lines, temp_memrefs)
        self.value_map = dict(saved)
        # Check for else block
        else_blk = None
        if hasattr(op, 'else_block') and op.else_block is not None:
          else_blk = op.else_block
        elif hasattr(op, 'elseRegion') and len(op.elseRegion.blocks) > 0:
          else_blk = list(op.elseRegion.blocks)[0]
        else_upd = self._emit_body(else_blk.operations, acc_names, else_lines, temp_memrefs) if else_blk else {}
        lines.extend(then_lines + else_lines)
        for i in set(then_upd) | set(else_upd):
          tmp = self.new_tmp()
          lines.append(f"    let {tmp} = if ({cond}) {{ {then_upd.get(i, acc_names[i])} }} else {{ {else_upd.get(i, acc_names[i])} }};")
          updated[i] = tmp
          self.register(self.state[i][0], tmp)
      elif isinstance(op, scf_d.ForOp):
        # Skip nested ForOps - they are handled at the top level in _emit_next
        # with proper state machine indices for each loop level
        pass
      elif isinstance(op, scf_d.YieldOp):
        pass
      else:
        lines.extend(self.inst_emitter.emit(op))
    return updated

  def _emit_init(self):
    fields = []
    for _, _, init in self.state:
      if init and init.startswith("__arg") and init.endswith("__"):
        fields.append("0")  # Placeholder, will be set by recv
      else:
        fields.append(init or "0")
    if self.loop_type == 'for':
      num_loops = len(self.loops) if self.loops else 1
      # For each loop level: index_i, upper_bound_i
      for _ in range(num_loops):
        fields += ["0", "0"]
      fields.append("false")  # busy flag
    else:
      fields.append("false")
    return f"  init {{ ({', '.join(fields)}) }}"

  def _emit_next(self):
    n = len(self.state)
    acc_names = [f"acc{i}" for i in range(n)]
    num_loops = len(self.loops) if self.loops else 1
    
    # Handle WHILE loops separately (different structure)
    if self.loop_type == 'while':
      return self._emit_next_while(acc_names)
    
    # FOR loops (1D, 2D, ND - unified logic)
    bt = self.input_types[0] if self.input_types else "s32"
    
    # State: accumulators + (index_i, ub_i) for each loop + busy
    loop_types = [bt, bt] * num_loops
    state_type = "(" + ", ".join([t for _, t, _ in self.state] + loop_types + ["bool"]) + ")"
    state_vars = acc_names[:]
    for i in range(num_loops):
      state_vars += [f"index{i}", f"ub{i}"]
    state_vars.append("busy")

    lines = [f"  next(state: {state_type}) {{"]
    lines.append(f"    let ({', '.join(state_vars)}) = state;")

    # Receive inputs (upper bounds for each loop level)
    ubs = []
    for i, ch in enumerate(self.inputs.values()):
      tok, inp = self.new_tok(), self.new_tmp()
      default = f"ub{i}" if i < num_loops else "s32:0"
      lines.append(f"    let ({tok}, {inp}) = recv_if(join(), {ch}, !busy, {default});")
      ubs.append(inp)
      if i < len(list(self.func.arguments)):
        self.register(list(self.func.arguments)[i], inp)
    base_tok = tok if self.inputs else "join()"
    
    # Pad ubs if fewer inputs than loops (shouldn't happen normally)
    while len(ubs) < num_loops:
      ubs.append("s32:0")

    # Register all loop induction variables
    for i, loop_op in enumerate(self.loops):
      self.register(loop_op.induction_variable, f"index{i}")
    
    # Register state memrefs
    for i, (m, _, _) in enumerate(self.state):
      self.register(m, acc_names[i])

    # Emit innermost loop body
    innermost_body = self.loops[-1].body
    updated = self._emit_body(innermost_body.operations, acc_names, lines)

    # Index update with carry logic (from innermost to outermost)
    # Each level: if inner wrapped, increment; if reached own ub, wrap to 0
    carry = None
    new_indices = [None] * num_loops
    for i in range(num_loops - 1, -1, -1):
      next_idx = self.new_tmp()
      if carry is None:
        # Innermost loop: always increment, wrap when >= ub
        lines.append(f"    let {next_idx} = if (index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else {{ index{i} + 1 }};")
        carry = self.new_tmp()
        lines.append(f"    let {carry} = index{i} + 1 >= {ubs[i]};")
      else:
        # Outer loop: increment only when inner carried, wrap when >= ub
        lines.append(f"    let {next_idx} = if ({carry} && index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else if ({carry}) {{ index{i} + 1 }} else {{ index{i} }};")
        new_carry = self.new_tmp()
        lines.append(f"    let {new_carry} = {carry} && (index{i} + 1 >= {ubs[i]});")
        carry = new_carry
      new_indices[i] = next_idx
    
    done = carry  # outermost carry = all loops completed

    # Get updated accumulator values
    upd_accs = [updated.get(i, acc_names[i]) for i in range(n)]

    # Send output when done (send the UPDATED value from last iteration)
    # Use return_state_idx to determine which accumulator to output
    if self.outputs and self.state:
      ret_idx = self.return_state_idx
      lines.append(f"    send_if({base_tok}, {list(self.outputs.values())[0]}, {done}, {upd_accs[ret_idx]});")

    # For next state: when done, reset to init; otherwise keep updated value
    final_accs = []
    for i in range(n):
      tmp = self.new_tmp()
      init = self.state[i][2] or "0"
      if init.startswith("__arg") and init.endswith("__"):
        init = f"tmp{int(init[5:-2])}"
      lines.append(f"    let {tmp} = if ({done}) {{ {init} }} else {{ {upd_accs[i]} }};")
      final_accs.append(tmp)

    # Build return state
    state_out = final_accs[:]
    for i in range(num_loops):
      idx_out = self.new_tmp()
      lines.append(f"    let {idx_out} = if ({done}) {{ s32:0 }} else {{ {new_indices[i]} }};")
      state_out.append(idx_out)
      state_out.append(ubs[i])
    busy_out = self.new_tmp()
    lines.append(f"    let {busy_out} = !{done};")
    state_out.append(busy_out)
    
    lines.append(f"    ({', '.join(state_out)})")
    lines.append("  }")
    return "\n".join(lines)

  def _emit_next_while(self, acc_names):
    """Emit next() for WHILE loops."""
    n = len(self.state)
    state_type = "(" + ", ".join([t for _, t, _ in self.state] + ["bool"]) + ")"
    state_vars = acc_names + ["busy"]

    lines = [f"  next(state: {state_type}) {{"]
    lines.append(f"    let ({', '.join(state_vars)}) = state;")

    # Receive inputs
    input_vars = []
    tok_names = []
    for i, ch in enumerate(self.inputs.values()):
      tok, inp = self.new_tok(), self.new_tmp()
      default = acc_names[i] if i < n else "s32:0"
      lines.append(f"    let ({tok}, {inp}) = recv_if(join(), {ch}, !busy, {default});")
      input_vars.append(inp)
      tok_names.append(tok)
      # Register function arguments
      if i < len(list(self.func.arguments)):
        self.register(list(self.func.arguments)[i], inp)

    # Join tokens if multiple receives
    if len(tok_names) > 1:
      base_tok = self.new_tmp()
      lines.append(f"    let {base_tok} = join({', '.join(tok_names)});")
    elif len(tok_names) == 1:
      base_tok = tok_names[0]
    else:
      base_tok = "join()"

    # Build working values: use received input when !busy, otherwise use state
    # For state vars initialized from args, select between input (when !busy) and state (when busy)
    working_accs = []
    for i in range(n):
      init = self.state[i][2] or "0"
      if init.startswith("__arg") and init.endswith("__"):
        arg_idx = int(init[5:-2])
        if arg_idx < len(input_vars):
          # Generate conditional: use input when !busy, state when busy
          work_var = self.new_tmp()
          lines.append(f"    let {work_var} = if (!busy) {{ {input_vars[arg_idx]} }} else {{ {acc_names[i]} }};")
          working_accs.append(work_var)
        else:
          working_accs.append(acc_names[i])
      else:
        working_accs.append(acc_names[i])

    # Register state memrefs with working values
    state_keys = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    for i, (m, _, _) in enumerate(self.state):
      self.register(m, working_accs[i])
    
    # Emit while condition
    before_block = list(self.loop.before.blocks)[0]
    cond = None
    for op in before_block.operations:
      if isinstance(op, scf_d.ConditionOp):
        cond = self.lookup(op.operands[0])
      elif isinstance(op, arith_d.ConstantOp):
        pass  # Already registered in _register_all_constants
      elif isinstance(op, affine_d.AffineLoadOp) and str(op.operands[0]) in state_keys:
        self.register(op.result, self.lookup(self.state[state_keys[str(op.operands[0])]][0]))
      else:
        lines.extend(self.inst_emitter.emit(op))
    cond_var = self.new_tmp()
    lines.append(f"    let {cond_var} = {cond};")

    # Emit body
    body = list(self.loop.after.blocks)[0]
    updated = self._emit_body(body.operations, working_accs, lines)

    # Update accumulators
    new_accs = []
    for i in range(n):
      tmp = self.new_tmp()
      upd = updated.get(i, working_accs[i])
      lines.append(f"    let {tmp} = if ({cond_var}) {{ {upd} }} else {{ {working_accs[i]} }};")
      new_accs.append(tmp)

    # Send output when done (use return_state_idx to determine which accumulator)
    if self.outputs and self.state:
      ret_idx = self.return_state_idx
      lines.append(f"    send_if({base_tok}, {list(self.outputs.values())[0]}, !{cond_var}, {new_accs[ret_idx]});")

    # Reset accumulators when done (use received inputs for arg-initialized state)
    reset = []
    for i in range(n):
      tmp = self.new_tmp()
      init = self.state[i][2] or "0"
      if init.startswith("__arg") and init.endswith("__"):
        arg_idx = int(init[5:-2])
        init = input_vars[arg_idx] if arg_idx < len(input_vars) else "0"
      lines.append(f"    let {tmp} = if (!{cond_var}) {{ {init} }} else {{ {new_accs[i]} }};")
      reset.append(tmp)

    lines.append(f"    ({', '.join(reset + [cond_var])})")
    lines.append("  }")
    return "\n".join(lines)


# ================================================================
# Module lowerer
# ================================================================
class DslxModuleLowerer:
  def __init__(self, module: Module, top_func_name: str):
    self.module = module
    self.func_lowerers = []

    for op in module.body.operations:
      if isinstance(op, func_d.FuncOp):

        # detect if function contains scf.for or scf.while
        has_loop = any(
          inner_op.operation.name in ("scf.for", "scf.while")
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
