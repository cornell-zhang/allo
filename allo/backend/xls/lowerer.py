# allo/allo/backend/xls/lowerer.py
# Lowers MLIR to DSLX proc definitions for XLS synthesis.

import re

from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d, arith as arith_d, affine as affine_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.ir import BlockArgument, Module, MemRefType
from allo._mlir.passmanager import PassManager
from ...utils import get_func_inputs_outputs, get_dtype_and_shape_from_type
from .utils import allo_dtype_to_dslx_type, emit_float_defs
from .instruction import InstructionEmitter
from .memory import RAM_TEMPLATE, MemoryEmitter, discover_memory_bindings, build_memory_channels


class DslxFuncLowererBase:
  # Base class for lowering a single MLIR func to a DSLX proc.
  def __init__(self, func: func_d.FuncOp):
    self.func = func
    self.value_map = {}
    self.const_map = {}
    self.inputs = {}
    self.outputs = {}
    self.channels = []
    self.input_types = []
    self.channel_handles = []
    self.scalar_arg_order = []
    self.tmp_counter = 0
    self.tok_counter = 0
    self.inst_emitter = InstructionEmitter(self)
    self.pending_tokens = []
    self._register_all_constants()

  # Pre-register all constants so they're available during lowering.
  def _register_all_constants(self):
    def visit(block):
      for op in block.operations:
        if isinstance(op, arith_d.ConstantOp):
          self.inst_emitter.emit(op)
        for region in op.regions:
          for blk in region.blocks:
            visit(blk)
    for block in self.func.body.blocks:
      visit(block)

  def new_tmp(self):
    name = f"tmp{self.tmp_counter}"
    self.tmp_counter += 1
    return name

  def new_tok(self):
    name = f"tok{self.tok_counter}"
    self.tok_counter += 1
    return name

  def register(self, mlir_name, dslx_name):
    self.value_map[mlir_name] = dslx_name

  def constant(self, mlir_name, dslx_name):
    self.const_map[mlir_name] = dslx_name

  def lookup(self, mlir_name):
    return self.const_map.get(mlir_name) or self.value_map[mlir_name]

  def track_token(self, token):
    if token and token != "join()":
      self.pending_tokens.append(token)

  # Consume pending tokens into a joined token expression.
  def _consume_tokens(self, base_tok, lines):
    tokens = ([base_tok] if base_tok and base_tok != "join()" else []) + self.pending_tokens
    self.pending_tokens = []
    if not tokens:
      return "join()"
    if len(tokens) == 1:
      return tokens[0]
    joined = self.new_tok()
    lines.append(f"    let {joined} = join({', '.join(tokens)});")
    return joined

  def _func_args_for_io(self):
    return [arg for arg in self.func.arguments if "!allo.stream" not in str(arg.type)]

  def _emit_channels(self):
    self.channel_handles = []
    self.channels = self._build_channel_decls()
    return "\n".join([f"  {s};" for s in self.channels])

  def _build_channel_decls(self):
    raise NotImplementedError

  def _emit_config(self):
    return f"  config({', '.join(self.channels)}) {{ ({', '.join(self.channel_handles)}) }}"

  # Receive inputs with conditional guard.
  def _recv_inputs(self, busy_guard, defaults=None, join_tokens=False):
    defaults = defaults or []
    lines, values, tokens = [], [], []
    for idx, ch in enumerate(self.inputs.values()):
      tok, val = self.new_tok(), self.new_tmp()
      default = defaults[idx] if idx < len(defaults) else "s32:0"
      lines.append(f"    let ({tok}, {val}) = recv_if(join(), {ch}, {busy_guard}, {default});")
      values.append(val)
      tokens.append(tok)
      if idx < len(self.scalar_arg_order):
        self.register(self.scalar_arg_order[idx], val)
    if not tokens:
      base_tok = "join()"
    elif join_tokens:
      if len(tokens) > 1:
        join_tok = self.new_tmp()
        lines.append(f"    let {join_tok} = join({', '.join(tokens)});")
        base_tok = join_tok
      else:
        base_tok = tokens[0]
    else:
      base_tok = tokens[-1]
    return lines, values, base_tok

  def _emit_inputs(self):
    raise NotImplementedError

  def _emit_body(self):
    raise NotImplementedError

  def _emit_init(self):
    raise NotImplementedError

  def _emit_next(self):
    raise NotImplementedError

  def emit_proc(self):
    return "\n".join([
      f"pub proc {self.func.name.value} {{",
      self._emit_channels(), "",
      self._emit_config(), "",
      self._emit_init(), "",
      self._emit_next(),
      "}"
    ])


class DslxCombLowerer(DslxFuncLowererBase):
  # Lowerer for combinational (non-looping) functions.
  def _build_channel_decls(self):
    inputs, outputs = get_func_inputs_outputs(self.func)
    func_args = self._func_args_for_io()
    assert len(inputs) == len(func_args)

    self.input_types, self.scalar_arg_order = [], []
    self.inputs, self.outputs = {}, {}
    channels, handles = [], []

    for idx, (arg, (dtype, _)) in enumerate(zip(func_args, inputs)):
      dslx_type = allo_dtype_to_dslx_type(dtype)
      self.input_types.append(dslx_type)
      self.inputs[idx] = f"in{idx}"
      self.scalar_arg_order.append(arg)
      channels.append(f"in{idx}: chan<{dslx_type}> in")
      handles.append(f"in{idx}")

    for idx, (dtype, _) in enumerate(outputs):
      dslx_type = allo_dtype_to_dslx_type(dtype)
      self.outputs[idx] = f"out{idx}"
      channels.append(f"out{idx}: chan<{dslx_type}> out")
      handles.append(f"out{idx}")

    self.channel_handles = handles
    return channels

  def _emit_inputs(self):
    lines = []
    for idx, mlir_arg in enumerate(self.scalar_arg_order):
      var, tok = self.new_tmp(), self.new_tok()
      lines.append(f"    let ({tok}, {var}) = recv(join(), {self.inputs[idx]});")
      self.register(mlir_arg, var)
    return lines

  def _emit_body(self):
    lines = []
    ops = list(self.func.body.blocks[0].operations)
    for idx, op in enumerate(ops):
      if isinstance(op, affine_d.AffineForOp) and _is_unrolled_for(op):
        emitter = UnrolledForEmitter(self, op, lines)
        emitter.emit(ops[idx + 1:])
      else:
        lines.extend(self.inst_emitter.emit(op))
    return lines

  def _emit_init(self):
    return "  init { () }"

  def _emit_next(self):
    return "\n".join(["  next(state: ()) {"] + self._emit_inputs() + self._emit_body() + ["  }"])


def _is_unrolled_for(loop_op):
  """Check if an affine.for has unroll=0 attribute and only accesses scalar memrefs."""
  if not isinstance(loop_op, affine_d.AffineForOp):
    return False
  # Check unroll attribute - if specified, must be 0
  unroll_attr = None
  for attr in loop_op.attributes:
    if attr.name == "unroll":
      unroll_attr = attr
      break
  if unroll_attr is not None:
    # Assert that unroll factor is 0 if specified
    attr_str = str(unroll_attr.attr)
    match = re.search(r"(\d+)\s*:", attr_str)
    assert match is not None, f"Could not parse unroll factor from: {attr_str}"
    unroll_factor = int(match.group(1))
    assert unroll_factor == 0, f"If unrolling, loop must be fully unrolled, got: {unroll_factor}"
  else:
    return False
  # Check only scalar memref access (no arrays)
  body = loop_op.body if hasattr(loop_op, "body") else list(loop_op.region.blocks)[0]
  for op in body.operations:
    if isinstance(op, affine_d.AffineLoadOp):
      mt = MemRefType(op.operands[0].type)
      if mt.shape:
        return False
    elif isinstance(op, affine_d.AffineStoreOp):
      mt = MemRefType(op.operands[1].type)
      if mt.shape:
        return False
  return True


def _has_non_unrolled_loops(func):
  """Check if function has any non-unrolled for loops or while loops."""
  def check_block(block):
    for op in block.operations:
      if isinstance(op, scf_d.WhileOp):
        return True
      if isinstance(op, scf_d.ForOp):
        return True
      if isinstance(op, affine_d.AffineForOp):
        if not _is_unrolled_for(op):
          return True
        # Check nested loops in body
        body = op.body if hasattr(op, "body") else list(op.region.blocks)[0]
        if check_block(body):
          return True
      for region in op.regions:
        for blk in region.blocks:
          if check_block(blk):
            return True
    return False
  return check_block(func.body.blocks[0])


def _get_loop_body(loop_op):
  """Get the body block of a loop operation."""
  if hasattr(loop_op, "body"):
    return loop_op.body
  if hasattr(loop_op, "region"):
    return list(loop_op.region.blocks)[0]
  raise NotImplementedError(f"unsupported loop: {type(loop_op)}")


def _extract_for_upper_bound(loop_op):
  """Extract upper bound from affine.for loop."""
  try:
    header = loop_op.operation.get_asm().splitlines()[0]
  except AttributeError:
    header = str(loop_op.operation).splitlines()[0]
  match = re.search(r"\bto\b\s+([0-9]+)", header)
  return match.group(1) if match else None


class UnrolledForEmitter:
  """Helper class to emit DSLX for expressions for unrolled affine.for loops."""

  def __init__(self, lowerer, loop_op, lines, indent="    "):
    self.lowerer = lowerer
    self.loop_op = loop_op
    self.lines = lines
    self.indent = indent
    self.body = _get_loop_body(loop_op)
    self.upper_bound = _extract_for_upper_bound(loop_op)
    self.carried = []  # [(memref_key, dslx_type, init_expr)]
    self.used_after = set()  # memref keys used after loop
    self.local_temps = {}

  def _find_carried_and_usage(self, ops_after_loop):
    """Find loop-carried values and which are used after the loop."""
    # Find memrefs loaded/stored in loop body (loop-carried)
    loaded, stored = set(), set()
    def scan_body(block):
      for op in block.operations:
        if isinstance(op, affine_d.AffineLoadOp):
          mt = MemRefType(op.operands[0].type)
          if not mt.shape:
            loaded.add(str(op.operands[0]))
        elif isinstance(op, affine_d.AffineStoreOp):
          mt = MemRefType(op.operands[1].type)
          if not mt.shape:
            stored.add(str(op.operands[1]))
        elif isinstance(op, affine_d.AffineForOp) and _is_unrolled_for(op):
          scan_body(_get_loop_body(op))
        for region in op.regions:
          for blk in region.blocks:
            scan_body(blk)
    scan_body(self.body)
    carried_keys = loaded & stored

    # Build carried list with types
    for op in self.body.operations:
      if isinstance(op, affine_d.AffineLoadOp):
        mkey = str(op.operands[0])
        if mkey in carried_keys and mkey not in [c[0] for c in self.carried]:
          mt = MemRefType(op.operands[0].type)
          dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
          init_expr = self.lowerer.lookup(op.operands[0])
          self.carried.append((mkey, allo_dtype_to_dslx_type(dtype), init_expr, op.operands[0]))

    # Find which carried values are used after the loop
    def scan_usage(ops):
      for op in ops:
        if isinstance(op, affine_d.AffineLoadOp):
          mkey = str(op.operands[0])
          if mkey in carried_keys:
            self.used_after.add(mkey)
        for region in op.regions:
          for blk in region.blocks:
            scan_usage(blk.operations)
    scan_usage(ops_after_loop)

  def emit(self, ops_after_loop=None):
    """Emit the DSLX for expression and return updated memref mappings."""
    ops_after_loop = ops_after_loop or []
    self._find_carried_and_usage(ops_after_loop)

    if not self.carried:
      # No loop-carried state, just emit body inline (will be unrolled)
      for op in self.body.operations:
        if isinstance(op, affine_d.AffineYieldOp):
          continue
        self.lines.extend(self.lowerer.inst_emitter.emit(op))
      return

    ub = self.upper_bound
    if not ub:
      raise NotImplementedError("Dynamic bounds not supported for unrolled for")

    # Register loop induction variable
    idx_var = self.lowerer.new_tmp()
    if hasattr(self.loop_op, "induction_variable"):
      self.lowerer.register(self.loop_op.induction_variable, idx_var)
    elif self.body.arguments:
      self.lowerer.register(self.body.arguments[0], idx_var)

    # Build accumulator type and initial value
    if len(self.carried) == 1:
      acc_type = self.carried[0][1]
      acc_var = self.lowerer.new_tmp()
      init_val = self.carried[0][2]
    else:
      acc_type = f"({', '.join(c[1] for c in self.carried)})"
      acc_var = self.lowerer.new_tmp()
      init_val = f"({', '.join(c[2] for c in self.carried)})"

    # Build result destructuring with _ for unused values
    result_names = []
    for mkey, _, _, memref in self.carried:
      if mkey in self.used_after:
        result_names.append(self.lowerer.new_tmp())
      else:
        result_names.append("_")

    if len(self.carried) == 1:
      result_pattern = result_names[0]
    else:
      result_pattern = f"({', '.join(result_names)})"

    # Emit for loop header: let (results) = for (i, acc) in type:0..type:N { ... }(init);
    self.lines.append(f"{self.indent}let {result_pattern} = for ({idx_var}, {acc_var}): (s32, {acc_type}) in s32:0..s32:{ub} {{")

    # Register accumulators for use in body
    if len(self.carried) == 1:
      self.lowerer.register(self.carried[0][3], acc_var)
    else:
      for i, (mkey, _, _, memref) in enumerate(self.carried):
        self.lowerer.register(memref, f"{acc_var}.{i}")

    # Emit body
    body_updates = self._emit_body_ops()

    # Emit return value (new accumulator state)
    if len(self.carried) == 1:
      new_val = body_updates.get(self.carried[0][0], acc_var)
      self.lines.append(f"{self.indent}  {new_val}")
    else:
      new_vals = [body_updates.get(c[0], f"{acc_var}.{i}") for i, c in enumerate(self.carried)]
      self.lines.append(f"{self.indent}  ({', '.join(new_vals)})")

    self.lines.append(f"{self.indent}}}({init_val});")

    # Update lowerer's value map with results for values used after loop
    for i, (mkey, _, _, memref) in enumerate(self.carried):
      if mkey in self.used_after:
        self.lowerer.register(memref, result_names[i])

  def _emit_body_ops(self):
    """Emit body operations and return map of memref updates."""
    body_updates = {}
    carried_keys = {c[0] for c in self.carried}

    for op in self.body.operations:
      if isinstance(op, affine_d.AffineYieldOp):
        continue
      elif isinstance(op, arith_d.ConstantOp):
        self.lowerer.inst_emitter.emit(op)
      elif isinstance(op, memref_d.AllocOp):
        self.local_temps[str(op.result)] = None
      elif isinstance(op, affine_d.AffineLoadOp):
        mkey = str(op.operands[0])
        if mkey in carried_keys:
          # Use current accumulator value
          for i, c in enumerate(self.carried):
            if c[0] == mkey:
              self.lowerer.register(op.result, self.lowerer.lookup(c[3]))
              break
        elif mkey in self.local_temps:
          self.lowerer.register(op.result, self.local_temps[mkey])
        else:
          for line in self.lowerer.inst_emitter.emit(op):
            self.lines.append(self.indent + "  " + line.strip())
      elif isinstance(op, affine_d.AffineStoreOp):
        val = self.lowerer.lookup(op.operands[0])
        mkey = str(op.operands[1])
        if mkey in carried_keys:
          body_updates[mkey] = val
          for c in self.carried:
            if c[0] == mkey:
              self.lowerer.register(c[3], val)
              break
        elif mkey in self.local_temps:
          self.local_temps[mkey] = val
      elif isinstance(op, affine_d.AffineForOp) and _is_unrolled_for(op):
        # Recursively handle nested unrolled for
        nested = UnrolledForEmitter(self.lowerer, op, self.lines, self.indent + "  ")
        nested.emit([])
      else:
        for line in self.lowerer.inst_emitter.emit(op):
          self.lines.append(self.indent + "  " + line.strip())

    return body_updates


class DslxStatefulLowerer(DslxFuncLowererBase):
  # Lowerer for stateful (looping) functions.
  uses_memory = False

  def __init__(self, func):
    super().__init__(func)
    self.memory_bindings, self.memory_map = discover_memory_bindings(self.func)
    self.mem_emitter = MemoryEmitter(self)
    self.uses_memory = any(b.needs_read or b.needs_write for b in self.memory_bindings)
    self.loop = None
    self.loop_type = None
    self.state = []
    self.loops = []
    self.loop_preambles = []
    self.loop_postambles = []
    self.loop_upper_bounds = []
    self.return_state_idx = 0
    self._analyze()

  # Analyze function structure to find loops and state variables.
  def _analyze(self):
    block = self.func.body.blocks[0]
    for op in block.operations:
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        self.loop, self.loop_type = op, 'for'
        self._collect_nested_loops(op)
        break
      elif isinstance(op, scf_d.WhileOp):
        self.loop, self.loop_type = op, 'while'
        self._collect_nested_loops_from_block(list(op.after.blocks)[0], op)
        break

    memrefs = {op.result for op in block.operations if isinstance(op, memref_d.AllocOp) and op.result != self.loop}
    if self.loop:
      if self.loop_type == 'for':
        memrefs |= self._collect_loop_allocs(self.loop)
        used = self._collect_memrefs(self._loop_body(self.loop), memrefs)
      else:
        after_block = list(self.loop.after.blocks)[0]
        memrefs |= self._collect_allocs_from_block(after_block)
        used = self._collect_memrefs(after_block, memrefs) | self._collect_memrefs(list(self.loop.before.blocks)[0], memrefs)
    else:
      used = set()

    for m in used:
      if "memref" in str(getattr(m, "type", "")) and str(m) not in self.memory_map:
        mt = MemRefType(m.type)
        dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
        self.state.append((m, allo_dtype_to_dslx_type(dtype), self._get_init(m)))

    self._find_return_state_idx(block)

  # Find which state variable is returned by the function.
  def _find_return_state_idx(self, block):
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    for op in block.operations:
      if isinstance(op, func_d.ReturnOp) and op.operands:
        owner = op.operands[0].owner.opview if hasattr(op.operands[0].owner, 'opview') else op.operands[0].owner
        if isinstance(owner, affine_d.AffineLoadOp):
          key = str(owner.operands[0])
          if key in state_memrefs:
            self.return_state_idx = state_memrefs[key]
        break

  # Recursively collect nested loop structure.
  def _collect_nested_loops(self, loop_op):
    self._collect_nested_loops_from_block(self._loop_body(loop_op), loop_op)

  def _collect_nested_loops_from_block(self, body, loop_op):
    preamble, postamble, nested, found = [], [], None, False
    for op in body.operations:
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp, scf_d.WhileOp)):
        if not found:
          nested, found = op, True
      elif found:
        postamble.append(op)
      else:
        preamble.append(op)

    self.loops.append(loop_op)
    self.loop_preambles.append(preamble if nested else [])
    self.loop_postambles.append(postamble if nested else [])
    self.loop_upper_bounds.append(None if isinstance(loop_op, scf_d.WhileOp) else self._extract_loop_upper_bound(loop_op))

    if nested:
      if isinstance(nested, scf_d.WhileOp):
        self._collect_nested_loops_from_block(list(nested.after.blocks)[0], nested)
      else:
        self._collect_nested_loops(nested)

  def _extract_loop_upper_bound(self, loop_op):
    if isinstance(loop_op, affine_d.AffineForOp):
      try:
        header = loop_op.operation.get_asm().splitlines()[0]
      except AttributeError:
        header = str(loop_op.operation).splitlines()[0]
      match = re.search(r"\bto\b\s+([0-9]+)", header)
      if match:
        return f"s32:{match.group(1)}"
    return None

  def _collect_loop_allocs(self, loop_op):
    return self._collect_allocs_from_block(self._loop_body(loop_op))

  def _collect_allocs_from_block(self, block):
    if not block:
      return set()
    allocs = set()
    for op in block.operations:
      if isinstance(op, memref_d.AllocOp):
        allocs.add(op.result)
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        allocs |= self._collect_allocs_from_block(self._loop_body(op))
      elif isinstance(op, (scf_d.IfOp, affine_d.AffineIfOp)):
        for attr in [('then_block', 'thenRegion'), ('else_block', 'elseRegion')]:
          allocs |= self._collect_allocs_from_block(self._get_region_block(op, *attr))
    return allocs

  def _collect_memrefs(self, block, memrefs):
    used = set()
    for op in block.operations:
      used.update(o for o in op.operands if o in memrefs)
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        used |= self._collect_memrefs(self._loop_body(op), memrefs)
      elif isinstance(op, (scf_d.IfOp, affine_d.AffineIfOp)):
        for attr in [('then_block', 'thenRegion'), ('else_block', 'elseRegion')]:
          blk = self._get_region_block(op, *attr)
          if blk:
            used |= self._collect_memrefs(blk, memrefs)
    return used

  # Get initial value for a state memref from stores before the loop.
  def _get_init(self, memref):
    func_args = list(self.func.arguments)
    for op in self.func.body.blocks[0].operations:
      if isinstance(op, (scf_d.ForOp, scf_d.WhileOp, affine_d.AffineForOp)):
        break
      if isinstance(op, affine_d.AffineStoreOp) and op.operands[1] == memref:
        v = op.operands[0]
        if isinstance(v, BlockArgument):
          return f"__arg{v.arg_number}__"
        if v in func_args:
          return f"__arg{func_args.index(v)}__"
        owner = v.owner.opview if hasattr(v.owner, 'opview') else v.owner
        if isinstance(owner, arith_d.ConstantOp):
          return str(owner.value).split(":")[0].strip()
    return "0"

  def _get_region_block(self, op, block_attr, region_attr):
    block = getattr(op, block_attr, None)
    if block:
      return block
    region = getattr(op, region_attr, None)
    return list(region.blocks)[0] if region and region.blocks else None

  def _loop_body(self, loop_op):
    if hasattr(loop_op, "body"):
      return loop_op.body
    if hasattr(loop_op, "region"):
      return list(loop_op.region.blocks)[0]
    raise NotImplementedError(f"unsupported loop: {type(loop_op)}")

  def _loop_induction_var(self, loop_op):
    if hasattr(loop_op, "induction_variable"):
      return loop_op.induction_variable
    body = getattr(loop_op, "body", None)
    if body and body.arguments:
      return body.arguments[0]
    raise NotImplementedError(f"no induction var for {type(loop_op)}")

  def _is_while_loop(self, i):
    return i < len(self.loops) and isinstance(self.loops[i], scf_d.WhileOp)

  def _loop_bound_is_dynamic(self, idx):
    return idx >= len(self.loop_upper_bounds) or self.loop_upper_bounds[idx] is None

  def _get_loop_body(self, loop_op):
    if isinstance(loop_op, scf_d.WhileOp):
      return list(loop_op.after.blocks)[0]
    return self._loop_body(loop_op)

  # Build loop guard condition for a given nesting level.
  def _inner_guard_condition(self, level, ubs):
    conds = []
    if level < len(ubs):
      conds.append(self.lookup(f"index{level}") if self._is_while_loop(level) else f"index{level} < {ubs[level]}")
    for i in range(level + 1, len(self.loops)):
      conds.append(self.lookup(f"index{i}") if self._is_while_loop(i) else f"index{i} == s32:0")
    return " && ".join(conds) if conds else "true"

  # Build guard for the entire loop nest.
  def _loop_iteration_guard(self, ubs):
    conds = []
    for i in range(min(len(self.loops), len(ubs))):
      conds.append(self.lookup(f"index{i}") if self._is_while_loop(i) else f"index{i} < {ubs[i]}")
    return " && ".join(conds) if conds else "true"

  # Evaluate while loop condition from before block.
  def _eval_while_condition(self, loop_op, state_keys, lines, level):
    before_block = list(loop_op.before.blocks)[0]
    cond = None
    for op in before_block.operations:
      if isinstance(op, scf_d.ConditionOp):
        cond = self.lookup(op.operands[0])
      elif isinstance(op, arith_d.ConstantOp):
        pass
      elif isinstance(op, affine_d.AffineLoadOp) and str(op.operands[0]) in state_keys:
        self.register(op.result, self.lookup(self.state[state_keys[str(op.operands[0])]][0]))
      else:
        lines.extend(self.inst_emitter.emit(op))
    cond_var = self.new_tmp()
    lines.append(f"    let {cond_var} = {cond};")
    self.register(f"index{level}", cond_var)
    return cond_var

  # Emit loop body operations and track state updates.
  def _emit_body(self, ops, acc_names, lines, temp_memrefs=None, predicate=None):
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    temp_memrefs = temp_memrefs or {}
    updated = {}
    ops_list = list(ops)

    for idx, op in enumerate(ops_list):
      if isinstance(op, (arith_d.ConstantOp, scf_d.ForOp, scf_d.YieldOp, affine_d.AffineYieldOp)):
        continue
      # Handle unrolled affine.for loops
      if isinstance(op, affine_d.AffineForOp):
        if _is_unrolled_for(op):
          emitter = UnrolledForEmitter(self, op, lines)
          emitter.emit(ops_list[idx + 1:])
        continue
      if isinstance(op, memref_d.AllocOp):
        temp_memrefs[str(op.result)] = None
      elif isinstance(op, affine_d.AffineLoadOp):
        if not self._handle_affine_load(op, lines, temp_memrefs, state_memrefs):
          lines.extend(self.inst_emitter.emit(op))
      elif isinstance(op, affine_d.AffineStoreOp):
        if not self._handle_affine_store(op, lines, temp_memrefs, state_memrefs, updated, predicate):
          lines.extend(self.inst_emitter.emit(op))
      elif isinstance(op, scf_d.IfOp):
        self._handle_if_op(op, acc_names, lines, temp_memrefs, updated)
      else:
        lines.extend(self.inst_emitter.emit(op))
    return updated

  def _handle_if_op(self, op, acc_names, lines, temp_memrefs, updated):
    cond = self.lookup(op.operands[0])
    saved = dict(self.value_map)
    then_lines, else_lines = [], []
    then_upd = self._emit_body(self._get_region_block(op, 'then_block', 'thenRegion').operations, acc_names, then_lines, temp_memrefs)
    self.value_map = dict(saved)
    else_blk = self._get_region_block(op, 'else_block', 'elseRegion')
    else_upd = self._emit_body(else_blk.operations, acc_names, else_lines, temp_memrefs) if else_blk else {}
    lines.extend(then_lines + else_lines)
    for i in set(then_upd) | set(else_upd):
      tmp = self.new_tmp()
      lines.append(f"    let {tmp} = if ({cond}) {{ {then_upd.get(i, acc_names[i])} }} else {{ {else_upd.get(i, acc_names[i])} }};")
      updated[i] = tmp
      self.register(self.state[i][0], tmp)

  def _handle_affine_load(self, op, lines, temp_memrefs, state_memrefs):
    mkey = str(op.operands[0])
    binding = self.memory_map.get(mkey)
    if binding and binding.needs_read:
      lines.extend(self.mem_emitter.emit_read(binding, op))
      return True
    if mkey in state_memrefs:
      self.register(op.result, self.lookup(self.state[state_memrefs[mkey]][0]))
      return True
    if mkey in temp_memrefs:
      self.register(op.result, temp_memrefs[mkey])
      return True
    return False

  def _handle_affine_store(self, op, lines, temp_memrefs, state_memrefs, updated, predicate=None):
    val, mkey = self.lookup(op.operands[0]), str(op.operands[1])
    binding = self.memory_map.get(mkey)
    if binding and binding.needs_write:
      lines.extend(self.mem_emitter.emit_write(binding, op, predicate))
      return True
    if mkey in state_memrefs:
      i = state_memrefs[mkey]
      tmp = self.new_tmp()
      updated[i] = tmp
      lines.append(f"    let {tmp} = {val};")
      self.register(self.state[i][0], tmp)
      return True
    if mkey in temp_memrefs:
      temp_memrefs[mkey] = val
      return True
    return False

  # Get initial expression for state variable, handling arg references.
  def _state_init_expr(self, idx, arg_sources=None):
    init = self.state[idx][2] or "0"
    if init.startswith("__arg") and init.endswith("__"):
      arg_idx = int(init[5:-2])
      if arg_sources and arg_idx < len(arg_sources):
        return arg_sources[arg_idx]
      return f"tmp{arg_idx}"
    return init

  def _emit_send_on_done(self, lines, base_tok, predicate, values):
    if not (self.outputs and self.state):
      return
    out_chan = list(self.outputs.values())[0]
    tok = self._consume_tokens(base_tok, lines)
    send_tok = self.new_tok()
    lines.append(f"    let {send_tok} = send_if({tok}, {out_chan}, {predicate}, {values[self.return_state_idx]});")
    self.track_token(send_tok)

  def _emit_state_reset(self, lines, predicate, values, arg_sources=None):
    reset = []
    for i in range(len(self.state)):
      tmp = self.new_tmp()
      lines.append(f"    let {tmp} = if ({predicate}) {{ {self._state_init_expr(i, arg_sources)} }} else {{ {values[i]} }};")
      reset.append(tmp)
    return reset

  def _emit_done_signal(self, lines, done):
    if self.control_channels:
      tok = self._consume_tokens("join()", lines)
      send = self.new_tok()
      lines.append(f"    let {send} = send_if({tok}, done, {done}, bool:1);")
      self.track_token(send)

  def _build_channel_decls(self):
    inputs, outputs = get_func_inputs_outputs(self.func)
    func_args = self._func_args_for_io()
    assert len(inputs) == len(func_args)

    self.input_types, self.scalar_arg_order = [], []
    self.inputs, self.outputs = {}, {}
    channels, handles = [], []

    for idx, (arg, (dtype, shape)) in enumerate(zip(func_args, inputs)):
      if shape:
        continue
      dslx_type = allo_dtype_to_dslx_type(dtype)
      self.input_types.append(dslx_type)
      self.inputs[idx] = f"in{idx}"
      self.scalar_arg_order.append(arg)
      channels.append(f"in{idx}: chan<{dslx_type}> in")
      handles.append(f"in{idx}")

    for idx, (dtype, shape) in enumerate(outputs):
      if shape:
        continue
      dslx_type = allo_dtype_to_dslx_type(dtype)
      self.outputs[idx] = f"out{idx}"
      channels.append(f"out{idx}: chan<{dslx_type}> out")
      handles.append(f"out{idx}")

    mem_channels, mem_handles, needs_control = build_memory_channels(self.memory_bindings)
    channels.extend(mem_channels)
    handles.extend(mem_handles)
    self.control_channels = ("go", "done") if needs_control else None
    self.channel_handles = handles
    return channels

  def _emit_init(self):
    fields = []
    for _, _, init in self.state:
      fields.append("0" if init and init.startswith("__arg") else (init or "0"))
    for i in range(len(self.loops) if self.loops else 1):
      if self._is_while_loop(i):
        fields.append("true")
      elif self._loop_bound_is_dynamic(i):
        fields.extend(["0", "0"])
      else:
        fields.append("0")
    fields.append("false")
    return f"  init {{ ({', '.join(fields)}) }}"

  def _build_loop_state_type(self, num_loops):
    bt = self.input_types[0] if self.input_types else "s32"
    loop_types, loop_vars = [], []
    for i in range(num_loops):
      loop_types.append("bool" if self._is_while_loop(i) else bt)
      loop_vars.append(f"index{i}")
      if self._loop_bound_is_dynamic(i) and not self._is_while_loop(i):
        loop_types.append(bt)
        loop_vars.append(f"ub{i}")
    state_type = "(" + ", ".join([t for _, t, _ in self.state] + loop_types + ["bool"]) + ")"
    return state_type, loop_vars

  def _setup_control_channels(self, lines):
    if self.control_channels:
      go_tok, go_val, start = self.new_tok(), self.new_tmp(), self.new_tmp()
      lines.append(f"    let ({go_tok}, {go_val}) = recv_if(join(), go, !busy, bool:0);")
      self.track_token(go_tok)
      lines.append(f"    let {start} = !busy && ({go_val} == bool:1);")
      return start, start
    return None, "!busy"

  def _compute_upper_bounds(self, num_loops, scalar_inputs):
    ubs, input_idx = [], 0
    for i in range(num_loops):
      if self._is_while_loop(i):
        ubs.append("bool:1")
      elif self._loop_bound_is_dynamic(i):
        ubs.append(scalar_inputs[input_idx] if input_idx < len(scalar_inputs) else f"ub{i}")
        input_idx += 1
      else:
        ubs.append(self.loop_upper_bounds[i] if i < len(self.loop_upper_bounds) and self.loop_upper_bounds[i] else "s32:0")
    return ubs

  def _emit_loop_preambles(self, acc_names, lines, temp_memrefs, ubs, state_keys):
    for level in range(len(self.loops)):
      if self._is_while_loop(level):
        self._eval_while_condition(self.loops[level], state_keys, lines, level)
      elif level < len(self.loop_preambles) and self.loop_preambles[level]:
        pre_updates = self._emit_body(self.loop_preambles[level], acc_names, lines, temp_memrefs)
        cond = self._inner_guard_condition(level, ubs)
        for idx, val in pre_updates.items():
          if cond != "true":
            guarded = self.new_tmp()
            lines.append(f"    let {guarded} = if ({cond}) {{ {val} }} else {{ {acc_names[idx]} }};")
            acc_names[idx] = guarded
            self.register(self.state[idx][0], guarded)
          else:
            acc_names[idx] = val

  def _update_accumulators_with_guard(self, acc_names, updated, loop_guard, lines, n):
    for i in range(n):
      if i in updated:
        new_val = updated[i]
        if loop_guard != "true":
          guarded = self.new_tmp()
          lines.append(f"    let {guarded} = if ({loop_guard}) {{ {new_val} }} else {{ {acc_names[i]} }};")
          acc_names[i] = guarded
        else:
          acc_names[i] = new_val
        self.register(self.state[i][0], acc_names[i])

  # Build carry logic for nested loops (innermost to outermost).
  def _emit_loop_carry_logic(self, num_loops, ubs, acc_names, lines, temp_memrefs):
    carry, new_indices = None, [None] * num_loops
    for i in range(num_loops - 1, -1, -1):
      is_while = self._is_while_loop(i)
      if carry is None:
        if is_while:
          cond = self.lookup(f"index{i}")
          new_indices[i] = cond
          carry = self.new_tmp()
          lines.append(f"    let {carry} = !{cond};")
        else:
          next_idx = self.new_tmp()
          lines.append(f"    let {next_idx} = if (index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else {{ index{i} + 1 }};")
          carry = self.new_tmp()
          lines.append(f"    let {carry} = index{i} + 1 >= {ubs[i]};")
          new_indices[i] = next_idx
        if i > 0 and (i - 1) < len(self.loop_postambles) and self.loop_postambles[i - 1]:
          outer_valid = self.lookup(f"index{i-1}") if self._is_while_loop(i-1) else f"index{i-1} < {ubs[i-1]}"
          post_updates = self._emit_body(self.loop_postambles[i - 1], acc_names, lines, temp_memrefs, predicate=f"{carry} && ({outer_valid})")
          for idx, val in post_updates.items():
            guarded = self.new_tmp()
            lines.append(f"    let {guarded} = if ({carry} && ({outer_valid})) {{ {val} }} else {{ {acc_names[idx]} }};")
            acc_names[idx] = guarded
            self.register(self.state[idx][0], guarded)
      else:
        if is_while:
          new_indices[i] = self.lookup(f"index{i}")
        else:
          next_idx = self.new_tmp()
          lines.append(f"    let {next_idx} = if ({carry} && index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else if ({carry}) {{ index{i} + 1 }} else {{ index{i} }};")
          new_carry = self.new_tmp()
          lines.append(f"    let {new_carry} = {carry} && (index{i} + 1 >= {ubs[i]});")
          carry = new_carry
          new_indices[i] = next_idx
    return new_indices, carry

  def _build_state_output(self, num_loops, final_accs, new_indices, done, ubs, start_flag, lines):
    state_out = final_accs[:]
    for i in range(num_loops):
      idx_out = self.new_tmp()
      reset_val = "false" if self._is_while_loop(i) else "s32:0"
      lines.append(f"    let {idx_out} = if ({done}) {{ {reset_val} }} else {{ {new_indices[i]} }};")
      state_out.append(idx_out)
      if self._loop_bound_is_dynamic(i) and not self._is_while_loop(i):
        state_out.append(ubs[i])
    busy_out = self.new_tmp()
    busy_expr = f"({start_flag}) || (busy && !{done})" if self.control_channels else f"!{done}"
    lines.append(f"    let {busy_out} = {busy_expr};")
    state_out.append(busy_out)
    return state_out

  def _emit_next(self):
    n = len(self.state)
    acc_names = [f"acc{i}" for i in range(n)]
    num_loops = len(self.loops) if self.loops else 1

    state_type, loop_vars = self._build_loop_state_type(num_loops)
    state_vars = acc_names + loop_vars + ["busy"]
    self.pending_tokens = []
    lines = [f"  next(state: {state_type}) {{", f"    let ({', '.join(state_vars)}) = state;"]

    start_flag, busy_guard = self._setup_control_channels(lines)
    is_outer_while = self.loop_type == 'while'
    recv_defaults = [acc_names[i] for i in range(n)] if is_outer_while else [f"ub{i}" for i in range(num_loops) if self._loop_bound_is_dynamic(i)]
    recv_lines, scalar_inputs, base_tok = self._recv_inputs(busy_guard, recv_defaults, join_tokens=is_outer_while)
    lines.extend(recv_lines)

    for i, loop_op in enumerate(self.loops):
      if not self._is_while_loop(i):
        self.register(self._loop_induction_var(loop_op), f"index{i}")
      elif i == 0 and is_outer_while:
        for j in range(n):
          init = self.state[j][2] or "0"
          if init.startswith("__arg") and init.endswith("__"):
            arg_idx = int(init[5:-2])
            if arg_idx < len(scalar_inputs):
              work = self.new_tmp()
              lines.append(f"    let {work} = if (!busy) {{ {scalar_inputs[arg_idx]} }} else {{ {acc_names[j]} }};")
              acc_names[j] = work

    ubs = self._compute_upper_bounds(num_loops, scalar_inputs)
    for i, (m, _, _) in enumerate(self.state):
      self.register(m, acc_names[i])

    state_keys = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    temp_memrefs = {}
    self._emit_loop_preambles(acc_names, lines, temp_memrefs, ubs, state_keys)

    loop_guard = self._loop_iteration_guard(ubs)
    updated = self._emit_body(self._get_loop_body(self.loops[-1]).operations, acc_names, lines, temp_memrefs)
    self._update_accumulators_with_guard(acc_names, updated, loop_guard, lines, n)
    new_indices, done = self._emit_loop_carry_logic(num_loops, ubs, acc_names, lines, temp_memrefs)

    self._emit_send_on_done(lines, base_tok, done, acc_names)
    self._emit_done_signal(lines, done)
    final_accs = self._emit_state_reset(lines, done, acc_names, scalar_inputs if is_outer_while else None)
    state_out = self._build_state_output(num_loops, final_accs, new_indices, done, ubs, start_flag, lines)

    lines.extend([f"    ({', '.join(state_out)})", "  }"])
    return "\n".join(lines)


class DslxModuleLowerer:
  # Lowerer for an entire MLIR module containing multiple functions.
  def __init__(self, module: Module, top_func_name: str):
    self.module = module
    self.func_lowerers = []
    for op in module.body.operations:
      if isinstance(op, func_d.FuncOp):
        # Use stateful lowerer only if there are non-unrolled loops
        if _has_non_unrolled_loops(op):
          self.func_lowerers.append(DslxStatefulLowerer(op))
        else:
          self.func_lowerers.append(DslxCombLowerer(op))

  def emit_module(self):
    body = "\n\n".join(fl.emit_proc() for fl in self.func_lowerers)
    # collect float types used from all instruction emitters
    float_types = set()
    for fl in self.func_lowerers:
      float_types.update(fl.inst_emitter.float_types_used)
    # build preamble with float defs and ram template if needed
    preamble = []
    if float_types:
      preamble.append(emit_float_defs(float_types))
    if any(getattr(fl, "uses_memory", False) for fl in self.func_lowerers):
      preamble.append(RAM_TEMPLATE)
    if preamble:
      return "\n\n".join(preamble) + "\n\n" + body
    return body


# Canonicalize MLIR before lowering.
def clean_mlir(module):
  with module.context:
    PassManager.parse("builtin.module(canonicalize, sccp, cse, symbol-dce)").run(module.operation)


def lower_mlir(module: Module, top_func_name: str, **kwargs) -> str:
  clean_mlir(module)
  return DslxModuleLowerer(module, top_func_name).emit_module()
