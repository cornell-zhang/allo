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

# ================================================================
# base class
# ================================================================
class DslxFuncLowererBase:
  def __init__(self, func: func_d.FuncOp):
    self.func = func

    # mappings
    self.value_map = {}  # mlir ssa names map to dslx identifiers
    self.const_map = {}  # mlir ssa names map to dslx constant literals

    # input and output channels
    self.inputs = {}  # mlir arg index maps to dslx channel name
    self.outputs = {}  # mlir return index maps to dslx channel name
    self.channels = []  # channel declaration strings
    self.input_types = []  # dslx types for inputs

    # naming variables
    self.tmp_counter = 0
    self.tok_counter = 0

    # instruction emitter
    self.inst_emitter = InstructionEmitter(self)
    
    # register all constants upfront
    self._register_all_constants()

  # find and register every constant reachable inside the function
  def _register_all_constants(self):
    def visit_block(block):
      for op in block.operations:
        if isinstance(op, arith_d.ConstantOp):
          self.inst_emitter.emit(op)
        # recurse into nested regions
        for region in op.regions:
          for blk in region.blocks:
            visit_block(blk)
    
    for block in self.func.body.blocks:
      visit_block(block)

  # ================================================================
  # naming
  # ================================================================
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

  # ================================================================
  # channels and config
  # ================================================================
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

  # ================================================================
  # input helpers
  # ================================================================
  def _recv_inputs(self, busy_guard, defaults=None, join_tokens=False):
    defaults = defaults or []
    lines = []
    values = []
    tokens = []
    func_args = list(self.func.arguments)
    for idx, ch in enumerate(self.inputs.values()):
      tok = self.new_tok()
      val = self.new_tmp()
      default = defaults[idx] if idx < len(defaults) else "s32:0"
      lines.append(f"    let ({tok}, {val}) = recv_if(join(), {ch}, {busy_guard}, {default});")
      values.append(val)
      tokens.append(tok)
      if idx < len(func_args):
        self.register(func_args[idx], val)
    if not tokens:
      base_tok = "join()"
    elif join_tokens and len(tokens) > 1:
      join_tok = self.new_tmp()
      lines.append(f"    let {join_tok} = join({', '.join(tokens)});")
      base_tok = join_tok
    elif join_tokens:
      base_tok = tokens[0]
    else:
      base_tok = tokens[-1]
    return lines, values, base_tok

  # ================================================================
  # subclass hooks
  # ================================================================
  def _emit_inputs(self):
    raise NotImplementedError

  def _emit_body(self):
    raise NotImplementedError

  def _emit_init(self):
    raise NotImplementedError

  def _emit_next(self):
    raise NotImplementedError

  # ================================================================
  # proc assembly
  # ================================================================
  def emit_proc(self):
    func = f"pub proc {self.func.name.value} {{\n"
    func += self._emit_channels() + "\n\n"
    func += self._emit_config() + "\n\n"
    func += self._emit_init() + "\n\n"
    func += self._emit_next() + "\n"
    func += "}"
    return func


# ================================================================
# combinational lowerer
# ================================================================
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


# ================================================================
# stateful lowerer
# ================================================================
class DslxStatefulLowerer(DslxFuncLowererBase):

  def __init__(self, func):
    super(DslxStatefulLowerer, self).__init__(func)
    self.loop = None
    self.loop_type = None  # 'for' or 'while'
    self.state = []        # list of tuples (memref value, dslx type, init value)
    self.loops = []        # list of nested for ops from outermost to innermost
    self.loop_preambles = []
    self.return_state_idx = 0  # index of the state variable returned by the func
    self._analyze()

  # ================================================================
  # loop analysis
  # ================================================================
  def _analyze(self):
    block = self.func.body.blocks[0]
    for op in block.operations:
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        self.loop, self.loop_type = op, 'for'
        break
      elif isinstance(op, scf_d.WhileOp):
        self.loop, self.loop_type = op, 'while'
        break
    # collect nested for loops from outermost to innermost
    if self.loop_type == 'for':
      self._collect_nested_loops(self.loop)

    # collect memrefs defined before the loop
    memrefs = {op.result for op in block.operations
               if isinstance(op, memref_d.AllocOp) and op.result != self.loop}
    if self.loop is not None and self.loop_type == 'for':
      memrefs |= self._collect_loop_allocs(self.loop)
    
    # find memrefs used inside the loop body
    # note: scf.for body is already a block, while scf.while before/after are regions
    if self.loop_type == 'while':
      body = list(self.loop.after.blocks)[0]
      used = self._collect_memrefs(body, memrefs)
      used |= self._collect_memrefs(list(self.loop.before.blocks)[0], memrefs)
    else:
      body = self._loop_body(self.loop)
      used = self._collect_memrefs(body, memrefs)

    for m in used:
      if "memref" in str(getattr(m, "type", "")):
        mt = MemRefType(m.type)
        dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
        self.state.append((m, allo_dtype_to_dslx_type(dtype), self._get_init(m)))

    # find which state variable feeds the return
    self._find_return_state_idx(block)

  # locate the state entry used as the function return
  def _find_return_state_idx(self, block):
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    for op in block.operations:
      if isinstance(op, func_d.ReturnOp):
        if len(op.operands) > 0:
          ret_val = op.operands[0]
          # the return value is typically loaded from a memref
          owner = ret_val.owner.opview if hasattr(ret_val.owner, 'opview') else ret_val.owner
          if isinstance(owner, affine_d.AffineLoadOp):
            memref_key = str(owner.operands[0])
            if memref_key in state_memrefs:
              self.return_state_idx = state_memrefs[memref_key]
              return
        break

  # recursively collect nested for ops so we can track loop order
  def _collect_nested_loops(self, loop_op):
    body = self._loop_body(loop_op)
    preamble = []
    nested = None
    for op in body.operations:
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        nested = op
        break
      preamble.append(op)
    self.loops.append(loop_op)
    self.loop_preambles.append(preamble if nested is not None else [])
    if nested is not None:
      self._collect_nested_loops(nested)

  def _collect_loop_allocs(self, loop_op):
    return self._collect_allocs_from_block(self._loop_body(loop_op))

  def _collect_memref_allocs_in_block(self, block):
    return self._collect_allocs_from_block(block)

  def _collect_allocs_from_block(self, block):
    if block is None:
      return set()
    allocs = set()
    for op in block.operations:
      allocs |= self._collect_allocs_from_op(op)
    return allocs

  def _collect_allocs_from_op(self, op):
    allocs = set()
    if isinstance(op, memref_d.AllocOp):
      allocs.add(op.result)
    if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
      allocs |= self._collect_allocs_from_block(self._loop_body(op))
    elif isinstance(op, (scf_d.IfOp, affine_d.AffineIfOp)):
      then_blk = self._get_region_block(op, 'then_block', 'thenRegion')
      else_blk = self._get_region_block(op, 'else_block', 'elseRegion')
      allocs |= self._collect_allocs_from_block(then_blk)
      allocs |= self._collect_allocs_from_block(else_blk)
    return allocs

  def _collect_memrefs(self, block, memrefs):
    used = set()
    for op in block.operations:
      for operand in op.operands:
        if operand in memrefs:
          used.add(operand)
      if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
        used |= self._collect_memrefs(self._loop_body(op), memrefs)
      elif isinstance(op, scf_d.IfOp):
        then_blk = self._get_region_block(op, 'then_block', 'thenRegion')
        used |= self._collect_memrefs(then_blk, memrefs)
        else_blk = self._get_region_block(op, 'else_block', 'elseRegion')
        if else_blk:
          used |= self._collect_memrefs(else_blk, memrefs)
      elif isinstance(op, affine_d.AffineIfOp):
        then_blk = self._get_region_block(op, 'then_block', 'thenRegion')
        used |= self._collect_memrefs(then_blk, memrefs)
        else_blk = self._get_region_block(op, 'else_block', 'elseRegion')
        if else_blk:
          used |= self._collect_memrefs(else_blk, memrefs)
    return used

  def _get_init(self, memref):
    func_args = list(self.func.arguments)
    for op in self.func.body.blocks[0].operations:
      if isinstance(op, (scf_d.ForOp, scf_d.WhileOp, affine_d.AffineForOp)):
        break
      if isinstance(op, affine_d.AffineStoreOp) and op.operands[1] == memref:
        v = op.operands[0]
        # check if the store value is a function argument
        if isinstance(v, BlockArgument):
          return f"__arg{v.arg_number}__"
        # also check by comparing with function arguments directly
        for i, arg in enumerate(func_args):
          if v == arg:
            return f"__arg{i}__"
        # check if it is a constant
        owner = v.owner.opview if hasattr(v.owner, 'opview') else v.owner
        if isinstance(owner, arith_d.ConstantOp):
          return str(owner.value).split(":")[0].strip()
    return "0"

  def _get_region_block(self, op, block_attr, region_attr):
    block = getattr(op, block_attr, None)
    if block is not None:
      return block
    region = getattr(op, region_attr, None)
    return list(region.blocks)[0] if region and len(region.blocks) > 0 else None

  def _loop_body(self, loop_op):
    if hasattr(loop_op, "body"):
      return loop_op.body
    if hasattr(loop_op, "region"):
      return list(loop_op.region.blocks)[0]
    raise NotImplementedError(f"unsupported loop op: {type(loop_op)}")

  def _loop_induction_var(self, loop_op):
    if hasattr(loop_op, "induction_variable"):
      return loop_op.induction_variable
    body = getattr(loop_op, "body", None)
    if body and len(body.arguments) > 0:
      return body.arguments[0]
    raise NotImplementedError(f"cannot determine induction variable for {type(loop_op)}")

  def _inner_guard_condition(self, level, ubs):
    conds = []
    if level < len(ubs):
      conds.append(f"index{level} < {ubs[level]}")
    conds.extend(f"index{i} == s32:0" for i in range(level + 1, len(self.loops)))
    if not conds:
      return "true"
    return " && ".join(conds)

  def _loop_iteration_guard(self, ubs):
    conds = [f"index{i} < {ubs[i]}" for i in range(min(len(self.loops), len(ubs)))]
    if not conds:
      return "true"
    return " && ".join(conds)

  # ================================================================
  # body lowering helpers
  # ================================================================
  def _emit_body(self, ops, acc_names, lines, temp_memrefs=None):
    state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    if temp_memrefs is None:
      temp_memrefs = {}  # track temporary memrefs inside the loop body (string keys to values)
    updated = {}
    
    for op in ops:
      if isinstance(op, (arith_d.ConstantOp, scf_d.ForOp, scf_d.YieldOp,
                         affine_d.AffineForOp, affine_d.AffineYieldOp)):
        continue
      if isinstance(op, memref_d.AllocOp):
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
        then_blk = self._get_region_block(op, 'then_block', 'thenRegion')
        then_upd = self._emit_body(then_blk.operations, acc_names, then_lines, temp_memrefs)
        self.value_map = dict(saved)
        else_blk = self._get_region_block(op, 'else_block', 'elseRegion')
        else_upd = self._emit_body(else_blk.operations, acc_names, else_lines, temp_memrefs) if else_blk else {}
        lines.extend(then_lines + else_lines)
        for i in set(then_upd) | set(else_upd):
          tmp = self.new_tmp()
          lines.append(f"    let {tmp} = if ({cond}) {{ {then_upd.get(i, acc_names[i])} }} else {{ {else_upd.get(i, acc_names[i])} }};")
          updated[i] = tmp
          self.register(self.state[i][0], tmp)
      else:
        lines.extend(self.inst_emitter.emit(op))
    return updated

  # ================================================================
  # shared state helpers
  # ================================================================
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
    ret_idx = self.return_state_idx
    out_chan = list(self.outputs.values())[0]
    lines.append(f"    send_if({base_tok}, {out_chan}, {predicate}, {values[ret_idx]});")

  def _emit_state_reset(self, lines, predicate, values, arg_sources=None):
    reset = []
    for i in range(len(self.state)):
      tmp = self.new_tmp()
      init_expr = self._state_init_expr(i, arg_sources)
      lines.append(f"    let {tmp} = if ({predicate}) {{ {init_expr} }} else {{ {values[i]} }};")
      reset.append(tmp)
    return reset

  # ================================================================
  # init and next assembly
  # ================================================================
  def _emit_init(self):
    fields = [
      "0" if init and init.startswith("__arg") and init.endswith("__") else (init or "0")
      for _, _, init in self.state
    ]
    if self.loop_type == 'for':
      num_loops = len(self.loops) if self.loops else 1
      # for each loop level store index_i and upper_bound_i
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
    
    # handle while loops separately because they use a different state shape
    if self.loop_type == 'while':
      return self._emit_next_while(acc_names)
    
    # for loops (1d, 2d, nd share the same logic here)
    bt = self.input_types[0] if self.input_types else "s32"
    
    # state packs accumulators, every loop index and bound, and a busy bit
    loop_types = [bt, bt] * num_loops
    state_type = "(" + ", ".join([t for _, t, _ in self.state] + loop_types + ["bool"]) + ")"
    idx_pairs = [val for i in range(num_loops) for val in (f"index{i}", f"ub{i}")]
    state_vars = acc_names + idx_pairs + ["busy"]

    lines = [f"  next(state: {state_type}) {{"]
    lines.append(f"    let ({', '.join(state_vars)}) = state;")

    # receive inputs (upper bounds for each loop level)
    recv_defaults = [f"ub{i}" for i in range(num_loops)]
    recv_lines, ubs, base_tok = self._recv_inputs("!busy", recv_defaults)
    lines.extend(recv_lines)
    
    # pad ubs if fewer inputs than loops (should be rare)
    ubs += ["s32:0"] * max(0, num_loops - len(ubs))

    # register each loop induction variable
    for i, loop_op in enumerate(self.loops):
      self.register(self._loop_induction_var(loop_op), f"index{i}")
    
    # register state memrefs with their accumulator aliases
    for i, (m, _, _) in enumerate(self.state):
      self.register(m, acc_names[i])

    temp_memrefs = {}

    # emit outer-loop preambles guarded on inner-loop resets
    for level in range(max(0, len(self.loops) - 1)):
      ops = self.loop_preambles[level] if level < len(self.loop_preambles) else []
      if not ops:
        continue
      pre_updates = self._emit_body(ops, acc_names, lines, temp_memrefs)
      cond = self._inner_guard_condition(level, ubs)
      for idx, val in pre_updates.items():
        prev = acc_names[idx]
        if cond != "true":
          guarded = self.new_tmp()
          lines.append(f"    let {guarded} = if ({cond}) {{ {val} }} else {{ {prev} }};")
          acc_names[idx] = guarded
          self.register(self.state[idx][0], guarded)
        else:
          acc_names[idx] = val

    # emit the innermost loop body
    innermost_body = self._loop_body(self.loops[-1])
    updated = self._emit_body(innermost_body.operations, acc_names, lines, temp_memrefs)

    loop_guard = self._loop_iteration_guard(ubs)
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

    # build carry logic from the innermost loop outward
    # each level wraps to zero when reaching its bound
    carry = None
    new_indices = [None] * num_loops
    for i in range(num_loops - 1, -1, -1):
      next_idx = self.new_tmp()
      if carry is None:
        # innermost loop always increments then wraps
        lines.append(f"    let {next_idx} = if (index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else {{ index{i} + 1 }};")
        carry = self.new_tmp()
        lines.append(f"    let {carry} = index{i} + 1 >= {ubs[i]};")
      else:
        # outer loops increment only when the inner loop carried
        lines.append(f"    let {next_idx} = if ({carry} && index{i} + 1 >= {ubs[i]}) {{ s32:0 }} else if ({carry}) {{ index{i} + 1 }} else {{ index{i} }};")
        new_carry = self.new_tmp()
        lines.append(f"    let {new_carry} = {carry} && (index{i} + 1 >= {ubs[i]});")
        carry = new_carry
      new_indices[i] = next_idx
    
    done = carry  # outermost carry indicates completion

    # pick updated accumulator values or fall back to the previous ones
    upd_accs = acc_names[:]

    # send results and reset accumulators when finished
    self._emit_send_on_done(lines, base_tok, done, upd_accs)
    final_accs = self._emit_state_reset(lines, done, upd_accs)

    # build the composite state tuple returned by next()
    state_out = final_accs[:]
    for i in range(num_loops):
      idx_out = self.new_tmp()
      lines.append(f"    let {idx_out} = if ({done}) {{ s32:0 }} else {{ {new_indices[i]} }};")
      state_out += [idx_out, ubs[i]]
    busy_out = self.new_tmp()
    lines.append(f"    let {busy_out} = !{done};")
    state_out.append(busy_out)
    
    lines.append(f"    ({', '.join(state_out)})")
    lines.append("  }")
    return "\n".join(lines)

  # emit next() for while loops
  # ================================================================
  # while loop support
  # ================================================================
  def _emit_next_while(self, acc_names):
    n = len(self.state)
    state_type = "(" + ", ".join([t for _, t, _ in self.state] + ["bool"]) + ")"
    state_vars = acc_names + ["busy"]

    lines = [f"  next(state: {state_type}) {{"]
    lines.append(f"    let ({', '.join(state_vars)}) = state;")

    # receive inputs
    defaults = [acc_names[i] for i in range(n)]
    recv_lines, input_vars, base_tok = self._recv_inputs("!busy", defaults, join_tokens=True)
    lines.extend(recv_lines)

    # build working values: use received input when not busy, otherwise keep state
    # for arg initialized state entries choose the input when not busy
    working_accs = []
    for i in range(n):
      init = self.state[i][2] or "0"
      if init.startswith("__arg") and init.endswith("__"):
        arg_idx = int(init[5:-2])
        if arg_idx < len(input_vars):
          # generate conditional so busy cycles keep their previous value
          work_var = self.new_tmp()
          lines.append(f"    let {work_var} = if (!busy) {{ {input_vars[arg_idx]} }} else {{ {acc_names[i]} }};")
          working_accs.append(work_var)
        else:
          working_accs.append(acc_names[i])
      else:
        working_accs.append(acc_names[i])

    # register state memrefs with the working values
    state_keys = {str(m): i for i, (m, _, _) in enumerate(self.state)}
    for i, (m, _, _) in enumerate(self.state):
      self.register(m, working_accs[i])
    
    # emit the while condition by visiting the before block
    before_block = list(self.loop.before.blocks)[0]
    cond = None
    for op in before_block.operations:
      if isinstance(op, scf_d.ConditionOp):
        cond = self.lookup(op.operands[0])
      elif isinstance(op, arith_d.ConstantOp):
        pass  # already registered in _register_all_constants
      elif isinstance(op, affine_d.AffineLoadOp) and str(op.operands[0]) in state_keys:
        self.register(op.result, self.lookup(self.state[state_keys[str(op.operands[0])]][0]))
      else:
        lines.extend(self.inst_emitter.emit(op))
    cond_var = self.new_tmp()
    lines.append(f"    let {cond_var} = {cond};")

    # emit the while body
    body = list(self.loop.after.blocks)[0]
    updated = self._emit_body(body.operations, working_accs, lines)

    # update accumulators using the new values when the loop continues
    new_accs = []
    for i in range(n):
      tmp = self.new_tmp()
      upd = updated.get(i, working_accs[i])
      lines.append(f"    let {tmp} = if ({cond_var}) {{ {upd} }} else {{ {working_accs[i]} }};")
      new_accs.append(tmp)

    # send output when the loop halts and reset accumulators using latest inputs
    done_cond = f"!{cond_var}"
    self._emit_send_on_done(lines, base_tok, done_cond, new_accs)
    reset = self._emit_state_reset(lines, done_cond, new_accs, input_vars)

    lines.append(f"    ({', '.join(reset + [cond_var])})")
    lines.append("  }")
    return "\n".join(lines)


# ================================================================
# module lowerer
# ================================================================
class DslxModuleLowerer:
  def __init__(self, module: Module, top_func_name: str):
    self.module = module
    self.func_lowerers = []

    for op in module.body.operations:
      if isinstance(op, func_d.FuncOp):

        # detect whether the function contains loops we can lower statefully
        loop_ops = ("scf.for", "scf.while", "affine.for")
        has_loop = any(
          inner_op.operation.name in loop_ops
          for block in op.body.blocks
          for inner_op in block.operations
        )

        # ensure we only instantiate the stateful lowerer when loops exist
        lowerer_cls = DslxStatefulLowerer if has_loop else DslxCombLowerer
        self.func_lowerers.append(lowerer_cls(op))

  def emit_module(self):
    return "\n\n".join(fl.emit_proc() for fl in self.func_lowerers)


# ================================================================
# mlir canonicalization and lowering entry
# ================================================================
def clean_mlir(module):
  with module.context:
    pm = PassManager.parse("builtin.module(canonicalize, sccp, cse, symbol-dce)")
    pm.run(module.operation)

def lower_mlir(module: Module, top_func_name: str, **kwargs) -> str:
  clean_mlir(module)
  return DslxModuleLowerer(module, top_func_name).emit_module()
