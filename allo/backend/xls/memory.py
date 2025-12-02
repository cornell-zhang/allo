# allo/allo/backend/xls/memory.py
# Memory binding and channel generation for DSLX proc lowering.

from allo._mlir.ir import MemRefType
from allo._mlir.dialects import affine as affine_d, func as func_d

from ...utils import get_bitwidth_from_type, get_dtype_and_shape_from_type
from .utils import allo_dtype_to_dslx_type, is_float_type

RAM_TEMPLATE = """\
// Simple dual-port RAM model with independent read and write ports.
// Parameterized on address width, data width, and depth.
// Reads observe the state before a concurrent write (read-before-write).
// Writes always update a full word.

pub struct SimpleReadReq<ADDR_WIDTH: u32> {
  addr: uN[ADDR_WIDTH],
}

pub struct SimpleReadResp<DATA_WIDTH: u32> {
  data: uN[DATA_WIDTH],
}

pub struct SimpleWriteReq<ADDR_WIDTH: u32, DATA_WIDTH: u32> {
  addr: uN[ADDR_WIDTH],
  data: uN[DATA_WIDTH],
}

pub struct SimpleWriteResp {}

pub proc Simple1R1WRam<ADDR_WIDTH: u32, DATA_WIDTH: u32, SIZE: u32> {
  read_req: chan<SimpleReadReq<ADDR_WIDTH>> in;
  read_resp: chan<SimpleReadResp<DATA_WIDTH>> out;
  write_req: chan<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>> in;
  write_resp: chan<SimpleWriteResp> out;

  config(read_req: chan<SimpleReadReq<ADDR_WIDTH>> in,
         read_resp: chan<SimpleReadResp<DATA_WIDTH>> out,
         write_req: chan<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>> in,
         write_resp: chan<SimpleWriteResp> out) {
    (read_req, read_resp, write_req, write_resp)
  }

  init { uN[DATA_WIDTH][SIZE]:[uN[DATA_WIDTH]:0, ...] }

  next(state: uN[DATA_WIDTH][SIZE]) {
    let (tok_r, r_req, r_valid) =
        recv_non_blocking(join(), read_req, zero!<SimpleReadReq<ADDR_WIDTH>>());
    let (tok_w, w_req, w_valid) =
        recv_non_blocking(join(), write_req, zero!<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>>());

    let addr_r = r_req.addr as u32;
    let addr_w = w_req.addr as u32;

    let state_before_write = state;
    let state =
        if w_valid { update(state, addr_w, w_req.data) } else { state };

    let read_data = state_before_write[addr_r];
    send_if(
        tok_r, read_resp, r_valid,
        SimpleReadResp<DATA_WIDTH> { data: read_data });

    send_if(
        tok_w, write_resp, w_valid,
        SimpleWriteResp {});

    state
  }
}
"""


# Metadata for a memory binding (memref argument or return value).
class MemoryBinding:
  def __init__(self, value, dtype, shape, index, arg_index=None):
    self.value = value
    self.key = str(value)
    self.dtype = dtype
    self.dslx_type = allo_dtype_to_dslx_type(dtype)
    self.shape = tuple(shape)
    self.data_width = get_bitwidth_from_type(dtype)
    self.size = 1
    for dim in self.shape:
      self.size *= dim
    self.addr_width = max(1, (self.size - 1).bit_length()) if self.size else 1
    self.base = f"mem{index}"
    self.read_req_chan = f"{self.base}__read_req"
    self.read_resp_chan = f"{self.base}__read_resp"
    self.write_req_chan = f"{self.base}__write_req"
    self.write_resp_chan = f"{self.base}__write_resp"
    self.arg_index = arg_index
    self.needs_read = False
    self.needs_write = False
    self.strides = self._compute_strides()

  def addr_param(self):
    return f"u32:{self.addr_width}"

  def data_param(self):
    return f"u32:{self.data_width}"

  def _compute_strides(self):
    if not self.shape:
      return []
    reversed_strides = []
    running = 1
    for dim in reversed(self.shape):
      reversed_strides.append(running)
      running *= dim
    return list(reversed(reversed_strides))

  # Convert multi-dimensional indices to linear address.
  def linearize(self, index_exprs):
    if not index_exprs:
      return "s32:0"
    if len(index_exprs) != len(self.shape):
      raise NotImplementedError("partial indexing of memories is not supported")
    terms = []
    for expr, stride in zip(index_exprs, self.strides):
      if stride == 1:
        terms.append(expr)
      else:
        terms.append(f"({expr} * s32:{stride})")
    if not terms:
      return "s32:0"
    if len(terms) == 1:
      return terms[0]
    return "(" + " + ".join(terms) + ")"


# Validate memref shape and raise if dynamic dimensions are found.
def _validate_memref_shape(shape):
  if any(dim == -1 for dim in shape):
    raise NotImplementedError("dynamic memrefs are not supported yet")


# Create a MemoryBinding and add it to the memory map.
def _create_memory_binding(value, dtype, shape, idx, memory_map, arg_index=None):
  binding = MemoryBinding(value, dtype, shape, idx, arg_index)
  memory_map[binding.key] = binding
  return binding


# Discover all memory bindings (memref args and returns) in a function.
def discover_memory_bindings(func):
  func_args = [
      arg for arg in func.arguments
      if "!allo.stream" not in str(arg.type)
  ]
  bindings = []
  memory_map = {}
  idx = 0
  for arg_index, arg in enumerate(func_args):
    if not MemRefType.isinstance(arg.type):
      continue
    mt = MemRefType(arg.type)
    shape = tuple(mt.shape)
    if not shape:
      continue
    _validate_memref_shape(shape)
    elem_dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
    binding = _create_memory_binding(arg, elem_dtype, shape, idx, memory_map, arg_index)
    bindings.append(binding)
    idx += 1
  idx = _discover_return_memrefs(func, bindings, memory_map, idx)
  if bindings:
    _mark_memory_usage(func, memory_map)
  return bindings, memory_map


# Mark which bindings need read/write channels based on usage.
def _mark_memory_usage(func, memory_map):
  def visit_block(block):
    for op in block.operations:
      if isinstance(op, affine_d.AffineLoadOp):
        binding = memory_map.get(str(op.operands[0]))
        if binding:
          binding.needs_read = True
      elif isinstance(op, affine_d.AffineStoreOp):
        binding = memory_map.get(str(op.operands[1]))
        if binding:
          binding.needs_write = True
      for region in op.regions:
        for blk in region.blocks:
          visit_block(blk)

  for block in func.body.blocks:
    visit_block(block)


# Discover memrefs returned by the function.
def _discover_return_memrefs(func, bindings, memory_map, start_idx):
  idx = start_idx
  for block in func.body.blocks:
    for op in block.operations:
      if isinstance(op, func_d.ReturnOp):
        for operand in op.operands:
          if not MemRefType.isinstance(operand.type):
            continue
          key = str(operand)
          if key in memory_map:
            continue
          mt = MemRefType(operand.type)
          shape = tuple(mt.shape)
          if not shape:
            continue
          _validate_memref_shape(shape)
          elem_dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
          binding = _create_memory_binding(operand, elem_dtype, shape, idx, memory_map)
          binding.needs_write = True
          bindings.append(binding)
          idx += 1
        return idx
  return idx


# Emits DSLX code for memory read/write operations.
class MemoryEmitter:
  def __init__(self, parent):
    self.p = parent

  # Emit DSLX code for a memory read operation.
  def emit_read(self, binding, op):
    lines = []
    index_exprs = self._index_exprs(op.operands, start=1)
    addr_expr = binding.linearize(index_exprs)
    addr_bits = self.p.new_tmp()
    lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
    req_tmp = self.p.new_tmp()
    lines.append(f"    let {req_tmp} = "
                 f"SimpleReadReq<{binding.addr_param()}> {{ addr: {addr_bits} }};")
    tok = self.p.new_tok()
    lines.append(f"    let {tok} = send(join(), {binding.read_req_chan}, {req_tmp});")
    resp_tok = self.p.new_tok()
    resp_val = self.p.new_tmp()
    lines.append(f"    let ({resp_tok}, {resp_val}) = recv({tok}, {binding.read_resp_chan});")
    data_tmp = self.p.new_tmp()
    # For float memories, convert flattened bits to APFloat via apfloat::unflatten.
    if is_float_type(binding.dtype):
      name = binding.dslx_type  # e.g. F32
      lines.append(
          f"    let {data_tmp} = "
          f"apfloat::unflatten<{name}_EXP_SZ, {name}_FRAC_SZ>({resp_val}.data);")
    else:
      lines.append(f"    let {data_tmp} = ({resp_val}.data as {binding.dslx_type});")
    self.p.register(op.result, data_tmp)
    self.p.track_token(resp_tok)
    return lines

  # Emit DSLX code for a memory write operation.
  def emit_write(self, binding, op, predicate=None):
    lines = []
    value_expr = self.p.lookup(op.operands[0])
    index_exprs = self._index_exprs(op.operands, start=2)
    addr_expr = binding.linearize(index_exprs)
    addr_bits = self.p.new_tmp()
    lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
    data_bits = self.p.new_tmp()
    # For float memories, flatten APFloat to bits before writing.
    if is_float_type(binding.dtype):
      name = binding.dslx_type  # e.g. F32
      flat = self.p.new_tmp()
      lines.append(
          f"    let {flat} = "
          f"apfloat::flatten<{name}_EXP_SZ, {name}_FRAC_SZ>({value_expr});")
      lines.append(
          f"    let {data_bits} = ({flat} as uN[{binding.data_width}]);")
    else:
      lines.append(f"    let {data_bits} = ({value_expr} as uN[{binding.data_width}]);")
    req_tmp = self.p.new_tmp()
    lines.append(
        "    let "
        f"{req_tmp} = SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}> "
        f"{{ addr: {addr_bits}, data: {data_bits} }};")
    token_expr = self.p._consume_tokens("join()", lines)
    tok = self.p.new_tok()
    if predicate:
      # Use send_if/recv_if when predicate is provided
      lines.append(f"    let {tok} = send_if({token_expr}, {binding.write_req_chan}, {predicate}, {req_tmp});")
      ack_tok = self.p.new_tok()
      lines.append(f"    let ({ack_tok}, _) = recv_if({tok}, {binding.write_resp_chan}, {predicate}, zero!<SimpleWriteResp>());")
    else:
      # Unconditional write
      lines.append(f"    let {tok} = send({token_expr}, {binding.write_req_chan}, {req_tmp});")
      ack_tok = self.p.new_tok()
      lines.append(f"    let ({ack_tok}, _) = recv({tok}, {binding.write_resp_chan});")
    self.p.track_token(ack_tok)
    return lines

  def _index_exprs(self, operands, start):
    if len(operands) <= start:
      return []
    exprs = []
    for idx in operands[start:]:
      exprs.append(f"({self.p.lookup(idx)} as s32)")
    return exprs


# Build channel declarations and handles for memory bindings.
def build_memory_channels(memory_bindings):
  channels = []
  handles = []
  for binding in memory_bindings:
    if binding.needs_read:
      channels.extend([
        f"{binding.read_req_chan}: chan<SimpleReadReq<{binding.addr_param()}>> out",
        f"{binding.read_resp_chan}: chan<SimpleReadResp<{binding.data_param()}>> in"
      ])
      handles.extend([binding.read_req_chan, binding.read_resp_chan])
    if binding.needs_write:
      channels.extend([
        f"{binding.write_req_chan}: chan<SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}>> out",
        f"{binding.write_resp_chan}: chan<SimpleWriteResp> in"
      ])
      handles.extend([binding.write_req_chan, binding.write_resp_chan])
  
  needs_control = len(memory_bindings) > 0
  if needs_control:
    channels.extend(["go: chan<bool> in", "done: chan<bool> out"])
    handles.extend(["go", "done"])
  
  return channels, handles, needs_control


# parse memory channels from ir and build io constraints for ram response delays.
def parse_memory_io_constraints(ir_content, ram_latency=1):
  import re
  # extract all memory channels from ir
  chan_pattern = re.compile(r'chan\s+(\w+)\s*\([^)]+\)')
  channels = {}
  for line in ir_content.split('\n'):
    match = chan_pattern.search(line)
    if match:
      chan_name = match.group(1)
      # match memory channel pattern: {func}__mem{idx}__{read_req|read_resp|write_req|write_resp}
      mem_match = re.match(r'(\w+)__(read_req|read_resp|write_req|write_resp)', chan_name)
      if mem_match:
        mem_name, chan_type = mem_match.group(1), mem_match.group(2)
        channels.setdefault(mem_name, {})[chan_type] = chan_name
  
  # build io constraints: req_channel:send:resp_channel:recv:latency:latency
  io_constraints = []
  for chans in channels.values():
    # add constraint for read channels if both req and resp exist
    if 'read_req' in chans and 'read_resp' in chans:
      io_constraints.append(f"{chans['read_req']}:send:{chans['read_resp']}:recv:{ram_latency}:{ram_latency}")
    # add constraint for write channels if both req and resp exist
    if 'write_req' in chans and 'write_resp' in chans:
      io_constraints.append(f"{chans['write_req']}:send:{chans['write_resp']}:recv:{ram_latency}:{ram_latency}")
  
  return io_constraints, len(channels) > 0


# detect memory channels and build io constraints from ir file.
def detect_memory_and_constraints(opt_path, ram_latency=1):
  # read ir file and check for memory channels
  has_memory = False
  io_constraints = []
  try:
    with open(opt_path, "r") as f:
      ir_content = f.read()
      # check for memory channel patterns (mem*__read_req, mem*__write_req)
      if "__read_req" in ir_content or "__write_req" in ir_content:
        # parse memory channels to build io constraints for ram timing
        io_constraints, has_memory = parse_memory_io_constraints(ir_content, ram_latency)
  except Exception:
    pass
  return io_constraints, has_memory