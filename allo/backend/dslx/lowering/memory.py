"""Memory binding and channel generation for DSLX proc lowering.

Provides infrastructure for mapping MLIR memrefs to DSLX RAM channels.
"""

from allo._mlir.ir import MemRefType
from allo._mlir.dialects import affine as affine_d, func as func_d

from ....utils import get_bitwidth_from_type, get_dtype_and_shape_from_type
from ..utils.type_utils import allo_dtype_to_dslx_type, is_float_type


# Simple dual-port RAM template for DSLX
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


class MemoryBinding:
    """Metadata for a memory binding (memref argument or return value)."""

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
        """Compute strides for linearization (row-major)."""
        if not self.shape:
            return []
        reversed_strides = []
        running = 1
        for dim in reversed(self.shape):
            reversed_strides.append(running)
            running *= dim
        return list(reversed(reversed_strides))

    def linearize(self, index_exprs):
        """Convert multi-dimensional indices to linear address."""
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


def _validate_memref_shape(shape):
    """Validate memref shape and raise if dynamic dimensions are found."""
    if any(dim == -1 for dim in shape):
        raise NotImplementedError("dynamic memrefs are not supported yet")


def _create_memory_binding(value, dtype, shape, idx, memory_map, arg_index=None):
    """Create a MemoryBinding and add it to the memory map."""
    binding = MemoryBinding(value, dtype, shape, idx, arg_index)
    memory_map[binding.key] = binding
    return binding


def discover_memory_bindings(func, channel_arg_keys=None):
    """Discover all memory bindings (memref args and returns) in a function.
    
    Args:
        func: MLIR FuncOp
        channel_arg_keys: Optional set of argument keys to skip (already channels)
        
    Returns:
        Tuple of (bindings list, memory_map dict)
    """
    channel_arg_keys = channel_arg_keys or set()
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
        if str(arg) in channel_arg_keys:
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


def _mark_memory_usage(func, memory_map):
    """Mark which bindings need read/write channels based on usage."""
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


def _discover_return_memrefs(func, bindings, memory_map, start_idx):
    """Discover memrefs returned by the function."""
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


def build_memory_channels(bindings):
    """Build channel declarations for memory bindings.
    
    Args:
        bindings: List of MemoryBinding objects
        
    Returns:
        List of channel declaration strings
    """
    channels = []
    for binding in bindings:
        if binding.needs_read:
            channels.append(
                f"{binding.read_req_chan}: chan<SimpleReadReq<{binding.addr_param()}>> out"
            )
            channels.append(
                f"{binding.read_resp_chan}: chan<SimpleReadResp<{binding.data_param()}>> in"
            )
        if binding.needs_write:
            channels.append(
                f"{binding.write_req_chan}: chan<SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}>> out"
            )
            channels.append(
                f"{binding.write_resp_chan}: chan<SimpleWriteResp> in"
            )
    return channels


class MemoryEmitter:
    """Emits DSLX code for memory read/write operations."""

    def __init__(self, context):
        """Initialize with a lowering context.
        
        Args:
            context: Object with lookup/register/new_tmp/new_tok/track_token methods
        """
        self.ctx = context

    def _index_exprs(self, operands, start=1):
        """Extract index expressions from operands."""
        return [self.ctx.lookup(op) for op in operands[start:]]

    def emit_read(self, binding, op):
        """Emit DSLX code for a memory read operation.
        
        Args:
            binding: MemoryBinding for the target memory
            op: AffineLoadOp
            
        Returns:
            List of DSLX code lines
        """
        lines = []
        index_exprs = self._index_exprs(op.operands, start=1)
        addr_expr = binding.linearize(index_exprs)
        addr_bits = self.ctx.new_tmp()
        lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
        
        req_tmp = self.ctx.new_tmp()
        lines.append(
            f"    let {req_tmp} = "
            f"SimpleReadReq<{binding.addr_param()}> {{ addr: {addr_bits} }};"
        )
        
        tok = self.ctx.new_tok()
        lines.append(f"    let {tok} = send(join(), {binding.read_req_chan}, {req_tmp});")
        
        resp_tok = self.ctx.new_tok()
        resp_val = self.ctx.new_tmp()
        lines.append(f"    let ({resp_tok}, {resp_val}) = recv({tok}, {binding.read_resp_chan});")
        
        data_tmp = self.ctx.new_tmp()
        # For float memories, convert flattened bits to APFloat via apfloat::unflatten
        if is_float_type(binding.dtype):
            name = binding.dslx_type  # e.g. F32
            lines.append(
                f"    let {data_tmp} = "
                f"apfloat::unflatten<{name}_EXP_SZ, {name}_FRAC_SZ>({resp_val}.data);"
            )
        else:
            lines.append(f"    let {data_tmp} = ({resp_val}.data as {binding.dslx_type});")
        
        self.ctx.register(op.result, data_tmp)
        self.ctx.track_token(resp_tok)
        return lines

    def emit_write(self, binding, op, predicate=None):
        """Emit DSLX code for a memory write operation.
        
        Args:
            binding: MemoryBinding for the target memory
            op: AffineStoreOp
            predicate: Optional condition for conditional write
            
        Returns:
            List of DSLX code lines
        """
        lines = []
        value_expr = self.ctx.lookup(op.operands[0])
        index_exprs = self._index_exprs(op.operands, start=2)
        addr_expr = binding.linearize(index_exprs)
        
        addr_bits = self.ctx.new_tmp()
        lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
        
        # For float values, flatten to bits
        if is_float_type(binding.dtype):
            flat_val = self.ctx.new_tmp()
            lines.append(f"    let {flat_val} = apfloat::flatten({value_expr});")
            data_expr = flat_val
        else:
            cast_val = self.ctx.new_tmp()
            lines.append(f"    let {cast_val} = ({value_expr} as uN[{binding.data_width}]);")
            data_expr = cast_val
        
        req_tmp = self.ctx.new_tmp()
        lines.append(
            f"    let {req_tmp} = "
            f"SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}> "
            f"{{ addr: {addr_bits}, data: {data_expr} }};"
        )
        
        tok = self.ctx.new_tok()
        if predicate:
            lines.append(
                f"    let {tok} = send_if(join(), {binding.write_req_chan}, {predicate}, {req_tmp});"
            )
        else:
            lines.append(f"    let {tok} = send(join(), {binding.write_req_chan}, {req_tmp});")
        
        resp_tok = self.ctx.new_tok()
        if predicate:
            lines.append(
                f"    let ({resp_tok}, _) = recv_if({tok}, {binding.write_resp_chan}, {predicate}, SimpleWriteResp {{}});"
            )
        else:
            lines.append(f"    let ({resp_tok}, _) = recv({tok}, {binding.write_resp_chan});")
        
        self.ctx.track_token(resp_tok)
        return lines
