"""MLIR to DSLX proc lowering implementation.

Lowers MLIR functions to DSLX proc definitions with channels and state.
"""

import re

from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d, arith as arith_d, affine as affine_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.ir import BlockArgument, Module, MemRefType
from allo._mlir.passmanager import PassManager

from ....utils import get_func_inputs_outputs, get_dtype_and_shape_from_type
from ..utils.type_utils import allo_dtype_to_dslx_type, emit_float_defs, get_zero_literal
from .instruction_emitter import InstructionEmitter
from .memory import (
    RAM_TEMPLATE,
    MemoryBinding,
    MemoryEmitter,
    discover_memory_bindings,
    build_memory_channels,
)


class ProcLoweringContext:
    """Context for proc lowering - tracks values, tokens, and state."""

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
        self.pending_tokens = []
        self.inst_emitter = InstructionEmitter(self)

    def new_tmp(self):
        """Generate a new temporary variable name."""
        name = f"tmp{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def new_tok(self):
        """Generate a new token variable name."""
        name = f"tok{self.tok_counter}"
        self.tok_counter += 1
        return name

    def register(self, mlir_value, dslx_name):
        """Register a mapping from MLIR value to DSLX name."""
        self.value_map[str(mlir_value)] = dslx_name

    def constant(self, mlir_value, dslx_literal):
        """Register a constant value."""
        self.const_map[str(mlir_value)] = dslx_literal

    def lookup(self, mlir_value):
        """Look up the DSLX name for an MLIR value."""
        key = str(mlir_value)
        return self.const_map.get(key) or self.value_map.get(key)

    def track_token(self, token):
        """Track a pending token for later joining."""
        if token and token != "join()":
            self.pending_tokens.append(token)

    def consume_tokens(self, base_tok, lines):
        """Consume pending tokens into a joined token expression."""
        tokens = ([base_tok] if base_tok and base_tok != "join()" else []) + self.pending_tokens
        self.pending_tokens = []
        if not tokens:
            return "join()"
        if len(tokens) == 1:
            return tokens[0]
        joined = self.new_tok()
        lines.append(f"    let {joined} = join({', '.join(tokens)});")
        return joined


class DslxCombProcLowerer:
    """Lowerer for combinational (non-looping) functions to DSLX procs."""

    def __init__(self, func: func_d.FuncOp):
        self.func = func
        self.ctx = ProcLoweringContext(func)
        self._register_all_constants()

    def _register_all_constants(self):
        """Pre-register all constants so they're available during lowering."""
        def visit(block):
            for op in block.operations:
                if isinstance(op, arith_d.ConstantOp):
                    self.ctx.inst_emitter.emit(op)
                for region in op.regions:
                    for blk in region.blocks:
                        visit(blk)
        for block in self.func.body.blocks:
            visit(block)

    def _func_args_for_io(self):
        """Get function arguments excluding streams."""
        return [arg for arg in self.func.arguments if "!allo.stream" not in str(arg.type)]

    def _build_channel_decls(self):
        """Build channel declarations for inputs and outputs."""
        inputs, outputs = get_func_inputs_outputs(self.func)
        func_args = self._func_args_for_io()
        assert len(inputs) == len(func_args)

        self.ctx.input_types = []
        self.ctx.scalar_arg_order = []
        self.ctx.inputs = {}
        self.ctx.outputs = {}
        channels = []
        handles = []

        for idx, (arg, (dtype, _)) in enumerate(zip(func_args, inputs)):
            dslx_type = allo_dtype_to_dslx_type(dtype)
            self.ctx.input_types.append(dslx_type)
            self.ctx.inputs[idx] = f"in{idx}"
            self.ctx.scalar_arg_order.append(arg)
            channels.append(f"in{idx}: chan<{dslx_type}> in")
            handles.append(f"in{idx}")

        for idx, (dtype, _) in enumerate(outputs):
            dslx_type = allo_dtype_to_dslx_type(dtype)
            self.ctx.outputs[idx] = f"out{idx}"
            channels.append(f"out{idx}: chan<{dslx_type}> out")
            handles.append(f"out{idx}")

        self.ctx.channel_handles = handles
        return channels

    def _emit_channels(self):
        """Emit channel declarations."""
        self.ctx.channels = self._build_channel_decls()
        return "\n".join([f"  {s};" for s in self.ctx.channels])

    def _emit_config(self):
        """Emit config section."""
        return f"  config({', '.join(self.ctx.channels)}) {{ ({', '.join(self.ctx.channel_handles)}) }}"

    def _emit_inputs(self):
        """Emit input receive operations."""
        lines = []
        for idx, mlir_arg in enumerate(self.ctx.scalar_arg_order):
            var = self.ctx.new_tmp()
            tok = self.ctx.new_tok()
            lines.append(f"    let ({tok}, {var}) = recv(join(), {self.ctx.inputs[idx]});")
            self.ctx.register(mlir_arg, var)
        return lines

    def _emit_body(self):
        """Emit function body operations."""
        lines = []
        ops = list(self.func.body.blocks[0].operations)
        for op in ops:
            lines.extend(self.ctx.inst_emitter.emit(op))
        return lines

    def _emit_init(self):
        """Emit init section."""
        return "  init { () }"

    def _emit_next(self):
        """Emit next section."""
        return "\n".join(["  next(state: ()) {"] + self._emit_inputs() + self._emit_body() + ["  }"])

    def emit_proc(self):
        """Emit complete proc definition."""
        return "\n".join([
            f"pub proc {self.func.name.value} {{",
            self._emit_channels(),
            "",
            self._emit_config(),
            "",
            self._emit_init(),
            "",
            self._emit_next(),
            "}"
        ])


class DslxStatefulProcLowerer:
    """Lowerer for stateful (looping) functions to DSLX procs.
    
    Handles functions with for/while loops by converting them to
    stateful procs with init/next semantics.
    """

    uses_memory = False

    def __init__(self, func: func_d.FuncOp):
        self.func = func
        self.ctx = ProcLoweringContext(func)
        
        # Memory bindings
        self.memory_bindings, self.memory_map = discover_memory_bindings(func)
        self.mem_emitter = MemoryEmitter(self.ctx)
        self.uses_memory = any(b.needs_read or b.needs_write for b in self.memory_bindings)
        
        # Loop analysis
        self.loop = None
        self.loop_type = None
        self.state = []
        self.loops = []
        self.loop_preambles = []
        self.loop_postambles = []
        self.loop_upper_bounds = []
        self.return_state_idx = 0
        
        self._register_all_constants()
        self._analyze()

    def _register_all_constants(self):
        """Pre-register all constants."""
        def visit(block):
            for op in block.operations:
                if isinstance(op, arith_d.ConstantOp):
                    self.ctx.inst_emitter.emit(op)
                for region in op.regions:
                    for blk in region.blocks:
                        visit(blk)
        for block in self.func.body.blocks:
            visit(block)

    def _analyze(self):
        """Analyze function structure to find loops and state variables."""
        block = self.func.body.blocks[0]
        
        # Find the main loop
        for op in block.operations:
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                self.loop = op
                self.loop_type = "for"
                self._collect_nested_loops(op)
                break
            elif isinstance(op, scf_d.WhileOp):
                self.loop = op
                self.loop_type = "while"
                self._collect_nested_loops_from_block(list(op.after.blocks)[0], op)
                break

        # Collect state variables (memref allocations used in loop)
        ordered_allocs = self._collect_ordered_allocs(block)
        memrefs = {m for m in ordered_allocs if m != self.loop}
        
        if self.loop:
            if self.loop_type == "for":
                memrefs |= self._collect_loop_allocs(self.loop)
                used = self._collect_used_memrefs(self._loop_body(self.loop), memrefs)
            else:
                after_block = list(self.loop.after.blocks)[0]
                memrefs |= self._collect_allocs_from_block(after_block)
                used = (
                    self._collect_used_memrefs(after_block, memrefs) |
                    self._collect_used_memrefs(list(self.loop.before.blocks)[0], memrefs)
                )
        else:
            used = set()

        for m in ordered_allocs:
            if m not in used:
                continue
            if "memref" in str(getattr(m, "type", "")) and str(m) not in self.memory_map:
                mt = MemRefType(m.type)
                dtype, shape = get_dtype_and_shape_from_type(mt)
                self.state.append((m, allo_dtype_to_dslx_type(dtype), self._get_init(m)))

        self._find_return_state_idx(block)

    def _collect_nested_loops(self, loop_op):
        """Recursively collect nested loop structure."""
        self._collect_nested_loops_from_block(self._loop_body(loop_op), loop_op)

    def _collect_nested_loops_from_block(self, body, loop_op):
        """Collect loops from a block."""
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
        self.loop_upper_bounds.append(
            None if isinstance(loop_op, scf_d.WhileOp) else self._extract_loop_upper_bound(loop_op)
        )

        if nested:
            if isinstance(nested, scf_d.WhileOp):
                self._collect_nested_loops_from_block(list(nested.after.blocks)[0], nested)
            else:
                self._collect_nested_loops(nested)

    def _extract_loop_upper_bound(self, loop_op):
        """Extract upper bound from affine.for loop."""
        if isinstance(loop_op, affine_d.AffineForOp):
            try:
                header = loop_op.operation.get_asm().splitlines()[0]
            except AttributeError:
                header = str(loop_op.operation).splitlines()[0]
            match = re.search(r"\bto\b\s+([0-9]+)", header)
            if match:
                return f"s32:{match.group(1)}"
        return None

    def _loop_body(self, loop_op):
        """Get the body block of a loop."""
        if hasattr(loop_op, "body"):
            return loop_op.body
        if hasattr(loop_op, "region"):
            return list(loop_op.region.blocks)[0]
        raise NotImplementedError(f"unsupported loop: {type(loop_op)}")

    def _loop_induction_var(self, loop_op):
        """Get the induction variable of a loop."""
        if hasattr(loop_op, "induction_variable"):
            return loop_op.induction_variable
        body = getattr(loop_op, "body", None)
        if body and body.arguments:
            return body.arguments[0]
        raise NotImplementedError(f"no induction var for {type(loop_op)}")

    def _collect_ordered_allocs(self, block):
        """Collect alloc operations in order."""
        allocs = []
        seen = set()

        def scan(b):
            for op in b.operations:
                if isinstance(op, memref_d.AllocOp):
                    key = str(op.result)
                    if key not in seen:
                        seen.add(key)
                        allocs.append(op.result)
                if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                    scan(self._loop_body(op))

        scan(block)
        return allocs

    def _collect_loop_allocs(self, loop_op):
        """Collect allocations in loop body."""
        return self._collect_allocs_from_block(self._loop_body(loop_op))

    def _collect_allocs_from_block(self, block):
        """Collect all alloc operations from a block."""
        if not block:
            return set()
        allocs = set()
        for op in block.operations:
            if isinstance(op, memref_d.AllocOp):
                allocs.add(op.result)
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                allocs |= self._collect_allocs_from_block(self._loop_body(op))
        return allocs

    def _collect_used_memrefs(self, block, memrefs):
        """Collect memrefs that are actually used in a block."""
        used = set()
        for op in block.operations:
            used.update(o for o in op.operands if o in memrefs)
            if isinstance(op, (scf_d.ForOp, affine_d.AffineForOp)):
                used |= self._collect_used_memrefs(self._loop_body(op), memrefs)
        return used

    def _get_init(self, memref):
        """Get initial value for a state memref from stores before the loop."""
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
                owner = v.owner.opview if hasattr(v.owner, "opview") else v.owner
                if isinstance(owner, arith_d.ConstantOp):
                    return str(owner.value).split(":")[0].strip()
        return "0"

    def _find_return_state_idx(self, block):
        """Find which state variable is returned by the function."""
        state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
        for op in block.operations:
            if isinstance(op, func_d.ReturnOp) and op.operands:
                owner = op.operands[0].owner.opview if hasattr(op.operands[0].owner, "opview") else op.operands[0].owner
                if isinstance(owner, affine_d.AffineLoadOp):
                    key = str(owner.operands[0])
                    if key in state_memrefs:
                        self.return_state_idx = state_memrefs[key]
                break

    def _func_args_for_io(self):
        """Get function arguments excluding streams."""
        return [arg for arg in self.func.arguments if "!allo.stream" not in str(arg.type)]

    def _build_channel_decls(self):
        """Build channel declarations for inputs, outputs, and memory."""
        inputs, outputs = get_func_inputs_outputs(self.func)
        func_args = self._func_args_for_io()

        self.ctx.input_types = []
        self.ctx.scalar_arg_order = []
        self.ctx.inputs = {}
        self.ctx.outputs = {}
        channels = []
        handles = []

        # Input channels (scalars only for stateful procs)
        for idx, (arg, (dtype, shape)) in enumerate(zip(func_args, inputs)):
            if shape:  # Skip array inputs - handled by memory
                continue
            dslx_type = allo_dtype_to_dslx_type(dtype)
            self.ctx.input_types.append(dslx_type)
            self.ctx.inputs[idx] = f"in{idx}"
            self.ctx.scalar_arg_order.append(arg)
            channels.append(f"in{idx}: chan<{dslx_type}> in")
            handles.append(f"in{idx}")

        # Output channels
        for idx, (dtype, shape) in enumerate(outputs):
            dslx_type = allo_dtype_to_dslx_type(dtype, shape)
            self.ctx.outputs[idx] = f"out{idx}"
            channels.append(f"out{idx}: chan<{dslx_type}> out")
            handles.append(f"out{idx}")

        # Memory channels
        channels.extend(build_memory_channels(self.memory_bindings))
        for binding in self.memory_bindings:
            if binding.needs_read:
                handles.extend([binding.read_req_chan, binding.read_resp_chan])
            if binding.needs_write:
                handles.extend([binding.write_req_chan, binding.write_resp_chan])

        self.ctx.channel_handles = handles
        return channels

    def _emit_channels(self):
        """Emit channel declarations."""
        self.ctx.channels = self._build_channel_decls()
        return "\n".join([f"  {s};" for s in self.ctx.channels])

    def _emit_config(self):
        """Emit config section."""
        return f"  config({', '.join(self.ctx.channels)}) {{ ({', '.join(self.ctx.channel_handles)}) }}"

    def _emit_init(self):
        """Emit init section with state initialization."""
        n = len(self.state)
        num_loops = len(self.loops)
        
        if n == 0 and num_loops == 0:
            return "  init { () }"
        
        # Build initial state tuple
        init_vals = []
        for _, dslx_type, init_val in self.state:
            if init_val.startswith("__arg"):
                init_vals.append(f"{dslx_type}:0")  # Placeholder
            else:
                init_vals.append(f"{dslx_type}:{init_val}")
        
        # Add loop indices (start at 0)
        for _ in range(num_loops):
            init_vals.append("s32:0")
        
        # Add busy flag (start not busy)
        init_vals.append("bool:false")
        
        init_tuple = ", ".join(init_vals)
        return f"  init {{ ({init_tuple}) }}"

    def _emit_next(self):
        """Emit next section with loop body and state updates."""
        n = len(self.state)
        acc_names = [f"acc{i}" for i in range(n)]
        num_loops = len(self.loops)
        
        # Build state type
        loop_types = ["s32"] * num_loops
        loop_vars = [f"index{i}" for i in range(num_loops)]
        state_types = [t for _, t, _ in self.state] + loop_types + ["bool"]
        state_vars = acc_names + loop_vars + ["busy"]
        state_type = "(" + ", ".join(state_types) + ")"
        
        lines = [f"  next(state: {state_type}) {{"]
        lines.append(f"    let ({', '.join(state_vars)}) = state;")
        
        # Receive inputs when not busy
        recv_lines = self._emit_recv_inputs("!busy")
        lines.extend(recv_lines)
        
        # Get upper bounds
        ubs = []
        for i in range(num_loops):
            if i < len(self.loop_upper_bounds) and self.loop_upper_bounds[i]:
                ubs.append(self.loop_upper_bounds[i])
            else:
                ubs.append("s32:0")
        
        # Register loop indices
        for i, loop_op in enumerate(self.loops):
            self.ctx.register(self._loop_induction_var(loop_op), f"index{i}")
        
        # Register state accumulators
        for i, (m, _, _) in enumerate(self.state):
            self.ctx.register(m, acc_names[i])
        
        # Build loop iteration guard
        guard_parts = [f"index{i} < {ubs[i]}" for i in range(num_loops)]
        loop_guard = " && ".join(guard_parts) if guard_parts else "true"
        
        # Emit loop body with guard
        lines.append(f"    if {loop_guard} {{")
        body_lines, updated = self._emit_body_ops()
        lines.extend(body_lines)
        
        # Update state for next iteration
        next_state = []
        for i, (m, _, _) in enumerate(self.state):
            next_state.append(updated.get(i, acc_names[i]))
        
        # Increment innermost loop index
        if num_loops > 0:
            for i in range(num_loops - 1):
                next_state.append(f"index{i}")
            next_state.append(f"index{num_loops - 1} + s32:1")
        
        next_state.append("true")  # Still busy
        
        lines.append(f"      ({', '.join(next_state)})")
        lines.append("    } else {")
        
        # Send output and reset
        if self.ctx.outputs:
            out_chan = list(self.ctx.outputs.values())[0]
            out_val = acc_names[self.return_state_idx] if acc_names else "s32:0"
            lines.append(f"      send(join(), {out_chan}, {out_val});")
        
        # Reset state
        reset_state = []
        for _, dslx_type, init_val in self.state:
            if init_val.startswith("__arg"):
                reset_state.append(f"{dslx_type}:0")
            else:
                reset_state.append(f"{dslx_type}:{init_val}")
        for _ in range(num_loops):
            reset_state.append("s32:0")
        reset_state.append("bool:false")
        
        lines.append(f"      ({', '.join(reset_state)})")
        lines.append("    }")
        lines.append("  }")
        
        return "\n".join(lines)

    def _emit_recv_inputs(self, guard):
        """Emit receive operations for inputs."""
        lines = []
        for idx, mlir_arg in enumerate(self.ctx.scalar_arg_order):
            if idx not in self.ctx.inputs:
                continue
            tok = self.ctx.new_tok()
            var = self.ctx.new_tmp()
            dslx_type = self.ctx.input_types[idx] if idx < len(self.ctx.input_types) else "s32"
            lines.append(
                f"    let ({tok}, {var}) = recv_if(join(), {self.ctx.inputs[idx]}, {guard}, {dslx_type}:0);"
            )
            self.ctx.register(mlir_arg, var)
        return lines

    def _emit_body_ops(self):
        """Emit loop body operations and track state updates."""
        state_memrefs = {str(m): i for i, (m, _, _) in enumerate(self.state)}
        updated = {}
        lines = []
        
        if not self.loops:
            return lines, updated
        
        # Get the innermost loop body
        body = self._loop_body(self.loops[-1])
        
        for op in body.operations:
            if isinstance(op, (arith_d.ConstantOp, scf_d.YieldOp, affine_d.AffineYieldOp)):
                continue
            
            # Handle memory loads
            if isinstance(op, affine_d.AffineLoadOp):
                mkey = str(op.operands[0])
                binding = self.memory_map.get(mkey)
                if binding:
                    lines.extend(self.mem_emitter.emit_read(binding, op))
                elif mkey in state_memrefs:
                    # State variable load
                    idx = state_memrefs[mkey]
                    self.ctx.register(op.result, f"acc{idx}")
                else:
                    lines.extend(self.ctx.inst_emitter.emit(op))
                continue
            
            # Handle memory stores
            if isinstance(op, affine_d.AffineStoreOp):
                mkey = str(op.operands[1])
                binding = self.memory_map.get(mkey)
                if binding:
                    lines.extend(self.mem_emitter.emit_write(binding, op))
                elif mkey in state_memrefs:
                    # State variable store
                    idx = state_memrefs[mkey]
                    val = self.ctx.lookup(op.operands[0])
                    updated[idx] = val
                    self.ctx.register(op.operands[1], val)
                else:
                    lines.extend(self.ctx.inst_emitter.emit(op))
                continue
            
            # Other operations
            lines.extend(self.ctx.inst_emitter.emit(op))
        
        return lines, updated

    def emit_proc(self):
        """Emit complete proc definition."""
        return "\n".join([
            f"pub proc {self.func.name.value} {{",
            self._emit_channels(),
            "",
            self._emit_config(),
            "",
            self._emit_init(),
            "",
            self._emit_next(),
            "}"
        ])


def _has_non_unrolled_loops(func):
    """Check if function has any non-unrolled for loops or while loops."""
    def check_block(block):
        for op in block.operations:
            if isinstance(op, scf_d.WhileOp):
                return True
            if isinstance(op, scf_d.ForOp):
                return True
            if isinstance(op, affine_d.AffineForOp):
                # Check if unrolled
                unroll_attr = None
                # Handle both dict-style and list-style attributes
                if isinstance(op.attributes, dict):
                    unroll_attr = op.attributes.get("unroll")
                else:
                    for attr in op.attributes:
                        if hasattr(attr, "name") and attr.name == "unroll":
                            unroll_attr = attr
                            break
                        elif isinstance(attr, str) and attr == "unroll":
                            # Attribute might be a string key
                            continue
                if unroll_attr is None:
                    return True
                # Check nested loops
                body = op.body if hasattr(op, "body") else list(op.region.blocks)[0]
                if check_block(body):
                    return True
            for region in op.regions:
                for blk in region.blocks:
                    if check_block(blk):
                        return True
        return False
    return check_block(func.body.blocks[0])


class MlirToDslxProcLowerer:
    """Top-level lowerer that chooses the appropriate strategy for each function."""

    def __init__(self, module: Module, top_func_name: str):
        self.module = module
        self.top_func_name = top_func_name
        self.func_lowerers = []
        
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                if _has_non_unrolled_loops(op):
                    self.func_lowerers.append(DslxStatefulProcLowerer(op))
                else:
                    self.func_lowerers.append(DslxCombProcLowerer(op))

    def emit_module(self):
        """Emit complete DSLX module."""
        body = "\n\n".join(fl.emit_proc() for fl in self.func_lowerers)
        
        # Collect float types used
        float_types = set()
        for fl in self.func_lowerers:
            float_types.update(fl.ctx.inst_emitter.float_types_used)
        
        # Build preamble
        preamble = []
        if float_types:
            preamble.append(emit_float_defs(float_types))
        if any(getattr(fl, "uses_memory", False) for fl in self.func_lowerers):
            preamble.append(RAM_TEMPLATE)
        
        if preamble:
            return "\n\n".join(preamble) + "\n\n" + body
        return body


def clean_mlir(module):
    """Canonicalize MLIR before lowering."""
    with module.context:
        PassManager.parse("builtin.module(canonicalize, sccp, cse, symbol-dce)").run(module.operation)


def lower_mlir_to_proc(module: Module, top_func_name: str) -> str:
    """Lower MLIR module to DSLX proc code.
    
    Args:
        module: MLIR module
        top_func_name: Name of the top-level function
        
    Returns:
        DSLX code string
    """
    clean_mlir(module)
    return MlirToDslxProcLowerer(module, top_func_name).emit_module()
