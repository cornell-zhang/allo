"""MLIR to XLS IR function lowering implementation."""

from allo._mlir.ir import Operation, MemRefType
from allo._mlir.ir import IntegerAttr, FloatAttr, DenseElementsAttr
from allo._mlir.dialects import affine as affine_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import linalg as linalg_d
from allo._mlir.dialects import memref as memref_d


class IRNode:
    """Base class for XLS IR nodes."""
    def __init__(self, name, ir_type):
        self.name = name
        self.ir_type = ir_type


class IRContext:
    """Context for XLS IR generation."""
    def __init__(self):
        self.bindings = {}  # MLIR value -> IR node name
        self.ir_lines = []  # Generated IR lines
        self.node_counter = 0
        self.memref_shapes = {}
        self.result_buffer = None
        self.last_result_value = None  # Track the final SSA value for return

    def fresh_name(self, prefix="tmp"):
        """Generate fresh temporary name."""
        name = f"{prefix}_{self.node_counter}"
        self.node_counter += 1
        return name

    def bind(self, mlir_value, ir_name):
        """Bind MLIR value to IR node name."""
        self.bindings[mlir_value] = ir_name

    def lookup(self, mlir_value):
        """Lookup IR node name for MLIR value."""
        return self.bindings.get(mlir_value, None)

    def emit(self, line):
        """Emit a line of IR."""
        self.ir_lines.append(line)


class MlirToXlsIRLowerer:
    """Lower MLIR to XLS IR."""

    def __init__(self, func_op: func_d.FuncOp):
        self.func_op = func_op
        self.ctx = IRContext()

    def lower(self):
        """Main lowering entry point."""
        # Process function signature
        func_name = self.func_op.name.value
        params = []

        for i, arg in enumerate(self.func_op.arguments):
            if isinstance(arg.type, MemRefType):
                param_name = f"arg{i}"
                shape = list(arg.type.shape)
                self.ctx.memref_shapes[param_name] = shape
                self.ctx.bind(arg, param_name)

                # XLS IR type: bits[32] for scalars, array types for arrays
                # For now, flatten 2D arrays to 1D
                total_size = 1
                for dim in shape:
                    total_size *= dim
                ir_type = f"bits[32][{total_size}]"
                params.append((param_name, ir_type))

        # Lower function body
        for block in self.func_op.body:
            for op in block.operations:
                self.lower_op(op)

        # Determine return value
        if self.ctx.result_buffer:
            return_name = self.ctx.result_buffer
        else:
            if self.ctx.memref_shapes:
                return_name = list(self.ctx.memref_shapes.keys())[-1]
            else:
                return_name = "arg0"

        return self.emit_ir(func_name, params, return_name)

    def lower_op(self, op: Operation):
        """Lower a single MLIR operation."""
        if isinstance(op, affine_d.AffineForOp):
            self.lower_for(op)
        elif isinstance(op, affine_d.AffineApplyOp):
            self.lower_affine_apply(op)
        elif isinstance(op, affine_d.AffineLoadOp):
            self.lower_load(op)
        elif isinstance(op, affine_d.AffineStoreOp):
            self.lower_store(op)
        elif isinstance(op, arith_d.ConstantOp):
            self.lower_constant(op)
        elif isinstance(op, arith_d.AddIOp):
            self.lower_add(op)
        elif isinstance(op, arith_d.MulIOp):
            self.lower_mul(op)
        elif isinstance(op, arith_d.ExtSIOp):
            self.lower_extsi(op)
        elif isinstance(op, arith_d.ExtUIOp):
            self.lower_extui(op)
        elif isinstance(op, arith_d.TruncIOp):
            self.lower_trunci(op)
        elif isinstance(op, linalg_d.FillOp):
            self.lower_fill(op)
        elif isinstance(op, memref_d.AllocOp):
            self.lower_alloc(op)
        elif isinstance(op, affine_d.AffineYieldOp):
            return
        elif isinstance(op, func_d.ReturnOp):
            return
        else:
            print(f"Warning: unhandled op {op.operation.name}")

    def lower_constant(self, op):
        """Lower constant operation."""
        attr = op.value
        if isinstance(attr, IntegerAttr) or isinstance(attr, FloatAttr):
            raw = attr.value
        elif isinstance(attr, DenseElementsAttr):
            raw = list(attr)[0]
        else:
            raw = int(str(attr).split(":")[0])

        name = self.ctx.fresh_name("const")
        self.ctx.emit(f"  {name}: bits[32] = literal(value={raw})")
        self.ctx.bind(op.result, name)

    def lower_alloc(self, op: memref_d.AllocOp):
        """Lower memory allocation."""
        try:
            name_attr = op.attributes['name']
            buf_name = name_attr.value if hasattr(name_attr, "value") else str(name_attr).strip('"')
        except (KeyError, AttributeError):
            buf_name = "C"

        memref_type = op.result.type
        if isinstance(memref_type, MemRefType):
            shape = list(memref_type.shape)
            self.ctx.memref_shapes[buf_name] = shape

        self.ctx.result_buffer = buf_name
        self.ctx.bind(op.result, buf_name)

        # Initialize array to zeros
        total_size = 1
        for dim in shape:
            total_size *= dim

        # Create zero literal
        zero_name = self.ctx.fresh_name("zero")
        self.ctx.emit(f"  {zero_name}: bits[32] = literal(value=0)")

        # Create array of zeros
        self.ctx.emit(f"  {buf_name}: bits[32][{total_size}] = array({', '.join([zero_name] * total_size)})")

    def lower_add(self, op: arith_d.AddIOp):
        """Lower addition."""
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        result = self.ctx.fresh_name("add")
        self.ctx.emit(f"  {result}: bits[32] = add({lhs}, {rhs})")
        self.ctx.bind(op.result, result)

    def lower_mul(self, op: arith_d.MulIOp):
        """Lower multiplication."""
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        result = self.ctx.fresh_name("mul")
        self.ctx.emit(f"  {result}: bits[64] = umul({lhs}, {rhs})")
        self.ctx.bind(op.result, result)

    def lower_extsi(self, op: arith_d.ExtSIOp):
        """Lower sign extension - for now just pass through."""
        operand = self.ctx.lookup(op.operands[0])
        result = self.ctx.fresh_name("ext")
        # XLS IR: sign_ext to 64 bits
        self.ctx.emit(f"  {result}: bits[64] = sign_ext({operand}, new_bit_count=64)")
        self.ctx.bind(op.result, result)

    def lower_extui(self, op: arith_d.ExtUIOp):
        """Lower unsigned zero extension."""
        operand = self.ctx.lookup(op.operands[0])
        result = self.ctx.fresh_name("ext")
        # XLS IR: zero_ext to 64 bits
        self.ctx.emit(f"  {result}: bits[64] = zero_ext({operand}, new_bit_count=64)")
        self.ctx.bind(op.result, result)

    def lower_trunci(self, op: arith_d.TruncIOp):
        """Lower truncation."""
        operand = self.ctx.lookup(op.operands[0])
        result = self.ctx.fresh_name("trunc")
        # Truncate back to 32 bits
        self.ctx.emit(f"  {result}: bits[32] = bit_slice({operand}, start=0, width=32)")
        self.ctx.bind(op.result, result)

    def lower_load(self, op: affine_d.AffineLoadOp):
        """Lower array load."""
        base = op.memref
        buf_name = self.ctx.lookup(base)

        if not buf_name:
            print(f"Warning: unknown buffer for load")
            return

        # Calculate linear index for 2D array
        indices = self.lower_affine_index(op.indices)
        if isinstance(indices, list) and len(indices) == 2:
            # 2D access: linearize as i * cols + j
            i_name = indices[0]
            j_name = indices[1]
            shape = self.ctx.memref_shapes.get(buf_name, [32, 32])
            cols = shape[1] if len(shape) > 1 else 1

            # i * cols
            cols_const = self.ctx.fresh_name("cols")
            self.ctx.emit(f"  {cols_const}: bits[32] = literal(value={cols})")
            i_times_cols = self.ctx.fresh_name("offset")
            self.ctx.emit(f"  {i_times_cols}: bits[32] = umul({i_name}, {cols_const})")

            # i * cols + j
            linear_idx = self.ctx.fresh_name("idx")
            self.ctx.emit(f"  {linear_idx}: bits[32] = add({i_times_cols}, {j_name})")
            idx_name = linear_idx
        else:
            idx_name = indices

        result = self.ctx.fresh_name("load")
        self.ctx.emit(f"  {result}: bits[32] = array_index({buf_name}, indices=[{idx_name}])")
        self.ctx.bind(op.result, result)

    def lower_store(self, op: affine_d.AffineStoreOp):
        """Lower array store."""
        base = op.memref
        buf_name = self.ctx.lookup(base)
        value = self.ctx.lookup(op.value)

        if not buf_name or not value:
            print(f"Warning: unknown buffer or value for store")
            return

        # Calculate linear index
        indices = self.lower_affine_index(op.indices)
        if isinstance(indices, list) and len(indices) == 2:
            i_name = indices[0]
            j_name = indices[1]
            shape = self.ctx.memref_shapes.get(buf_name, [32, 32])
            cols = shape[1] if len(shape) > 1 else 1

            cols_const = self.ctx.fresh_name("cols")
            self.ctx.emit(f"  {cols_const}: bits[32] = literal(value={cols})")
            i_times_cols = self.ctx.fresh_name("offset")
            self.ctx.emit(f"  {i_times_cols}: bits[32] = umul({i_name}, {cols_const})")
            linear_idx = self.ctx.fresh_name("idx")
            self.ctx.emit(f"  {linear_idx}: bits[32] = add({i_times_cols}, {j_name})")
            idx_name = linear_idx
        else:
            idx_name = indices

        # Get array size for the type annotation
        shape = self.ctx.memref_shapes.get(buf_name, [])
        total_size = 1
        for dim in shape:
            total_size *= dim

        result = self.ctx.fresh_name("update")
        self.ctx.emit(f"  {result}: bits[32][{total_size}] = array_update({buf_name}, {value}, indices=[{idx_name}])")
        # Update binding to new array version
        self.ctx.bind(base, result)
        # Also update in bindings by name
        for key, val in list(self.ctx.bindings.items()):
            if val == buf_name:
                self.ctx.bindings[key] = result
        # Keep the shape information for the new buffer name
        if buf_name in self.ctx.memref_shapes:
            self.ctx.memref_shapes[result] = self.ctx.memref_shapes[buf_name]

        # Track this as the last result value - the final update will be the return value
        # Check if this buffer is tracked in memref_shapes (meaning it's an allocated result buffer)
        if buf_name in self.ctx.memref_shapes or result in self.ctx.memref_shapes:
            self.ctx.last_result_value = result

    def lower_fill(self, op: linalg_d.FillOp):
        """Lower fill operation."""
        val = self.ctx.lookup(op.inputs[0])
        out_base = op.outputs[0]
        buf_name = self.ctx.lookup(out_base)

        if not buf_name:
            print(f"Warning: unknown buffer for fill")
            return

        # Get array size
        shape = self.ctx.memref_shapes.get(buf_name, [32, 32])
        total_size = 1
        for dim in shape:
            total_size *= dim

        # Create array filled with value
        fill_result = self.ctx.fresh_name("fill")
        self.ctx.emit(f"  {fill_result}: bits[32][{total_size}] = array({', '.join([val] * total_size)})")
        self.ctx.bind(out_base, fill_result)
        # Keep shape information for the filled buffer
        if buf_name in self.ctx.memref_shapes:
            self.ctx.memref_shapes[fill_result] = self.ctx.memref_shapes[buf_name]

    def lower_for(self, op: affine_d.AffineForOp):
        """Lower for loop by unrolling."""
        # TODO: This kinda just pattern matches on some simple cases and doesnt generalize well.

        # Extract bounds
        lb = self.extract_bound(op.lowerBoundMap)
        ub = self.extract_bound(op.upperBoundMap)

        # Unroll the loop
        for i in range(lb, ub):
            # Create constant for loop index
            iv_name = self.ctx.fresh_name("i")
            self.ctx.emit(f"  {iv_name}: bits[32] = literal(value={i})")
            self.ctx.bind(op.induction_variable, iv_name)

            # Lower body operations for this iteration
            # Note: op.body is a Block, iterating over it gives operations directly
            for nested_op in op.body:
                if not isinstance(nested_op, affine_d.AffineYieldOp):
                    self.lower_op(nested_op)

    def lower_affine_apply(self, op: affine_d.AffineApplyOp):
        """Lower affine.apply operation.

        Typical affine map: (d0, d1) -> (d0 + d1 * factor)
        This computes: inner + outer * factor
        """
        # Get the affine map
        amap = op.map.value
        expr = amap.results[0]

        # Get operands
        operands = list(op.operands)
        operand_names = [self.ctx.lookup(operand) for operand in operands]

        # Parse and evaluate the affine expression
        # For simple cases like (d0, d1) -> (d0 + d1 * factor)
        expr_str = str(expr).strip()

        result_name = self.ctx.fresh_name("affine")

        # Try to parse common affine expression patterns
        if len(operands) == 2:
            # Common case: d0 + d1 * factor
            # Expression like: "d0 + d1 * 2"
            d0_name = operand_names[0]
            d1_name = operand_names[1]

            # Try to extract the factor from the expression
            # Look for "d1 * <number>" or just use simple addition if no multiplication
            if '*' in expr_str:
                # Extract factor - look for number after *
                parts = expr_str.split('*')
                if len(parts) >= 2:
                    factor_str = parts[-1].strip().rstrip(')')
                    try:
                        factor = int(factor_str)
                    except:
                        factor = 2  # default
                else:
                    factor = 2

                # Emit: factor_const = literal(value=factor)
                factor_const = self.ctx.fresh_name("factor")
                self.ctx.emit(f"  {factor_const}: bits[32] = literal(value={factor})")

                # Emit: mul_result = umul(d1, factor)
                mul_result = self.ctx.fresh_name("mul")
                self.ctx.emit(f"  {mul_result}: bits[32] = umul({d1_name}, {factor_const})")

                # Emit: result = add(d0, mul_result)
                self.ctx.emit(f"  {result_name}: bits[32] = add({d0_name}, {mul_result})")
            else:
                # Simple addition: d0 + d1
                self.ctx.emit(f"  {result_name}: bits[32] = add({d0_name}, {d1_name})")
        elif len(operands) == 1:
            # Identity map: just use the operand
            result_name = operand_names[0]
        else:
            # Fallback: create a zero for unsupported affine maps
            self.ctx.emit(f"  {result_name}: bits[32] = literal(value=0)")

        self.ctx.bind(op.result, result_name)

    def lower_affine_index(self, indices):
        """Lower affine indices."""
        index_names = []
        for idx in indices:
            name = self.ctx.lookup(idx)
            if not name:
                # Create constant if needed
                name = self.ctx.fresh_name("idx")
                self.ctx.emit(f"  {name}: bits[32] = literal(value=0)")
                self.ctx.bind(idx, name)
            index_names.append(name)

        if len(index_names) == 1:
            return index_names[0]
        else:
            return index_names

    def extract_bound(self, map_like):
        """Extract constant from affine map."""
        try:
            amap = map_like.value
            expr = amap.results[0]
            if hasattr(expr, "value"):
                return expr.value
            expr_str = str(expr).strip()
            if expr_str.isdigit():
                return int(expr_str)
            return 0
        except:
            return 0

    def emit_ir(self, func_name, params, return_name):
        """Emit complete XLS IR function."""
        lines = []

        # Function signature
        param_list = ", ".join(f"{name}: {typ}" for name, typ in params)

        # Determine return type
        if return_name in self.ctx.memref_shapes:
            shape = self.ctx.memref_shapes[return_name]
            total_size = 1
            for dim in shape:
                total_size *= dim
            ret_type = f"bits[32][{total_size}]"
        else:
            ret_type = "bits[32][1024]"  # default

        lines.append(f"fn {func_name}({param_list}) -> {ret_type} {{")

        # Function body
        lines.extend(self.ctx.ir_lines)

        # Return statement - use last_result_value if available
        if self.ctx.last_result_value:
            lines.append(f"  ret result: {ret_type} = identity({self.ctx.last_result_value})")
        else:
            # Fallback: create a literal array (shouldn't normally happen)
            lines.append(f"  ret result: {ret_type} = identity({return_name})")
        lines.append("}")

        return "\n".join(lines)
