"""MLIR to DSLX function lowering implementation."""

from allo._mlir.ir import Operation, MemRefType
from allo._mlir.ir import IntegerAttr, FloatAttr, DenseElementsAttr
from allo._mlir.dialects import affine as affine_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import linalg as linalg_d
from allo._mlir.dialects import memref as memref_d

from .shared.codegen_context import CodegenContext
from .shared.dslx_nodes import (
    DslxVar, DslxConst, DslxBinOp, DslxLoad, DslxStore,
    DslxFor, DslxLet, DslxArrayInit
)


class MlirToDslxLowerer:
    def __init__(self, func_op: func_d.FuncOp):
        self.func_op = func_op
        self.ctx = CodegenContext()

    def lower(self):
        params = []
        # Get function name
        func_name = "simple_add"  # default
        if "sym_name" in self.func_op.attributes:
            func_name = self.func_op.attributes["sym_name"].value
        elif hasattr(self.func_op, "name") and self.func_op.name:
            func_name = self.func_op.name.value if hasattr(self.func_op.name, "value") else str(self.func_op.name)
        
        # Get return type
        return_type = None
        if self.func_op.type.results:
            ret_mlir_type = self.func_op.type.results[0]
            if isinstance(ret_mlir_type, MemRefType):
                shape = list(ret_mlir_type.shape)
                return_type = "u32" + "".join(f"[{dim}]" for dim in shape)
            else:
                # Scalar return type
                return_type = "u32"
        
        # Process arguments
        for i, arg in enumerate(self.func_op.arguments):
            if isinstance(arg.type, MemRefType):
                name = f"arg{i}"
                shape = list(arg.type.shape)
                self.ctx.memref_shapes[name] = shape
                self.ctx.bind(arg, DslxVar(name))
                type_str = "u32" + "".join(f"[{dim}]" for dim in shape)
                params.append((name, type_str))
            else:
                # Scalar argument
                name = f"arg{i}"
                self.ctx.bind(arg, DslxVar(name))
                params.append((name, "u32"))

        # Process all operations and track return value
        return_value = None
        for op in self.func_op.body.blocks[0].operations:
            if isinstance(op, func_d.ReturnOp):
                if op.operands:
                    return_value = self.ctx.lookup(op.operands[0])
                # Don't break - still process other ops if any
                continue
            self.lower_op(op)

        # Determine return expression
        if return_value:
            return_expr = self.emit_expr(return_value)
        elif self.ctx.result_buffer:
            return_expr = self.ctx.result_buffer
        elif self.ctx.memref_shapes:
            return_expr = list(self.ctx.memref_shapes.keys())[-1]
        else:
            return_expr = "arg0"

        return self.emit_dslx(func_name, params, return_expr, return_type)

    def lower_op(self, op: Operation):
        if isinstance(op, affine_d.AffineForOp):
            self.lower_for(op)
        elif isinstance(op, affine_d.AffineLoadOp):
            self.lower_load(op)
        elif isinstance(op, affine_d.AffineStoreOp):
            self.lower_store(op)
        elif isinstance(op, affine_d.AffineApplyOp):
            self.lower_affine_apply(op)
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
            print("Warning: unhandled op", op.operation.name)

    def lower_constant(self, op):
        attr = op.value
        if isinstance(attr, IntegerAttr) or isinstance(attr, FloatAttr):
            raw = attr.value
        elif isinstance(attr, DenseElementsAttr):
            raw = list(attr)[0]
        else:
            raw = int(str(attr).split(":")[0])

        node = DslxConst(raw)
        self.ctx.bind(op.result, node)

    def lower_alloc(self, op: memref_d.AllocOp):
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
        self.ctx.bind(op.result, DslxVar(buf_name))

    def lower_add(self, op: arith_d.AddIOp):
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        node = DslxBinOp("+", lhs, rhs)
        self.ctx.bind(op.result, node)

    def lower_mul(self, op: arith_d.MulIOp):
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        node = DslxBinOp("*", lhs, rhs)
        self.ctx.bind(op.result, node)

    def lower_extsi(self, op: arith_d.ExtSIOp):
        operand = self.ctx.lookup(op.operands[0])
        self.ctx.bind(op.result, operand)

    def lower_extui(self, op: arith_d.ExtUIOp):
        operand = self.ctx.lookup(op.operands[0])
        self.ctx.bind(op.result, operand)

    def lower_trunci(self, op: arith_d.TruncIOp):
        operand = self.ctx.lookup(op.operands[0])
        self.ctx.bind(op.result, operand)

    def lower_affine_apply(self, op: affine_d.AffineApplyOp):
        """Lower affine.apply operations by evaluating the affine expression."""
        # Get the affine map
        aff_map = op.map.value

        # Get operands (the values being substituted into the affine expression)
        operands = [self.ctx.lookup(operand) for operand in op.operands]

        # Evaluate the affine expression (assuming single result)
        if len(aff_map.results) != 1:
            raise RuntimeError(f"Expected single result in affine map, got {len(aff_map.results)}")

        expr = aff_map.results[0]
        result_node = self._eval_affine_expr(expr, operands)

        # Bind the result
        self.ctx.bind(op.result, result_node)

    def _eval_affine_expr(self, expr, operands):
        """Recursively evaluate an affine expression tree.

        Common pattern for split loops: d0 + d1 * factor
        which means: inner + outer * factor
        """
        expr_str = str(expr).strip()

        # Check if it's a dimension reference (d0, d1, etc.)
        if expr_str.startswith('d') and len(expr_str) <= 3:
            try:
                dim_num = int(expr_str[1:])
                if dim_num < len(operands):
                    return operands[dim_num]
            except (ValueError, IndexError):
                pass

        # Check if it's a constant
        try:
            const_val = int(expr_str)
            return DslxConst(const_val)
        except ValueError:
            pass

        # For common split pattern: "d0 + d1 * N"
        # Parse the string representation as a fallback
        if '+' in expr_str and '*' in expr_str:
            # Pattern: d0 + d1 * factor
            parts = expr_str.split('*')
            if len(parts) == 2:
                try:
                    factor = int(parts[-1].strip().rstrip(')'))
                    # Get the two operands (d0 and d1)
                    if len(operands) >= 2:
                        # Result: d0 + (d1 * factor)
                        mul_node = DslxBinOp("*", operands[1], DslxConst(factor))
                        return DslxBinOp("+", operands[0], mul_node)
                except:
                    pass

        # Try using expression structure if available
        if hasattr(expr, '__iter__') and not isinstance(expr, str):
            try:
                children = list(expr)
                if len(children) >= 2:
                    lhs = self._eval_affine_expr(children[0], operands)
                    rhs = self._eval_affine_expr(children[1], operands)
                    # Infer operation from string
                    if '+' in expr_str and '*' not in expr_str.split('+')[0]:
                        return DslxBinOp("+", lhs, rhs)
                    elif '*' in expr_str:
                        return DslxBinOp("*", lhs, rhs)
            except:
                pass

        # Fallback for simple identity (single operand)
        if len(operands) == 1:
            return operands[0]

        raise RuntimeError(f"Cannot evaluate affine expression: {expr_str}")

    def lower_load(self, op: affine_d.AffineLoadOp):
        base = op.memref

        buf_node = self.ctx.lookup(base)
        if not isinstance(buf_node, DslxVar):
            raise RuntimeError(f"Unknown or non-buffer memref base: {base}")

        memref_name = buf_node.name

        idx = self.lower_affine_index(op.indices)
        node = DslxLoad(memref_name, idx)
        self.ctx.bind(op.result, node)

    def lower_store(self, op: affine_d.AffineStoreOp):
        base = op.memref

        buf_node = self.ctx.lookup(base)
        if not isinstance(buf_node, DslxVar):
            raise RuntimeError(f"Unknown or non-buffer memref base in store: {base}")

        memref_name = buf_node.name

        idx = self.lower_affine_index(op.indices)
        value = self.ctx.lookup(op.value)
        store = DslxStore(memref_name, idx, value)
        self.ctx.dslx_stmts.append(store)

    def lower_fill(self, op: linalg_d.FillOp):
        val = self.ctx.lookup(op.inputs[0])
        out_base = op.outputs[0]

        buf_node = self.ctx.lookup(out_base)
        if not isinstance(buf_node, DslxVar):
            raise RuntimeError(f"Unknown output buffer in fill: {out_base}")

        memref_name = buf_node.name

        if memref_name in self.ctx.memref_shapes:
            shape = self.ctx.memref_shapes[memref_name]
            init_expr = self._create_array_init(val, shape)
            self.ctx.dslx_stmts.append(DslxLet(memref_name, init_expr))
        else:
            self.ctx.dslx_stmts.append(DslxLet(memref_name, val))

    def _create_array_init(self, val, shape):
        return DslxArrayInit(val, shape)

    def lower_for(self, op: affine_d.AffineForOp):
        def get_constant_bound(map_like):
            amap = map_like.value
            expr = amap.results[0]

            if hasattr(expr, "value"):
                return expr.value

            expr_str = str(expr).strip()

            if expr_str.isdigit():
                return int(expr_str)

            if expr_str.startswith("-") and expr_str[1:].isdigit():
                return int(expr_str)

            if isinstance(expr, int):
                return expr

            raise NotImplementedError(
                f"Unsupported affine bound expr: {expr!r} (string form: {expr_str})"
            )

        lb = get_constant_bound(op.lowerBoundMap)
        ub = get_constant_bound(op.upperBoundMap)

        try:
            loop_name_attr = op.attributes.get('loop_name')
            if loop_name_attr:
                iter_name = str(loop_name_attr).strip('"')
            else:
                depth = len(self.ctx.loop_stack)
                iter_name = chr(ord('i') + depth)
        except:
            depth = len(self.ctx.loop_stack)
            iter_name = chr(ord('i') + depth) if depth < 26 else f"i{depth}"

        self.ctx.bind(op.induction_variable, DslxVar(iter_name))
        self.ctx.loop_stack.append(iter_name)

        body_nodes = []
        old_stmts = self.ctx.dslx_stmts
        self.ctx.dslx_stmts = body_nodes

        for nested in op.body.operations:
            self.lower_op(nested)

        loop = DslxFor(iter_name, lb, ub, body_nodes)
        self.ctx.dslx_stmts = old_stmts
        self.ctx.dslx_stmts.append(loop)

        self.ctx.loop_stack.pop()

    def lower_affine_index(self, indices):
        index_nodes = [self.ctx.lookup(idx) for idx in indices]
        if len(index_nodes) == 1:
            return index_nodes[0]
        else:
            return index_nodes

    def emit_dslx(self, func_name, params, return_expr, return_type=None):
        out = ["", ""]

        param_list = ", ".join(f"{name}: {typ}" for name, typ in params)

        # Determine return type
        if return_type:
            ret_type = return_type
        elif return_expr in self.ctx.memref_shapes:
            shape = self.ctx.memref_shapes[return_expr]
            ret_type = "u32" + "".join(f"[{dim}]" for dim in shape)
        else:
            # Default to u32 for scalar
            ret_type = "u32"

        # Build function signature
        if return_type:
            sig = f"fn {func_name}({param_list}) -> {ret_type} {{"
        else:
            sig = f"fn {func_name}({param_list}) {{"

        out.append(sig)

        # Emit body statements
        for stmt in self.ctx.dslx_stmts:
            out.extend(self.emit_stmt(stmt, indent=1))

        # Emit return expression
        out.append(f"  {return_expr}")
        out.append("}")

        return "\n".join(out)

    def emit_expr(self, node):
        if isinstance(node, DslxConst):
            return f"u32:{node.value}"
        if isinstance(node, DslxVar):
            return node.name
        if isinstance(node, DslxBinOp):
            return f"({self.emit_expr(node.lhs)} {node.op} {self.emit_expr(node.rhs)})"
        if isinstance(node, DslxLoad):
            # Check if this is a scalar (no index or empty index list)
            is_scalar = (node.index_expr is None or 
                        (isinstance(node.index_expr, list) and len(node.index_expr) == 0))
            
            if is_scalar:
                # Scalar: just return the variable name
                return node.buffer_name
            elif isinstance(node.index_expr, list):
                indices = "][".join(self.emit_expr(idx) for idx in node.index_expr)
                return f"{node.buffer_name}[{indices}]"
            else:
                return f"{node.buffer_name}[{self.emit_expr(node.index_expr)}]"
        if isinstance(node, DslxArrayInit):
            if len(node.shape) == 0:
                return self.emit_expr(node.elem_expr)
            elif len(node.shape) == 1:
                # 1D array: u32[N]:[elem, elem, ...]
                elem_str = self.emit_expr(node.elem_expr)
                elements = ", ".join([elem_str] * node.shape[0])
                return f"u32[{node.shape[0]}]:[{elements}]"
            elif len(node.shape) == 2:
                # 2D array: u32[INNER][OUTER]:[[elem, ...], ...]
                elem_str = self.emit_expr(node.elem_expr)
                return f"u32[{node.shape[1]}][{node.shape[0]}]:[[{elem_str}, ...], ...]"
            else:
                return self.emit_expr(node.elem_expr)
        if isinstance(node, str):
            return node
        else:
            return f"/* Cannot emit expr for {type(node).__name__} */"

    def emit_stmt(self, stmt, indent):
        tab = "  " * indent
        if isinstance(stmt, DslxLet):
            return [f"{tab}let {stmt.name} = {self.emit_expr(stmt.expr)};"]
        if isinstance(stmt, DslxStore):
            val_expr = self.emit_expr(stmt.value_expr)
            
            # Check if this is a scalar accumulator (not in memref_shapes)
            # For scalar accumulators in for loops, we just return the value directly
            is_scalar = stmt.buffer_name not in self.ctx.memref_shapes
            
            # Check if index_expr is empty/None (scalar accumulator case)
            has_index = stmt.index_expr is not None and (
                (isinstance(stmt.index_expr, list) and len(stmt.index_expr) > 0) or
                (not isinstance(stmt.index_expr, list))
            )
            
            if is_scalar or not has_index:
                # Scalar accumulator: just return the value (for loop will handle it)
                return [f"{tab}{val_expr}"]
            
            # Array update: use update() function
            if isinstance(stmt.index_expr, list):
                if len(stmt.index_expr) == 2:
                    i_expr = self.emit_expr(stmt.index_expr[0])
                    j_expr = self.emit_expr(stmt.index_expr[1])
                    update_expr = f"update({stmt.buffer_name}, {i_expr}, update({stmt.buffer_name}[{i_expr}], {j_expr}, {val_expr}))"
                    return [f"{tab}{update_expr}"]
                else:
                    idx_expr = ", ".join(self.emit_expr(idx) for idx in stmt.index_expr)
                    update_expr = f"update({stmt.buffer_name}, {idx_expr}, {val_expr})"
                    return [f"{tab}{update_expr}"]
            else:
                idx_expr = self.emit_expr(stmt.index_expr)
                update_expr = f"update({stmt.buffer_name}, {idx_expr}, {val_expr})"
                return [f"{tab}{update_expr}"]
        if isinstance(stmt, DslxFor):
            stores = self._find_stores_in_body(stmt.body)

            if stores:
                accum_names = list(set(s.buffer_name for s in stores))
                accum_tuple = "(" + ", ".join(accum_names) + ")" if len(accum_names) > 1 else accum_names[0]

                # Determine if this is a top-level for loop (needs 'let') or nested (no 'let')
                # We check if indent > 1, which means we're inside another for loop
                is_nested = indent > 1

                if is_nested:
                    # Nested for loop: no 'let', just the for expression
                    out = [f"{tab}for ({stmt.iter_name}, {accum_tuple}) in u32:{stmt.lb}..u32:{stmt.ub} {{"]
                else:
                    # Top-level for loop: has 'let'
                    out = [f"{tab}let {accum_tuple} = for ({stmt.iter_name}, {accum_tuple}) in u32:{stmt.lb}..u32:{stmt.ub} {{"]

                # Emit body statements
                # For scalar accumulators, the body should just return the expression
                for b in stmt.body:
                    body_lines = self.emit_stmt(b, indent + 1)
                    out.extend(body_lines)

                # Determine initializer value
                # For scalar accumulators, use u32:0, for arrays use the variable name
                init_values = []
                for accum_name in accum_names:
                    if accum_name in self.ctx.memref_shapes:
                        # Array: use variable name as initializer
                        init_values.append(accum_name)
                    else:
                        # Scalar: use u32:0 as initializer
                        init_values.append("u32:0")
                
                init_str = "(" + ", ".join(init_values) + ")" if len(init_values) > 1 else init_values[0]

                # Add the accumulator initializer: }(init) or }(init);
                if is_nested:
                    out.append(f"{tab}}}({init_str})")
                else:
                    out.append(f"{tab}}}({init_str});")
            else:
                out = [f"{tab}for ({stmt.iter_name}, _) in u32:{stmt.lb}..u32:{stmt.ub} {{"]
                for b in stmt.body:
                    out.extend(self.emit_stmt(b, indent + 1))
                out.append(f"{tab}  ()")
                out.append(f"{tab}}};")
            return out
        return [f"{tab}"]

    def _find_stores_in_body(self, body):
        stores = []
        for stmt in body:
            if isinstance(stmt, DslxStore):
                stores.append(stmt)
            elif isinstance(stmt, DslxFor):
                stores.extend(self._find_stores_in_body(stmt.body))
        return stores
