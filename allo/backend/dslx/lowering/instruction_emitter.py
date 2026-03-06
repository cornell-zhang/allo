"""DSLX instruction emitter for MLIR operations.

Emits DSLX code for individual MLIR arithmetic and control operations.
"""

from allo._mlir.ir import Operation, IndexType, F16Type, F32Type, F64Type, BF16Type
from allo._mlir.dialects import arith as arith_d
from allo._mlir.dialects import func as func_d
from allo._mlir.dialects import scf as scf_d
from allo._mlir.dialects import memref as memref_d
from allo._mlir.dialects import affine as affine_d

from ..utils.type_utils import (
    allo_dtype_to_dslx_type,
    float_to_dslx_literal,
    float_dtype_to_dslx,
)

# MLIR float type to dtype string mapping
MLIR_FLOAT_MAP = {
    F16Type: "f16",
    F32Type: "f32",
    F64Type: "f64",
    BF16Type: "bf16",
}


def get_float_dtype(mlir_type):
    """Check if MLIR type is a float type and return dtype string."""
    for ftype, dtype in MLIR_FLOAT_MAP.items():
        if ftype.isinstance(mlir_type):
            return dtype
    return None


class InstructionEmitter:
    """Emits DSLX code for MLIR arithmetic and control operations."""

    def __init__(self, context):
        """Initialize with a lowering context that provides lookup/register methods.
        
        Args:
            context: Object with lookup(value), register(value, name), new_tmp() methods
        """
        self.ctx = context
        self.float_types_used = set()

    def dslx_type(self, op) -> str:
        """Convert MLIR type to DSLX type string."""
        result_type = op.result.type
        if isinstance(result_type, IndexType):
            return "s32"
        
        # Handle float types
        dtype = get_float_dtype(result_type)
        if dtype:
            self.float_types_used.add(dtype)
            return float_dtype_to_dslx(dtype)
        
        # Handle integer types
        # Check for unsigned attribute (handle both dict and list-style attributes)
        is_unsigned = False
        if isinstance(op.attributes, dict):
            is_unsigned = "unsigned" in op.attributes
        else:
            is_unsigned = any(
                (hasattr(a, "name") and a.name == "unsigned") or a == "unsigned"
                for a in op.attributes
            )
        sgn = "u" if is_unsigned else "s"
        bw = result_type.width
        return f"{sgn}{bw}" if (bw <= 64) else f"{sgn}N[{bw}]"

    def emit(self, op: Operation) -> list[str]:
        """Emit DSLX code for an MLIR operation.
        
        Returns:
            List of DSLX code lines
        """
        # Constant
        if isinstance(op, arith_d.ConstantOp):
            return self._emit_constant(op)
        
        # Integer arithmetic
        elif isinstance(op, arith_d.AddIOp):
            return self._emit_binary(op, "+")
        elif isinstance(op, arith_d.SubIOp):
            return self._emit_binary(op, "-")
        elif isinstance(op, arith_d.MulIOp):
            return self._emit_binary(op, "*")
        elif isinstance(op, arith_d.RemSIOp):
            return self._emit_binary(op, "%")
        elif isinstance(op, arith_d.FloorDivSIOp):
            return self._emit_binary(op, "/")
        elif isinstance(op, arith_d.DivSIOp):
            return self._emit_binary(op, "/")
        elif isinstance(op, arith_d.DivUIOp):
            return self._emit_binary(op, "/")
        elif isinstance(op, arith_d.RemUIOp):
            return self._emit_binary(op, "%")
        
        # Bitwise operations
        elif isinstance(op, arith_d.OrIOp):
            return self._emit_binary(op, "|")
        elif isinstance(op, arith_d.AndIOp):
            return self._emit_binary(op, "&")
        elif isinstance(op, arith_d.XOrIOp):
            return self._emit_binary(op, "^")
        elif isinstance(op, arith_d.ShLIOp):
            return self._emit_binary(op, "<<")
        elif isinstance(op, arith_d.ShRSIOp):
            return self._emit_binary(op, ">>")  # arithmetic shift
        elif isinstance(op, arith_d.ShRUIOp):
            return self._emit_binary(op, ">>")  # logical shift
        
        # Comparison
        elif isinstance(op, arith_d.CmpIOp):
            return self._emit_cmpi(op)
        
        # Float arithmetic
        elif isinstance(op, arith_d.AddFOp):
            return self._emit_float_binary(op, "add")
        elif isinstance(op, arith_d.SubFOp):
            return self._emit_float_binary(op, "sub")
        elif isinstance(op, arith_d.MulFOp):
            return self._emit_float_binary(op, "mul")
        elif isinstance(op, arith_d.DivFOp):
            return self._emit_float_binary(op, "div")
        elif isinstance(op, arith_d.NegFOp):
            return self._emit_negf(op)
        elif isinstance(op, arith_d.CmpFOp):
            return self._emit_cmpf(op)
        
        # Casts and conversions
        elif isinstance(op, arith_d.SelectOp):
            return self._emit_select(op)
        elif isinstance(op, arith_d.ExtUIOp):
            return self._emit_cast(op)
        elif isinstance(op, arith_d.ExtSIOp):
            return self._emit_cast(op)
        elif isinstance(op, arith_d.TruncIOp):
            return self._emit_cast(op)
        elif isinstance(op, arith_d.IndexCastOp):
            return self._emit_cast(op)
        elif isinstance(op, arith_d.SIToFPOp):
            return self._emit_sitofp(op)
        elif isinstance(op, arith_d.FPToSIOp):
            return self._emit_fptosi(op)
        
        # Control flow
        elif isinstance(op, scf_d.YieldOp):
            return []
        elif isinstance(op, func_d.ReturnOp):
            return self._emit_return(op)
        
        # Memory operations (basic)
        elif isinstance(op, memref_d.AllocOp):
            return []
        elif isinstance(op, affine_d.AffineStoreOp):
            return self._emit_affine_store(op)
        elif isinstance(op, affine_d.AffineLoadOp):
            return self._emit_affine_load(op)
        elif isinstance(op, affine_d.AffineYieldOp):
            return []
        
        raise NotImplementedError(f"not implemented: {repr(op)}")

    def _emit_binary(self, op, opcode: str) -> list[str]:
        """Emit binary operation with given opcode."""
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        return [f"    let {tmp} = ({lhs} {opcode} {rhs});"]

    def _emit_cast(self, op) -> list[str]:
        """Emit precision cast operation."""
        src = self.ctx.lookup(op.operands[0])
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        return [f"    let {tmp} = ({src} as {self.dslx_type(op)});"]

    def _emit_constant(self, op: arith_d.ConstantOp) -> list[str]:
        """Emit constant operation."""
        result_type = op.result.type
        
        # Handle float constants
        dtype = get_float_dtype(result_type)
        if dtype:
            self.float_types_used.add(dtype)
            value_str = str(op.value).split(":")[0].strip()
            self.ctx.constant(op.result, float_to_dslx_literal(float(value_str), dtype))
            return []
        
        # Handle integer constants
        value = str(op.value).split(":", 1)[0].strip()
        dslx_prefix = allo_dtype_to_dslx_type(str(op.result.type))
        self.ctx.constant(op.result, f"{dslx_prefix}:{value}")
        return []

    def _emit_cmpi(self, op: arith_d.CmpIOp) -> list[str]:
        """Emit integer comparison."""
        opcodes = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="]
        opcode = opcodes[op.predicate.value]
        return self._emit_binary(op, opcode)

    def _emit_float_binary(self, op, func_name: str) -> list[str]:
        """Emit float binary operation using apfloat function."""
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        dtype = get_float_dtype(op.result.type)
        if dtype:
            self.float_types_used.add(dtype)
        return [f"    let {tmp} = apfloat::{func_name}({lhs}, {rhs});"]

    def _emit_negf(self, op: arith_d.NegFOp) -> list[str]:
        """Emit float negation."""
        src = self.ctx.lookup(op.operands[0])
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        dtype = get_float_dtype(op.result.type)
        if dtype:
            self.float_types_used.add(dtype)
        return [f"    let {tmp} = apfloat::negate({src});"]

    def _emit_cmpf(self, op: arith_d.CmpFOp) -> list[str]:
        """Emit float comparison."""
        lhs = self.ctx.lookup(op.lhs)
        rhs = self.ctx.lookup(op.rhs)
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        dtype = get_float_dtype(op.lhs.type)
        if dtype:
            self.float_types_used.add(dtype)
        
        # cmpf predicates: 0=false, 1=oeq, 2=ogt, 3=oge, 4=olt, 5=ole, 6=one, 7=ord, etc.
        cmp_funcs = {1: "eq_2", 2: "gt_2", 3: "gte_2", 4: "lt_2", 5: "lte_2"}
        if op.predicate.value in cmp_funcs:
            return [f"    let {tmp} = apfloat::{cmp_funcs[op.predicate.value]}({lhs}, {rhs});"]
        raise NotImplementedError(f"float comparison predicate {op.predicate.value} not supported")

    def _emit_select(self, op: arith_d.SelectOp) -> list[str]:
        """Emit select operation."""
        sel = self.ctx.lookup(op.operands[0])
        lhs = self.ctx.lookup(op.operands[1])
        rhs = self.ctx.lookup(op.operands[2])
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        return [f"    let {tmp} = if ({sel}) {{ {lhs} }} else {{ {rhs} }};"]

    def _emit_sitofp(self, op: arith_d.SIToFPOp) -> list[str]:
        """Emit signed int to float conversion."""
        src = self.ctx.lookup(op.operands[0])
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        dtype = get_float_dtype(op.result.type)
        if dtype:
            self.float_types_used.add(dtype)
            dslx_name = float_dtype_to_dslx(dtype)
            return [f"    let {tmp} = apfloat::from_signed({src}, {dslx_name}_EXP_SZ, {dslx_name}_FRAC_SZ);"]
        return [f"    let {tmp} = ({src} as {self.dslx_type(op)});"]

    def _emit_fptosi(self, op: arith_d.FPToSIOp) -> list[str]:
        """Emit float to signed int conversion."""
        src = self.ctx.lookup(op.operands[0])
        tmp = self.ctx.new_tmp()
        self.ctx.register(op.result, tmp)
        dtype = get_float_dtype(op.operands[0].type)
        if dtype:
            self.float_types_used.add(dtype)
        return [f"    let {tmp} = apfloat::to_signed({src});"]

    def _emit_affine_store(self, op: affine_d.AffineStoreOp) -> list[str]:
        """Register memref with stored value for later loads (scalar only)."""
        stored_val = self.ctx.lookup(op.operands[0])
        memref = op.operands[1]
        self.ctx.register(memref, stored_val)
        return []

    def _emit_affine_load(self, op: affine_d.AffineLoadOp) -> list[str]:
        """Load value from memref by looking up registered value (scalar only)."""
        memref = op.operands[0]
        stored_val = self.ctx.lookup(memref)
        self.ctx.register(op.result, stored_val)
        return []

    def _emit_return(self, op: func_d.ReturnOp) -> list[str]:
        """Emit return operation with channel sends."""
        # For proc-based lowering, return sends to output channels
        if not hasattr(self.ctx, "outputs") or not hasattr(self.ctx, "tok_counter"):
            return []
        
        if self.ctx.tok_counter > 1:
            lines = [f"    let tok = join({', '.join([f'tok{i}' for i in range(self.ctx.tok_counter)])});"]
            token = "tok"
        else:
            lines = []
            token = "tok0"
        
        for idx, operand in enumerate(op.operands):
            src = self.ctx.lookup(operand)
            out_chan = self.ctx.outputs[idx]
            lines.append(f"    send({token}, {out_chan}, {src});")
        return lines
