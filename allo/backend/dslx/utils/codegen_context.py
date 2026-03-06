"""Code generation context for tracking variables and state during lowering."""


class CodegenContext:
    def __init__(self):
        self.var_map = {}
        self.memref_shapes = {}
        self.memref_types = {}  # Track element types (f32, i32, etc.)
        self.dslx_stmts = []
        self.loop_stack = []
        self.result_buffer = None
        self.is_float = False  # Track if we're dealing with floats

    def bind(self, mlir_value, dslx_node):
        self.var_map[mlir_value] = dslx_node

    def lookup(self, mlir_value):
        return self.var_map.get(mlir_value)
