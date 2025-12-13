import re
from typing import List, Tuple
from dataclasses import dataclass

from .xls_error_handler import (
    ValidationError,
    write_diagnostic_file,
    format_error_summary,
)


@dataclass
class AffineForBound:
    iv: str        # induction variable, e.g. "%arg2"
    lower: int     # lower bound, e.g. 0
    upper: int     # upper bound (exclusive), e.g. 32
    raw_line: str  # full line for debugging

AFFINE_FOR_RE = re.compile(
    r'^\s*affine\.for\s+(%[\w\d]+)\s*=\s*([0-9]+)\s+to\s+([0-9]+)'
)

def parse_affine_for_bounds(mlir_text: str) -> List[AffineForBound]:
    """
    Scan MLIR text and return a list of all affine.for bounds.
    e.x. 'affine.for %arg2 = 0 to 32 ...'
    """
    bounds: List[AffineForBound] = []

    for line in mlir_text.splitlines():
        m = AFFINE_FOR_RE.match(line)
        if not m:
            continue
        iv, lo_str, hi_str = m.groups()
        bounds.append(
            AffineForBound(
                iv=iv,
                lower=int(lo_str),
                upper=int(hi_str),
                raw_line=line,
            )
        )
    return bounds

def get_loop_extents(mlir_text: str) -> List[int]:
    """
    Return a list of extents (upper - lower) for each affine.for in module
    """
    bounds = parse_affine_for_bounds(mlir_text)
    return [b.upper - b.lower for b in bounds]

def extract_function_body(core_code: str, func_name: str) -> str:
    """
    Extract the body of a void function from the generated C++ code.
    Returns the content inside the braces.
    """
    # Find the function: void func_name() { ... }
    # We need to match balanced braces
    pattern = rf'void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{'
    match = re.search(pattern, core_code)
    if not match:
        return f"// Could not extract body of {func_name}\n"
    
    start = match.end()  # Position after opening brace
    brace_count = 1
    pos = start
    
    while pos < len(core_code) and brace_count > 0:
        if core_code[pos] == '{':
            brace_count += 1
        elif core_code[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        return f"// Could not find matching brace for {func_name}\n"
    
    # Extract body (excluding final closing brace)
    body = core_code[start:pos-1]
    return body


def reindent_body(body: str, base_indent: str = "    ") -> str:
    """
    Re-indent the function body to align with the class method.
    Preserves relative indentation structure (nested blocks stay nested).
    base_indent: 4 spaces (2 for class + 2 for method body)
    """
    lines = body.split('\n')
    
    # Find minimum indentation (excluding empty lines)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # non-empty line
            # Count leading spaces
            leading = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading)
    
    if min_indent == float('inf'):
        min_indent = 0
    
    # Re-indent: remove min_indent and add base_indent
    result = []
    for line in lines:
        if not line.strip():
            result.append("")
        else:
            # Remove the common minimum indentation, then add our base indent
            dedented = line[min_indent:] if len(line) >= min_indent else line.lstrip()
            result.append(base_indent + dedented)
    
    return '\n'.join(result)


def render_testblock_with_body(input_channels: List[str],
                                output_channels: List[str],
                                function_body: str,
                                top_name: str) -> str:
    """
    Render a TestBlock class with the function body embedded directly as the top method.
    """
    # Format channel declarations using direct XLS types (2-space indentation)
    input_decls = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_In> {name};" for name in input_channels
    )
    output_decls = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_Out> {name};" for name in output_channels
    )
    
    # Re-indent the function body to align properly inside the class method
    indented_body = reindent_body(function_body)
    
    # Build the class with the function body embedded directly (2-space indentation)
    testblock = f"""class TestBlock {{
public:
  {input_decls}
  {output_decls}

  #pragma hls_top
  void {top_name}() {{
{indented_body}
  }}
}};
"""
    return testblock


def parse_memory_declarations(function_body: str) -> Tuple[List[str], List[str], str]:
    """
    Parse __xls_memory_decl__ and __xls_state_vars__ comment placeholders from function body.
    Returns (memory_declarations, state_vars, cleaned_body) where:
    - memory_declarations: list of "__xls_memory<type, size> name;" strings
    - state_vars: list of state variable declarations ("int state = 0;", "int i = 0;")
    - cleaned_body: function body with comment placeholders removed
    """
    memory_decls = []
    state_vars = []
    cleaned_lines = []
    
    # Pattern: // __xls_memory_decl__: type, size, name, dim1, dim2, ...
    memory_pattern = re.compile(r'^\s*//\s*__xls_memory_decl__:\s*(.+)$')
    # Pattern: // __xls_state_vars__: var1, var2, ...
    state_pattern = re.compile(r'^\s*//\s*__xls_state_vars__:\s*(.+)$')
    
    for line in function_body.split('\n'):
        mem_match = memory_pattern.match(line)
        state_match = state_pattern.match(line)
        
        if mem_match:
            # Parse: type, size, name, dims...
            parts = [p.strip() for p in mem_match.group(1).split(',')]
            if len(parts) >= 3:
                elem_type = parts[0]
                total_size = parts[1]
                var_name = parts[2]
                # Original dims for comment
                orig_dims = parts[3:] if len(parts) > 3 else []
                
                # Build the __xls_memory declaration
                decl = f"__xls_memory<{elem_type}, {total_size}> {var_name};"
                if orig_dims:
                    orig_shape = ''.join(f'[{d}]' for d in orig_dims)
                    decl += f"  // original: {elem_type} {var_name}{orig_shape}"
                memory_decls.append(decl)
        elif state_match:
            # Parse: var1, var2, ...
            var_names = [v.strip() for v in state_match.group(1).split(',')]
            for var_name in var_names:
                state_vars.append(f"int {var_name} = 0;")
        else:
            cleaned_lines.append(line)
    
    return memory_decls, state_vars, '\n'.join(cleaned_lines)


@dataclass
class MemoryInfo:
    """Information about a memory declaration for textproto generation."""
    name: str           # e.g., "v0"
    elem_type: str      # e.g., "int"
    total_size: int     # e.g., 16
    dims: List[int]     # e.g., [16] or [4, 4]


def parse_memory_info(function_body: str) -> List[MemoryInfo]:
    """
    Parse __xls_memory_decl__ comments and return structured memory info.
    Used for generating RAM rewrites textproto.
    """
    memories = []
    memory_pattern = re.compile(r'^\s*//\s*__xls_memory_decl__:\s*(.+)$')
    
    for line in function_body.split('\n'):
        match = memory_pattern.match(line)
        if match:
            parts = [p.strip() for p in match.group(1).split(',')]
            if len(parts) >= 3:
                elem_type = parts[0]
                total_size = int(parts[1])
                var_name = parts[2]
                dims = [int(d) for d in parts[3:]] if len(parts) > 3 else [total_size]
                memories.append(MemoryInfo(
                    name=var_name,
                    elem_type=elem_type,
                    total_size=total_size,
                    dims=dims
                ))
    return memories


def generate_ram_rewrites_textproto(memories: List[MemoryInfo], 
                                     class_name: str = "TestBlock",
                                     top_func: str = "vvadd") -> str:
    """
    Generate XLS RAM rewrites textproto for the given memories.
    
    This textproto tells XLS opt_main how to map __xls_memory channels
    to actual RAM configurations for synthesis.
    
    Args:
        memories: List of MemoryInfo from parse_memory_info
        class_name: The class name (e.g., "TestBlock")
        top_func: The top function name
        
    Returns:
        Textproto string for --ram_rewrites_pb flag
    """
    if not memories:
        return "# No memories to rewrite\n"
    
    lines = [
        "# XLS RAM Rewrites Configuration",
        "# Generated by Allo XLS backend",
        "#",
        "# Usage:",
        "#   1. Compile: xlscc design.cc --block_from_class TestBlock --block_pb block.pb > design.ir",
        "#   2. Check channels: grep 'chan ' design.ir",
        "#   3. Optimize: opt_main design.ir --ram_rewrites_pb=this_file.textproto > design.opt.ir",
        "#",
        "# If you get 'No channel with name' errors, run 'grep chan design.ir' to find",
        "# the actual channel names and update from_name_prefix accordingly.",
        ""
    ]
    
    for mem in memories:
        # Calculate word size in bits (assuming int = 32 bits)
        word_bits = 32
        if mem.elem_type in ["short", "int16_t", "uint16_t"]:
            word_bits = 16
        elif mem.elem_type in ["long long", "int64_t", "uint64_t"]:
            word_bits = 64
        elif mem.elem_type in ["char", "int8_t", "uint8_t"]:
            word_bits = 8
        
        # XLS generates memory channels with format: {class}__{top_func}__{memory_name}
        # or sometimes just {class}__{memory_name} depending on XLS version
        # The actual format can be found by: grep "chan " design.ir
        channel_prefix = f"{class_name}__{top_func}__{mem.name}"
        
        lines.append(f"# Memory: {mem.name} ({mem.elem_type}[{mem.total_size}])")
        lines.append("ram_rewrites {")
        lines.append(f'  from_channels_logical_name: "{mem.name}"')
        lines.append(f'  from_name_prefix: "{channel_prefix}"')
        lines.append(f'  to_name_prefix: "{channel_prefix}_ram"')
        lines.append("  from_config {")
        lines.append("    kind: RAM_ABSTRACT")
        lines.append(f"    depth: {mem.total_size}")
        lines.append(f"    word_partition_size: {word_bits}")
        lines.append("  }")
        lines.append("  to_config {")
        lines.append("    kind: RAM_1RW")
        lines.append(f"    depth: {mem.total_size}")
        lines.append(f"    word_partition_size: {word_bits}")
        lines.append("  }")
        lines.append("}")
        lines.append("")
    
    return "\n".join(lines)


def render_testblock_with_memory(input_channels: List[str],
                                  output_channels: List[str],
                                  function_body: str,
                                  top_name: str) -> str:
    """
    Render a TestBlock class with __xls_memory for arrays (memory mode).
    In memory mode:
    - __xls_memory declarations are placed at class level (after channels)
    - State variables (state, i) are placed at class level for state machine
    - Function body uses state-based control flow instead of for loops
    """
    # Parse and extract __xls_memory declarations and state variables from function body
    memory_decls, state_vars, cleaned_body = parse_memory_declarations(function_body)
    
    # Format channel declarations using direct XLS types (2-space indentation)
    input_decls = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_In> {name};" for name in input_channels
    )
    output_decls = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_Out> {name};" for name in output_channels
    )
    
    # Format memory declarations at class level
    memory_decls_str = "\n  ".join(memory_decls) if memory_decls else ""
    
    # Format state variable declarations at class level
    state_vars_str = "\n  ".join(state_vars) if state_vars else ""
    
    # Re-indent the cleaned function body
    indented_body = reindent_body(cleaned_body)
    
    # Build the class with memory and state declarations at class level
    parts = []
    parts.append("class TestBlock {\npublic:\n")
    
    # Channels section
    parts.append("  // Channels\n")
    parts.append(f"  {input_decls}\n")
    parts.append(f"  {output_decls}\n")
    
    # Memory section (if any)
    if memory_decls_str:
        parts.append("\n  // Memory (SRAM/BRAM)\n")
        parts.append(f"  {memory_decls_str}\n")
    
    # State variables section (if any)
    if state_vars_str:
        parts.append("\n  // State machine variables\n")
        parts.append(f"  {state_vars_str}\n")
    
    # Function
    parts.append(f"\n  #pragma hls_top\n")
    parts.append(f"  void {top_name}() {{\n")
    parts.append(f"{indented_body}\n")
    parts.append("  }\n")
    parts.append("};\n")
    
    return "".join(parts)


# XLS legality checks on MLIR text

DYNAMIC_TYPE_RE = re.compile(
    r'\b(memref|tensor)<[^>]*\?>'
)

LOOP_HEADER_RE = re.compile(
    r'^\s*(affine\.for|scf\.for)\b.*$'
)

# Patterns for unsupported types
FLOAT_TYPE_RE = re.compile(r'\b(f16|f32|f64|bf16)\b')
FIXED_TYPE_RE = re.compile(r'\ballo\.(fixed|ufixed)<[^>]+>')

# Unsupported Allo pragmas/ops (XLS only supports pipeline and unroll)
UNSUPPORTED_PRAGMAS = {
    'allo.partition': 'Array partitioning is not supported in XLS. Use register mode (use_memory=False) for automatic register allocation.',
    'allo.parallel': 'Parallel loops are not supported in XLS. Use pipeline or unroll instead.',
    'allo.bind': 'Thread binding is not supported in XLS (no GPU-style parallelism).',
    'allo.dataflow': 'Dataflow regions are not supported in XLS. Consider restructuring as sequential pipelined loops.',
    'allo.buffer_at': 'Buffer insertion is not supported in XLS. Arrays are either registers or __xls_memory.',
    'allo.reuse_at': 'Reuse buffers are not supported in XLS. Manual buffering required.',
    'allo.reshape': 'Reshape operations are not supported in XLS. Use flat arrays instead.',
    'allo.compute_at': 'Compute-at scheduling is not supported in XLS.',
    'allo.unfold': 'Unfold (spatial PE arrays) is not supported in XLS.',
    'allo.fuse': 'Loop fusion is not supported in XLS. Manually fuse loops in source.',
    'allo.tile': 'Loop tiling is not supported in XLS. Manually tile loops in source.',
    'allo.split': 'Loop splitting is not supported in XLS. Manually split loops in source.',
    'allo.reorder': 'Loop reordering is not supported in XLS. Manually reorder loops in source.',
    'allo.inter_kernel_to': 'Inter-kernel streaming is not supported in XLS.',
}


def validate_xls_ir(mlir_text: str, project: str = None) -> None:
    """
    Validates MLIR text for XLS[cc] backend compatibility.
    
    Raises RuntimeError with a human-readable message if validation fails.
    If project is provided, writes a diagnostic file showing annotated MLIR.
    
    Checks for:
    - Floating-point types (f16, f32, f64, bf16) - not supported
    - Fixed-point types (allo.fixed, allo.ufixed) - not supported  
    - Dynamic memref/tensor shapes (no '?' in type)
    - Dynamic allocation (memref.alloc, memref.alloca)
    - Unsupported Allo pragmas (partition, parallel, dataflow, etc.)
    """
    errors: List[ValidationError] = []
    lines = mlir_text.splitlines()
    
    for line_num, line in enumerate(lines, start=1):
        # Check for floating-point types
        float_match = FLOAT_TYPE_RE.search(line)
        if float_match:
            errors.append(ValidationError(
                line_num=line_num,
                line_content=line,
                error_type='FLOAT_TYPE',
                message=f"Floating-point type '{float_match.group()}' is not supported. "
                        f"XLS requires integer types. Consider using fixed-point emulation with integers."
            ))
        
        # Check for fixed-point types
        fixed_match = FIXED_TYPE_RE.search(line)
        if fixed_match:
            errors.append(ValidationError(
                line_num=line_num,
                line_content=line,
                error_type='FIXED_TYPE',
                message=f"Fixed-point type '{fixed_match.group()}' is not supported. "
                        f"Use integer types with manual scaling instead."
            ))
        
        # Check for dynamic shapes
        if DYNAMIC_TYPE_RE.search(line):
            errors.append(ValidationError(
                line_num=line_num,
                line_content=line,
                error_type='DYNAMIC_SHAPE',
                message="Dynamic shapes ('?') are not supported. All dimensions must be static constants."
            ))
        
        # Check for dynamic allocation
        if "memref.alloca" in line:
            errors.append(ValidationError(
                line_num=line_num,
                line_content=line,
                error_type='DYNAMIC_ALLOC',
                message="memref.alloca (stack allocation) is not supported. Use static arrays."
            ))
        if "memref.alloc " in line or line.strip().endswith("memref.alloc"):
            errors.append(ValidationError(
                line_num=line_num,
                line_content=line,
                error_type='DYNAMIC_ALLOC',
                message="memref.alloc (heap allocation) is not supported. Use static arrays."
            ))
        
        # Check for unsupported pragmas
        for pragma, help_msg in UNSUPPORTED_PRAGMAS.items():
            if pragma in line:
                errors.append(ValidationError(
                    line_num=line_num,
                    line_content=line,
                    error_type='UNSUPPORTED_PRAGMA',
                    message=f"'{pragma}' is not supported. {help_msg}"
                ))
    
    if not errors:
        return  # Validation passed
    
    # Write diagnostic file if project path provided
    if project:
        write_diagnostic_file(errors, mlir_text, project)
    
    # Raise exception with summary
    raise RuntimeError(format_error_summary(errors, project))

def render_combinational_with_channels(core_code: str,
                                        top_name: str,
                                        generated_func_name: str,
                                        scalar_inputs: List[str],
                                        scalar_output: str = "result") -> str:
    """
    Render a TestBlock class for combinational logic with channels.
    The combinational function is called once per invocation, reading from input channels
    and writing to output channel.
    
    scalar_inputs: List of scalar input names (e.g., ["a", "b"])
    scalar_output: Name for the output channel
    """
    # Extract headers from core_code (everything before the function)
    func_pattern = rf'void\s+{re.escape(generated_func_name)}\s*\([^)]*\)\s*\{{'
    match = re.search(func_pattern, core_code)
    if not match:
        # Try matching return type function
        func_pattern = rf'\w+\s+{re.escape(generated_func_name)}\s*\([^)]*\)\s*\{{'
        match = re.search(func_pattern, core_code)
    headers = core_code[:match.start()] if match else ""
    
    # Build channel declarations
    input_channel_decls = "\n  ".join(
        f"__xls_channel<int, __xls_channel_dir_In> {name}_in;" for name in scalar_inputs
    )
    output_channel_decl = f"__xls_channel<int, __xls_channel_dir_Out> {scalar_output}_out;"
    
    # Build read statements
    read_stmts = "\n    ".join(
        f"int {name} = {name}_in.read();" for name in scalar_inputs
    )
    
    # Build function call
    args = ", ".join(scalar_inputs)
    
    testblock = f"""{headers}
// ---- TestBlock with channels (combinational mode) ----
class TestBlock {{
public:
  // Input channels
  {input_channel_decls}
  // Output channel
  {output_channel_decl}

  #pragma hls_top
  void {top_name}() {{
    // Read inputs from channels
    {read_stmts}
    // Compute
    int result = {generated_func_name}({args});
    // Write output to channel
    {scalar_output}_out.write(result);
  }}
}};
"""
    return testblock


def wrap_xlscc(mlir_module_text: str,
               core_code: str,
               function_names: Tuple[str, str],
               function_inputs: List[str],
               use_memory: bool = False,
               use_channels: bool = False) -> Tuple[str, str]:
    """
    mlir_module_text: MLIR (string) with affine.for loops.
    core_code:        C++ code produced by emitXlscc (contains the generated function).
    function_names:   (top_name, generated_func_name) -> we extract body from generated_func_name
    function_inputs:  List of input argument names (used to generate channel names)
    use_memory:       If True, arrays are emitted as __xls_memory<T, size> (for SRAM/BRAM).
                      If False (default), arrays are plain C arrays (for registers).
    use_channels:     If True, combinational functions are wrapped in TestBlock with channels.
                      If False (default), combinational functions are emitted as plain functions.
    Returns:          Tuple of (cpp_code, textproto) where:
                      - cpp_code: Full C++ code
                      - textproto: RAM rewrites textproto (empty string if not memory mode)
    """
    top_name, generated_func_name = function_names
    
    # COMBINATIONAL MODE: No array inputs
    if len(function_inputs) == 0:
        if use_channels:
            # Combinational with channels: wrap in TestBlock with scalar channels
            # Extract scalar input names from the function signature
            # For now, use generic names - caller can provide specific names
            cpp_code = render_combinational_with_channels(
                core_code=core_code,
                top_name=top_name,
                generated_func_name=generated_func_name,
                scalar_inputs=["a", "b"],  # Default scalar inputs
                scalar_output="result"
            )
            return cpp_code, ""
        else:
            # Combinational without channels: just return core code as-is
            # The #pragma hls_top is already added by EmitXlsHLS.cpp
            return core_code, ""
    
    # SEQUENTIAL MODE: Has array inputs -> wrap in TestBlock with channels
    # Extract the function body from the generated code
    func_body = extract_function_body(core_code, generated_func_name)
    
    # Generate channel names: v0_in, v1_in, ... for input arrays
    n_inputs = len(function_inputs)
    input_channels = [f"v{i}_in" for i in range(n_inputs)]
    output_channels = ["out"]
    
    # Extract headers from core_code (everything before the function definition)
    func_pattern = rf'void\s+{re.escape(generated_func_name)}\s*\(\s*\)\s*\{{'
    match = re.search(func_pattern, core_code)
    headers = core_code[:match.start()] if match else ""
    
    # Parse memory info for textproto generation
    memories = parse_memory_info(func_body) if use_memory else []
    
    # Build the final code: headers + TestBlock class with embedded function body
    parts: List[str] = []
    parts.append(headers)
    
    if use_memory:
        parts.append("\n// ---- TestBlock with __xls_memory (SRAM/BRAM mode) ----\n")
        testblock = render_testblock_with_memory(
            input_channels=input_channels,
            output_channels=output_channels,
            function_body=func_body,
            top_name=top_name,
        )
    else:
        parts.append("\n// ---- TestBlock with embedded function (register mode) ----\n")
        testblock = render_testblock_with_body(
            input_channels=input_channels,
            output_channels=output_channels,
            function_body=func_body,
            top_name=top_name,
        )
    parts.append(testblock)
    
    cpp_code = "".join(parts)
    
    # Generate textproto for memory mode
    if use_memory and memories:
        textproto = generate_ram_rewrites_textproto(
            memories=memories,
            class_name="TestBlock",
            top_func=top_name
        )
    else:
        textproto = ""
    
    return cpp_code, textproto



if __name__ == "__main__":
    # Example core C++ code that emitXlscc might return for array function (REGISTER MODE)
    core_code_register = """
// Headers...
#include <cstdint>

void vvadd() {
    int v0[16];
    int v1[16];
    
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        v0[i] = v0_in.read();
    }
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        v1[i] = v1_in.read();
    }
    
    int result[16];
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        result[i] = v0[i] + v1[i];
    }
    
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        out.write(result[i]);
    }
}
    """

    # Example core C++ code for MEMORY MODE (__xls_memory)
    # Note: Uses __xls_memory_decl__ comments for the wrapper to parse
    core_code_memory = """
// Headers...
#include <cstdint>

void vvadd() {
    // __xls_memory_decl__: int, 16, v0, 16
    // __xls_memory_decl__: int, 16, v1, 16
    // __xls_memory_decl__: int, 16, result, 16
    
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        v0[i] = v0_in.read();
    }
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        v1[i] = v1_in.read();
    }
    
    #pragma hls_unroll yes
    for (int i = 0; i < 16; ++i) {
        result[i] = v0[i] + v1[i];
    }
    
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < 16; ++i) {
        out.write(result[i]);
    }
}
    """

    # (top_name, generated_function_name)
    function_names = ("Run", "vvadd")
    function_inputs = ["v0", "v1"]

    print("=" * 60)
    print("REGISTER MODE (use_memory=False):")
    print("=" * 60)
    wrapped_cpp_register, textproto_register = wrap_xlscc(
        mlir_module_text="",
        core_code=core_code_register,
        function_names=function_names,
        function_inputs=function_inputs,
        use_memory=False,
    )
    print(wrapped_cpp_register)

    print("\n" + "=" * 60)
    print("MEMORY MODE (use_memory=True):")
    print("=" * 60)
    wrapped_cpp_memory, textproto_memory = wrap_xlscc(
        mlir_module_text="",
        core_code=core_code_memory,
        function_names=function_names,
        function_inputs=function_inputs,
        use_memory=True,
    )
    print(wrapped_cpp_memory)
    
    # Print the RAM rewrites textproto for memory mode
    if textproto_memory:
        print("\n" + "=" * 60)
        print("RAM REWRITES TEXTPROTO (for use with opt_main):")
        print("=" * 60)
        print(textproto_memory)
        print("\nUsage:")
        print("  1. Save the C++ code above to 'vvadd_mem.cc'")
        print("  2. Save the textproto above to 'rewrites.textproto'")
        print("  3. Compile: ~/xls/bazel-bin/xls/contrib/xlscc/xlscc vvadd_mem.cc > vvadd_mem.ir")
        print("  4. Optimize: ~/xls/bazel-bin/xls/tools/opt_main vvadd_mem.ir --ram_rewrites_pb=rewrites.textproto > vvadd_mem.opt.ir")