import re
from typing import List, Tuple

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

def render_testblock(input: List[str],
                     function_names: Tuple[str, str],
                     combinational: bool = True,
                     memory: List[Tuple[int, str]] = [],
                     output: List[str] = ["out"]) -> str:

    # --- Channels ---
    input_channels = "\n    ".join(
        f"InputChannel<int> {name};" for name in input
    )
    output_channels = "\n    ".join(
        f"OutputChannel<int> {name};" for name in output
    )
    memory_blocks = "\n    ".join(
        f"Memory<int, {size}> {name};" for size, name in memory
    )

    if combinational:
        # Combinational block: return type is int
        run_body = f"""
    #pragma hls_top
    int {function_names[0]}({", ".join(f"int {param}" for param in input)}) {{
        return {function_names[1]}({", ".join(f"{param}" for param in input)});
    }}
        """
    else:
        # Sequential block: return type is void
        run_body = f"""
    {input_channels}
    {output_channels}
    {memory_blocks}
        
    #pragma hls_top
    void {function_names[0]}() {{
        // TODO: Fill in sequential FSM logic
    }}
    """

    # --- Build class ---
    testblock = f"""class TestBlock {{
public:
    {run_body}
}};
"""

    return testblock

# XLS legality checks on MLIR text

DYNAMIC_TYPE_RE = re.compile(
    r'\b(memref|tensor)<[^>]*\?>'
)

LOOP_HEADER_RE = re.compile(
    r'^\s*(affine\.for|scf\.for)\b.*$'
)

def validate_xls_ir(mlir_text: str) -> None:
    """
    Raises RuntimeError with a human-readable message if not.
    Cannot have
    - Dynamic memref/tensor shapes (no '?' in type).
    - Memref.alloc or memref.alloca (dynamic allocation).
    - Every affine.for / scf.for must be annotated with allo.pipeline
      or allo.unroll in its header line.
    """
    errors = []

    # Dynamic shapes: look for '?' inside memref<...> or tensor<...>.
    if DYNAMIC_TYPE_RE.search(mlir_text):
        errors.append(
            "XLS backend requires static shapes: found memref/tensor type "
            "with dynamic dimension ('?')."
        )

    # Disallow dynamic allocation ops.
    if "memref.alloca" in mlir_text:
        errors.append(
            "memref.alloca is not supported by the XLS backend (no dynamic stack "
            "allocation)."
        )
    if "memref.alloc " in mlir_text or "memref.alloc\n" in mlir_text:
        errors.append(
            "memref.alloc is not supported by the XLS backend (no dynamic heap "
            "allocation). Lower to static buffers or globals instead."
        )

    # For each affine.for/scf.for, require allo.pipeline or allo.unroll
    for line in mlir_text.splitlines():
        m = LOOP_HEADER_RE.match(line)
        if not m:
            continue

        if "allo.pipeline" not in line and "allo.unroll" not in line:
            errors.append(
                "Loop missing XLS directive: each affine.for/scf.for must have "
                "either 'allo.pipeline' or 'allo.unroll' attribute.\n"
                f"  Offending loop header: {line.strip()}"
            )

    if errors:
        raise RuntimeError(
            "MLIR module is not legal for XLS[cc] backend:\n"
            + "\n".join(f"  - {msg}" for msg in errors)
        )

def wrap_xlscc(mlir_module_text: str,
               core_code: str,
               function_names: Tuple[str, str],
               function_inputs: List[str]) -> str:
    """
    mlir_module_text: MLIR (string) with affine.for loops.
    core_code:        C++ code produced by emitXlscc (contains 'top_name').
    function_name:    (foo, bar) -> foo is top, bar is generated function
    Returns:          Full C++ code with channel, core code, etc. wrapped in Xls [cc] Style
    """
    parts: List[str] = []
    parts.append(core_code)
    parts.append("\n// ---- End core code ----\n")
    return "".join(parts)



if __name__ == "__main__":
    # Example core C++ code that emitXlscc might return
    core_code = """
int add(int a, int b) {
    return a + b;
}
    """

    # (top_name, inner_function_name)
    function_names = ("Run", "add")
    function_inputs = ["a", "b"]

    wrapped_cpp = wrap_xlscc(
        mlir_module_text="",
        core_code=core_code,
        function_names=function_names,
        function_inputs=function_inputs,
    )

    print(wrapped_cpp)