# Allo XLS Backend

> **Note:** This backend will be merged into the main Allo repository via PR over break.

This directory contains the complete Allo codebase with backend support for Google's XLS (Accelerated HW Synthesis) toolchain. The backend provides two compilation targets: **DSLX** and **XLS[cc]**.

---

## DSLX Backend

DSLX is XLS's domain-specific language for hardware description. Allo can lower Python functions directly to DSLX code with full XLS toolchain integration.

### Basic Usage

```python
import allo
from allo.ir.types import int32, uint32

def add(a: uint32, b: uint32) -> uint32:
    return a + b

s = allo.customize(add)
code = s.build(target='xls')

# Print the generated DSLX code
print(code)
```

### Testing with the XLS Interpreter

The DSLX backend provides integrated testing via the XLS interpreter:

```python
# Single test case: add(1, 2) should equal 3
code.test(1, 2, 3)

# Batch testing with multiple cases
code.test([(0, 0, 0), (1, 2, 3), (123, 456, 579), (2**16, 2**16, 2**17)])
```

### Full Compilation Pipeline

```python
# Step-by-step pipeline
code.interpret()     # Generate and validate DSLX
code.to_ir()         # Convert to XLS IR
code.opt()           # Optimize IR
code.to_vlog()       # Generate Verilog

# Or run the full pipeline in one call
code.flow()
```

### Supported Features

The DSLX backend supports:

- **Scalar operations**: arithmetic, bitwise, comparisons
- **Signed and unsigned integers**: `int32`, `uint32`, etc.
- **Floating-point**: `float32` (via XLS apfloat library)
- **Conditionals**: `if`/`else` expressions
- **For loops**: with automatic state machine generation
- **While loops**: with proper state handling
- **Arrays/Memories**: with automatic RAM model generation
- **Multiple outputs**: tuple returns
- **Loop unrolling**: via `s.unroll("loop_name")`

### Examples

```python
# Multiply-accumulate
def mac(a: int32, b: int32, c: int32) -> int32:
    return (a * b) + c

# Factorial with for loop
def fact(a: int32) -> int32:
    acc: int32 = 1
    for i in range(a):
        acc *= (i + 1)
    return acc

# GCD with while loop
def gcd(a: int32, b: int32) -> int32:
    x: int32 = a
    y: int32 = b
    while y > 0:
        temp: int32 = y
        y = x % y
        x = temp
    return x

# Vector-vector add
def vvadd(a: int32[16], b: int32[16]) -> int32[16]:
    c: int32[16] = 0
    for i in range(16):
        c[i] = a[i] + b[i]
    return c

# Matrix-vector multiply
def mv[N](A: int32[N, N], x: int32[N]) -> int32[N]:
    C: int32[N] = 0
    for i in range(N):
        acc: int32 = 0
        for j in range(N):
            acc += A[i, j] * x[j]
        C[i] = acc
    return C
```

### Verilog Generation Options

```python
code.to_vlog(
    ram_latency=1,       # RAM access latency
    pipeline_stages=3,   # Number of pipeline stages
    delay_model="sky130" # Target delay model
)
```

### Requirements

The DSLX backend requires XLS tools to be installed and available in PATH:
- `interpreter_main` - DSLX interpreter
- `ir_converter_main` - DSLX to XLS IR converter
- `opt_main` - XLS IR optimizer
- `codegen_main` - Verilog code generator

---

## XLS[cc] Backend

XLS[cc] compiles C++ to hardware via XLS. Allo generates XLS[cc]-compatible C++ code from Python functions.

### Basic Usage

```python
import allo
from allo.ir.types import int32

def add(a: int32, b: int32) -> int32:
    return a + b

s = allo.customize(add)

# Combinational logic (scalar inputs/outputs)
mod = s.build(target="xls", project="add.prj")
print(mod.hls_code)

# Sequential logic with register mode
mod_reg = s.build(target="xls", project="gemm_reg.prj", use_memory=False)

# Sequential logic with memory mode
mod_mem = s.build(target="xls", project="gemm_mem.prj", use_memory=True)
```

### Output Files

After calling `s.build()`, the generated files are written to the specified project directory:

- `test_block.cpp` - The generated C++ code
- `block.textproto` - HLSBlock configuration for XLS[cc]
- `rewrites.textproto` - RAM rewrite configuration (memory mode only)

### Printing Configuration

```python
# Print the generated C++ code
print(mod.hls_code)

# Print the HLSBlock textproto
mod.print_textproto()
```

### XLS Integration

> **Note:** Allo currently does not have direct XLS toolchain integration for XLS[cc]. Users are expected to have XLS installed separately and copy the generated files to their XLS workflow.

To compile the generated code with XLS:

1. Copy the project directory contents to your XLS environment
2. Run `xlscc` to compile the C++ to XLS IR:
   ```bash
   xlscc test_block.cpp \
     --block_pb block.textproto \
     --block_from_class TestBlock \
     > output.ir
   ```
   The `--block_from_class TestBlock` flag is required since Allo generates code using a `TestBlock` class wrapper.

3. Optimize the IR:
   ```bash
   opt_main output.ir > output.opt.ir
   ```

4. Generate Verilog with `codegen_main`:
   
   **For register mode (no memories):**
   ```bash
   codegen_main output.opt.ir \
     --generator=pipeline \
     --delay_model="sky130" \
     --output_verilog_path=output.v \
     --module_name=my_module \
     --top=TestBlock_proc \
     --reset=rst \
     --reset_active_low=false \
     --reset_asynchronous=false \
     --reset_data_path=true \
     --pipeline_stages=2 \
     --flop_inputs=false \
     --flop_outputs=false
   ```

   **For memory mode:**
   ```bash
   codegen_main output.opt.ir \
     --generator=pipeline \
     --delay_model="sky130" \
     --output_verilog_path=output.v \
     --module_name=my_module \
     --top=TestBlock_proc \
     --reset=rst \
     --reset_active_low=false \
     --reset_asynchronous=false \
     --reset_data_path=true \
     --pipeline_stages=2 \
     --flop_inputs=false \
     --flop_outputs=false \
     --ram_rewrites_pb rewrites.textproto \
     --ram_configurations=v0:1R1W:v0__read_req:v0__read_resp:v0__write_req:v0__write_completion \
     --ram_configurations=v1:1R1W:v1__read_req:v1__read_resp:v1__write_req:v1__write_completion \
     --ram_configurations=c:1R1W:c__read_req:c__read_resp:c__write_req:c__write_completion
   ```
   
   The `--ram_rewrites_pb rewrites.textproto` flag specifies the RAM rewrite configuration file (generated automatically in memory mode). Each memory requires a separate `--ram_configurations` flag with the format:
   ```
   --ram_configurations=<name>:<kind>:<read_req>:<read_resp>:<write_req>:<write_completion>
   ```
   
   Where:
   - `<name>`: Memory name (e.g., `v0`, `v1`, `c`)
   - `<kind>`: RAM type (e.g., `1R1W` for 1-read-1-write)
   - Channel names use double underscores (`__`) as separators

### Testing

XLS[cc] tests compile generated C++ with mock XLS primitives and execute against reference Python implementations:

```bash
python3 tests/test_xls_functionality.py
```

Example test output:

```
XLS[cc] Functionality Tests
============================================================

Test: add ......
Test: multiply .....
Test: subtract .....
Test: complex_expr ....
Test: vvadd (sequential) ...
Test: vvadd_16 (size 16) .
Test: mv (mat-vec) ..

============================================================
TOTAL: 26 passed, 0 failed
============================================================
```

Test outputs are written to `tests/xlscc_tests/`. The test framework supports:
- **Combinational logic**: Scalar inputs/outputs tested directly
- **Sequential logic**: Streaming interfaces with channel simulation

### Supported Features

- Fixed-width integers (`int8`, `int16`, `int32`, `int64`, `uint8`, etc.)
- Fixed-point arithmetic
- Single-function lowering for combinational and sequential logic
- Loop pipelining with initiation interval (`pipeline(ii=N)`)
- Full loop unrolling (`unroll`)
- Register mode and memory mode for array storage

### Current Limitations

- Floating-point types are not supported in XLS[cc] (use DSLX backend instead)
- Multi-function lowering is not yet implemented
- Other Allo optimizations (tiling, reordering, etc.) are currently unsupported for XLS targets
