# Allo XLS Backend

> **Note:** This backend will be merged into the main Allo repository via PR over break.

This directory contains Allo's backend support for Google's XLS (Accelerated HW Synthesis) toolchain. The backend provides two compilation targets: **DSLX** and **XLS[cc]**.

---

## DSLX Backend

DSLX is XLS's domain-specific language for hardware description. Allo can lower Python functions to DSLX code.

### Basic Usage

```python
import allo
from allo.ir.types import int32

def add(a: int32, b: int32) -> int32:
    return a + b

s = allo.customize(add)
mod = s.build(target="dslx")

# Print the generated DSLX code
print(mod.module)
```

### Testing

DSLX tests use the XLS interpreter to validate generated modules against reference Python implementations. Run tests with:

```bash
pytest tests/test_dslx_*.py -v
```

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

> **Note:** Allo currently does not have direct XLS toolchain integration. Users are expected to have XLS installed separately and copy the generated files to their XLS workflow.

To compile the generated code with XLS:

1. Copy the project directory contents to your XLS environment
2. Run `xlscc` to compile the C++ to XLS IR:
   ```bash
   xlscc test_block.cpp --block_pb block.textproto > output.ir
   ```
3. Optimize the IR:
   ```bash
   opt_main output.ir > output.opt.ir
   ```
4. Generate Verilog with `codegen_main`:
   ```bash
   codegen_main output.opt.ir \
     --generator=pipeline \
     --delay_model="sky130" \
     --output_verilog_path=output.v \
     --module_name=my_module \
     --top=TestBlock_proc \
     --pipeline_stages=2
   ```

For memory mode, add `--ram_configurations` flags for each memory.

### Testing

XLS[cc] tests compile generated C++ with mock XLS primitives and execute against reference Python implementations:

```bash
pytest tests/test_xls_*.py -v
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

- Floating-point types are not supported
- Multi-function lowering is not yet implemented
- Other Allo optimizations (tiling, reordering, etc.) are currently unsupported for XLS targets

