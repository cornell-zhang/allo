# allo/allo/backend/xls/xls.py
# XLS backend interface for code generation and testing.

import os
import subprocess

from .lowerer import lower_mlir
from .tester import DslxTestBuilder
from .memory import detect_memory_and_constraints

# Main interface for XLS code generation and toolchain execution.
class XLSModule:
  def __init__(self, mod, top_func_name):
    self.module = mod
    self.func = top_func_name
    self._out_dir = "abax"

  @property
  def _dslx_path(self):
    return os.path.join(self._out_dir, f"{self.func}.x")

  def _write_dslx(self, source):
    os.makedirs(self._out_dir, exist_ok=True)
    with open(self._dslx_path, "w") as f:
      f.write(source)
    return self._dslx_path

  # Generate DSLX code from MLIR.
  def codegen(self) -> str:
    return lower_mlir(self.module, self.func)
  
  # Run DSLX interpreter on generated code.
  def interpret(self, verbose=True):
    path = self._write_dslx(self.codegen())
    result = subprocess.run(["interpreter_main", path],
                            capture_output=True, text=True)
    # echo output
    if verbose:
      print(result.stdout)
      print(result.stderr)
    return result

  # Generate test harness and run interpreter with provided values.
  def test(self, *values):
    builder = DslxTestBuilder(self.module, self.func)
    base_source = self.codegen()
    harness = builder.emit_from_values(values)
    path = self._write_dslx(f"{base_source}\n\n{harness}\n")
    result = subprocess.run(["interpreter_main", path],
                            capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    self._write_dslx(base_source)
    return result
  
  # Convert DSLX to XLS IR.
  def to_ir(self, verbose=True):
    if not os.path.isfile(self._dslx_path):
      raise FileNotFoundError(f"please run interpret() first")
    result = subprocess.run(["ir_converter_main", f"--top={self.func}",
                             self._dslx_path],
                             capture_output=True, text=True)
    # write output to file
    with open(f"{self._out_dir}/{self.func}.ir", "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  # Optimize XLS IR.
  def opt(self, verbose=True):
    ir_path = os.path.join(self._out_dir, f"{self.func}.ir")
    if not os.path.isfile(ir_path):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", ir_path],
                             capture_output=True, text=True)
    # write output to file
    opt_path = os.path.join(self._out_dir, f"{self.func}.opt.ir")
    with open(opt_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  # generate verilog from optimized ir.
  def to_vlog(self, verbose=True, pipeline_stages=None, clock_period_ps=None, 
              delay_model="sky130", reset="rst", ram_latency=1):
    opt_path = os.path.join(self._out_dir, f"{self.func}.opt.ir")
    if not os.path.isfile(opt_path):
      raise FileNotFoundError(f"please run opt() first")
    
    # cannot specify both pipeline_stages and clock_period_ps
    if pipeline_stages is not None and clock_period_ps is not None:
      raise ValueError("cannot specify both pipeline_stages and clock_period_ps")
    
    # detect memory channels and build io constraints
    io_constraints, has_memory = detect_memory_and_constraints(opt_path, ram_latency)
    
    # if both are None, auto-detect based on memory presence
    if pipeline_stages is None and clock_period_ps is None:
      if has_memory:
        # memory requests take ram_latency cycles for response plus overhead
        pipeline_stages = max(ram_latency + 2, 3)
      else:
        pipeline_stages = 1
    
    # build codegen_main command
    cmd = ["codegen_main", f"--delay_model={delay_model}", f"--reset={reset}"]
    
    # add either pipeline_stages or clock_period_ps (not both)
    if pipeline_stages is not None:
      cmd.append(f"--pipeline_stages={pipeline_stages}")
    elif clock_period_ps is not None:
      cmd.append(f"--clock_period_ps={clock_period_ps}")
    
    # add io constraints for ram response delays if we have memory channels
    if io_constraints:
      io_constraints_str = ",".join(io_constraints)
      cmd.append(f"--io_constraints={io_constraints_str}")
    
    cmd.append(opt_path)
    
    # print command if verbose
    if verbose:
      print(" ".join(cmd))
      print()
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # write output to file if successful
    if result.returncode == 0:
      vlog_path = os.path.join(self._out_dir, f"{self.func}.v")
      with open(vlog_path, "w") as f:
        f.write(result.stdout)
      if verbose:
        print(result.stdout)
        print(result.stderr)
    else:
      # print error message on failure
      print(f"codegen_main failed with exit code {result.returncode}")
      if result.stderr:
        print("stderr:")
        print(result.stderr)
      if result.stdout:
        print("stdout:")
        print(result.stdout)
      raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
  
  # Run full pipeline: interpret -> IR -> optimize -> Verilog.
  def flow(self):
    self.interpret(verbose=False)
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.codegen()
