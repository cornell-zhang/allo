# allo/allo/backend/xls/xls.py
# xls backend interface for code generation and testing.

import os
import subprocess
import shutil

from .lowerer import lower_mlir
from .tester import DslxTestBuilder
from .memory import detect_memory_and_constraints

# find xls base directory by locating interpreter_main
def find_xls_stdlib():
  interpreter_path = shutil.which("interpreter_main")
  if not interpreter_path:
    return None
  interpreter_dir = os.path.dirname(os.path.abspath(interpreter_path))
  stdlib_path = os.path.join(interpreter_dir, "xls", "dslx", "stdlib")
  if os.path.isdir(stdlib_path):
    return interpreter_dir
  stdlib_path = os.path.join(interpreter_dir, "dslx", "stdlib")
  if os.path.isdir(stdlib_path):
    return interpreter_dir
  return None

# main interface for xls code generation and toolchain execution
class DSLXModule:
  def __init__(self, mod, top_func_name):
    self.module = mod
    self.func = top_func_name
    self._out_dir = "abax"
    self._stdlib_path = find_xls_stdlib()

  @property
  def _dslx_path(self):
    return os.path.join(self._out_dir, f"{self.func}.x")

  def _write_dslx(self, source):
    os.makedirs(self._out_dir, exist_ok=True)
    with open(self._dslx_path, "w") as f:
      f.write(source)
    return self._dslx_path

  # generate dslx code from mlir
  def codegen(self) -> str:
    return lower_mlir(self.module, self.func)
  
  # run interpreter_main on a dslx file
  def _run_interpreter(self, path, verbose=True):
    cmd = ["interpreter_main"]
    if self._stdlib_path:
      cmd.append(f"--dslx_path={self._stdlib_path}")
    cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
      print(result.stdout)
      print(result.stderr)
    return result

  # run dslx interpreter on generated code
  def interpret(self, verbose=True):
    path = self._write_dslx(self.codegen())
    return self._run_interpreter(path, verbose)

  # generate test harness and run interpreter
  # accepts either: test(a, b, expected) for single case
  #             or: test([(a1, b1, exp1), (a2, b2, exp2), ...]) for batch
  def test(self, *args):
    # If first arg is a list of tuples/lists, treat as batch
    if len(args) == 1 and isinstance(args[0], list) and args[0] and isinstance(args[0][0], (list, tuple)):
      test_cases = args[0]
    else:
      test_cases = [args]  # single test case
    base_source = self.codegen()
    builder = DslxTestBuilder(self.module, self.func)
    harness = builder.emit_from_values(test_cases)
    path = self._write_dslx(f"{base_source}\n\n{harness}\n")
    return self._run_interpreter(path, verbose=True)
  
  # convert dslx to xls ir
  def to_ir(self, verbose=True):
    if not os.path.isfile(self._dslx_path):
      raise FileNotFoundError(f"please run interpret() first")
    cmd = ["ir_converter_main", f"--top={self.func}"]
    if self._stdlib_path:
      cmd.append(f"--dslx_path={self._stdlib_path}")
    cmd.append(self._dslx_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(f"{self._out_dir}/{self.func}.ir", "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  # optimize xls ir
  def opt(self, verbose=True):
    ir_path = os.path.join(self._out_dir, f"{self.func}.ir")
    if not os.path.isfile(ir_path):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", ir_path],
                             capture_output=True, text=True)
    opt_path = os.path.join(self._out_dir, f"{self.func}.opt.ir")
    with open(opt_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  # generate verilog from optimized ir
  def to_vlog(self, verbose=True, pipeline_stages=None, clock_period_ps=None, 
              delay_model="sky130", reset="rst", ram_latency=1):
    opt_path = os.path.join(self._out_dir, f"{self.func}.opt.ir")
    if not os.path.isfile(opt_path):
      raise FileNotFoundError(f"please run opt() first")
    
    if pipeline_stages is not None and clock_period_ps is not None:
      raise ValueError("cannot specify both pipeline_stages and clock_period_ps")
    
    io_constraints, has_memory = detect_memory_and_constraints(opt_path, ram_latency)
    
    if pipeline_stages is None and clock_period_ps is None:
      if has_memory:
        pipeline_stages = max(ram_latency + 2, 3)
      else:
        pipeline_stages = 1
    
    cmd = ["codegen_main", f"--delay_model={delay_model}", f"--reset={reset}"]
    if pipeline_stages is not None:
      cmd.append(f"--pipeline_stages={pipeline_stages}")
    elif clock_period_ps is not None:
      cmd.append(f"--clock_period_ps={clock_period_ps}")
    if io_constraints:
      io_constraints_str = ",".join(io_constraints)
      cmd.append(f"--io_constraints={io_constraints_str}")
    cmd.append(opt_path)
    
    if verbose:
      print(" ".join(cmd))
      print()
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
      vlog_path = os.path.join(self._out_dir, f"{self.func}.v")
      with open(vlog_path, "w") as f:
        f.write(result.stdout)
      if verbose:
        print(result.stdout)
        print(result.stderr)
    else:
      print(f"codegen_main failed with exit code {result.returncode}")
      if result.stderr:
        print("stderr:")
        print(result.stderr)
      if result.stdout:
        print("stdout:")
        print(result.stdout)
      raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
  
  # run full pipeline: interpret -> ir -> optimize -> verilog
  def flow(self):
    self.interpret(verbose=False)
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.codegen()
