# allo/allo/backend/xls/xls.py

import os
import subprocess

from .lowerer import lower_mlir
from .tester import DslxTestBuilder

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

  def codegen(self) -> str:
    return lower_mlir(self.module, self.func)
  
  def interpret(self, verbose=True):
    path = self._write_dslx(self.codegen())
    result = subprocess.run(["interpreter_main", path],
                            capture_output=True, text=True)
    # echo output
    if verbose:
      print(result.stdout)
      print(result.stderr)
    return result

  def test(self, *values):
    """
    Emit a temporary #[test_proc] and run interpreter_main.

    Arguments mirror the original Allo function signature: provide all inputs
    first, followed by the expected outputs (if any).
    """
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
  
  def to_vlog(self, verbose=True):
    opt_path = os.path.join(self._out_dir, f"{self.func}.opt.ir")
    if not os.path.isfile(opt_path):
      raise FileNotFoundError(f"please run opt() first")
    result = subprocess.run(["codegen_main", "--pipeline_stages=1", "--delay_model=unit", 
                             "--reset=rst", opt_path], capture_output=True, 
                             text=True, check=True)
    # write output to file
    vlog_path = os.path.join(self._out_dir, f"{self.func}.v")
    with open(vlog_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  def flow(self):
    self.interpret(verbose=False)
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.codegen()
