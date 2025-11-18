# allo/allo/backend/xls/xls.py

from .lowerer import lower_mlir
import os
import subprocess

class XLSModule:
  def __init__(self, mod, top_func_name):
    self.module = mod
    self.func = top_func_name

  def codegen(self) -> str:
    return lower_mlir(self.module, self.func)
  
  def interpret(self, verbose=True):
    os.makedirs("abax", exist_ok=True)
    # write code to file
    with open(f"abax/{self.func}.x", "w") as f:
      f.write(self.codegen())
    # run interpreter
    result = subprocess.run(["interpreter_main", f"abax/{self.func}.x"], 
                            capture_output=True, text=True)
    # echo output
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  def to_ir(self, verbose=True):
    if not os.path.isfile(f"abax/{self.func}.x"):
      raise FileNotFoundError(f"please run interpret() first")
    result = subprocess.run(["ir_converter_main", f"--top={self.func}",
                             f"abax/{self.func}.x"],
                             capture_output=True, text=True)
    # write output to file
    with open(f"abax/{self.func}.ir", "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  def opt(self, verbose=True):
    if not os.path.isfile(f"abax/{self.func}.ir"):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", f"abax/{self.func}.ir"],
                             capture_output=True, text=True)
    # write output to file
    with open(f"abax/{self.func}.opt.ir", "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      print(result.stderr)
  
  def to_vlog(self, verbose=True):
    if not os.path.isfile(f"abax/{self.func}.opt.ir"):
      raise FileNotFoundError(f"please run opt() first")
    result = subprocess.run(["codegen_main", "--pipeline_stages=1", "--delay_model=unit", 
                             "--reset=rst", f"abax/{self.func}.opt.ir"], capture_output=True, 
                             text=True, check=True)
    # write output to file
    with open(f"abax/{self.func}.v", "w") as f:
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
