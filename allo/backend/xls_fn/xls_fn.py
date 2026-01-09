# allo/backend/xls_fn/xls.py
# xls backend interface for function-based code generation and testing.

import os
import subprocess
import shutil

from allo._mlir.ir import Module as MlirModule
from allo._mlir import ir as mlir_ir
from allo._mlir.dialects import func as func_d

from .lowering import MlirToDslxFnLowerer, MlirToXlsIRLowerer


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


# parse mlir text and filter out allo dialect metadata
def parse_mlir_text(mlir_text):
  lines = mlir_text.splitlines()
  filtered_lines = []
  for line in lines:
    # skip allo dialect metadata operations
    if 'allo.create_op_handle' in line or 'allo.yield' in line:
      continue
    filtered_lines.append(line)

  cleaned_mlir = '\n'.join(filtered_lines)

  with mlir_ir.Context() as ctx:
    module = MlirModule.parse(cleaned_mlir, ctx)
    return module


# extract first function from mlir module
def extract_func_op(mlir_module):
  for op in mlir_module.body.operations:
    if isinstance(op, func_d.FuncOp):
      return op
  raise RuntimeError("no function found in mlir module")


# main interface for dslx function lowering (mlir -> dslx -> ir -> verilog)
class DslxFunctionModule:
  def __init__(self, mod, top_func_name):
    # accept either mlir string or module object
    if isinstance(mod, str):
      self.module = parse_mlir_text(mod)
    else:
      self.module = mod

    self.func = top_func_name
    self._out_dir = "xls_output"
    self._stdlib_path = find_xls_stdlib()

    # extract function op for lowering
    self.func_op = extract_func_op(self.module)

  @property
  def _dslx_path(self):
    return os.path.join(self._out_dir, f"{self.func}.x")

  @property
  def _ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.ir")

  @property
  def _opt_ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.opt.ir")

  @property
  def _verilog_path(self):
    return os.path.join(self._out_dir, f"{self.func}.v")

  def _write_dslx(self, source):
    os.makedirs(self._out_dir, exist_ok=True)
    with open(self._dslx_path, "w") as f:
      f.write(source)
    return self._dslx_path

  # generate dslx code from mlir using function lowerer
  def codegen(self) -> str:
    lowerer = MlirToDslxFnLowerer(self.func_op)
    return lowerer.lower()

  # run interpreter_main on a dslx file
  def _run_interpreter(self, path, verbose=True, compare_mode=None):
    cmd = ["interpreter_main"]
    if self._stdlib_path:
      cmd.append(f"--dslx_path={self._stdlib_path}")
    if compare_mode:
      cmd.append(f"--compare={compare_mode}")
    cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
      print(result.stdout)
      if result.stderr:
        print(result.stderr)
    return result

  # run dslx interpreter on generated code
  def interpret(self, verbose=True):
    path = self._write_dslx(self.codegen())
    return self._run_interpreter(path, verbose)

  # convert dslx to xls ir
  def to_ir(self, verbose=True):
    if not os.path.isfile(self._dslx_path):
      raise FileNotFoundError(f"please run interpret() first")
    cmd = ["ir_converter_main", f"--top={self.func}"]
    if self._stdlib_path:
      cmd.append(f"--dslx_path={self._stdlib_path}")
    cmd.append(self._dslx_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(self._ir_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      if result.stderr:
        print(result.stderr)
    return result

  # optimize xls ir
  def opt(self, verbose=True):
    if not os.path.isfile(self._ir_path):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", self._ir_path],
                             capture_output=True, text=True)
    with open(self._opt_ir_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      if result.stderr:
        print(result.stderr)
    return result

  # generate verilog from optimized ir
  def to_vlog(self, verbose=True, pipeline_stages=None, clock_period_ps=None,
              delay_model="sky130", reset="rst"):
    if not os.path.isfile(self._opt_ir_path):
      raise FileNotFoundError(f"please run opt() first")

    if pipeline_stages is not None and clock_period_ps is not None:
      raise ValueError("cannot specify both pipeline_stages and clock_period_ps")

    # default to 1 stage for function-based lowering
    if pipeline_stages is None and clock_period_ps is None:
      pipeline_stages = 1

    cmd = ["codegen_main", f"--delay_model={delay_model}", f"--reset={reset}"]
    if pipeline_stages is not None:
      cmd.append(f"--pipeline_stages={pipeline_stages}")
    elif clock_period_ps is not None:
      cmd.append(f"--clock_period_ps={clock_period_ps}")
    cmd.append(self._opt_ir_path)

    if verbose:
      print(" ".join(cmd))
      print()

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
      with open(self._verilog_path, "w") as f:
        f.write(result.stdout)
      if verbose:
        print(result.stdout)
        if result.stderr:
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
    return result

  # run full pipeline: mlir -> dslx -> ir -> optimize -> verilog
  def flow(self):
    self.interpret(verbose=False)
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.codegen()


# interface for direct ir lowering (mlir -> ir -> verilog, bypassing dslx)
class XlsIRModule:
  def __init__(self, mod, top_func_name):
    # accept either mlir string or module object
    if isinstance(mod, str):
      self.module = parse_mlir_text(mod)
    else:
      self.module = mod

    self.func = top_func_name
    self._out_dir = "xls_output"

    # extract function op for lowering
    self.func_op = extract_func_op(self.module)

  @property
  def _ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.ir")

  @property
  def _opt_ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.opt.ir")

  @property
  def _verilog_path(self):
    return os.path.join(self._out_dir, f"{self.func}.v")

  # generate xls ir directly from mlir (bypassing dslx)
  def to_ir(self, verbose=True):
    lowerer = MlirToXlsIRLowerer(self.func_op)
    ir_code = lowerer.lower()

    os.makedirs(self._out_dir, exist_ok=True)
    with open(self._ir_path, "w") as f:
      f.write(ir_code)

    if verbose:
      print(ir_code)

    return ir_code

  # optimize xls ir
  def opt(self, verbose=True):
    if not os.path.isfile(self._ir_path):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", self._ir_path],
                             capture_output=True, text=True)
    with open(self._opt_ir_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      if result.stderr:
        print(result.stderr)
    return result

  # generate verilog from optimized ir
  def to_vlog(self, verbose=True, pipeline_stages=None, clock_period_ps=None,
              delay_model="sky130", reset="rst"):
    if not os.path.isfile(self._opt_ir_path):
      raise FileNotFoundError(f"please run opt() first")

    if pipeline_stages is not None and clock_period_ps is not None:
      raise ValueError("cannot specify both pipeline_stages and clock_period_ps")

    # default to 1 stage for function-based lowering
    if pipeline_stages is None and clock_period_ps is None:
      pipeline_stages = 1

    cmd = ["codegen_main", f"--delay_model={delay_model}", f"--reset={reset}"]
    if pipeline_stages is not None:
      cmd.append(f"--pipeline_stages={pipeline_stages}")
    elif clock_period_ps is not None:
      cmd.append(f"--clock_period_ps={clock_period_ps}")
    cmd.append(self._opt_ir_path)

    if verbose:
      print(" ".join(cmd))
      print()

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
      with open(self._verilog_path, "w") as f:
        f.write(result.stdout)
      if verbose:
        print(result.stdout)
        if result.stderr:
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
    return result

  # run full pipeline: mlir -> ir -> optimize -> verilog
  def flow(self):
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.to_ir(verbose=False)
