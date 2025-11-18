# XLS backend module
# holds MLIR module and emits DSLX

from .lowerer import lower_mlir

class XLSModule:
  def __init__(self, mod, top_func_name):
    self.module = mod
    self.top_func_name = top_func_name

  def codegen(self) -> str:
    return lower_mlir(self.module, self.top_func_name)

  def __str__(self):
    return self.codegen()
