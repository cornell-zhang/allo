# allo/allo/backend/xls/lowerer.py

from allo._mlir.dialects import func as func_d
from allo._mlir.ir import Module

def lower_mlir(module: Module, top_func_name: str,**kwargs) -> str:
  return "soon to be XLS code"