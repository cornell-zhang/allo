"""Helpers for generating DSLX #[test_proc] blocks."""

from allo._mlir.dialects import func as func_d

from ...utils import get_func_inputs_outputs
from .utils import allo_dtype_to_dslx_type


class DslxTestBuilder:
  def __init__(self, module, func_name):
    self.module = module
    self.func_name = func_name
    self.func = self._find_func()
    inputs, outputs = get_func_inputs_outputs(self.func)
    self.input_types = [allo_dtype_to_dslx_type(dtype) for dtype, _ in inputs]
    self.output_types = [allo_dtype_to_dslx_type(dtype) for dtype, _ in outputs]
    self.num_inputs = len(self.input_types)
    self.num_outputs = len(self.output_types)

  def _find_func(self):
    for op in self.module.body.operations:
      if isinstance(op, func_d.FuncOp) and op.name.value == self.func_name:
        return op
    raise RuntimeError(f"function {self.func_name} not found")

  def _literal(self, value, dtype):
    if isinstance(value, str):
      return value if ":" in value else f"{dtype}:{value}"
    if isinstance(value, bool):
      return f"{dtype}:{1 if value else 0}"
    return f"{dtype}:{value}"

  def _split_values(self, values):
    total = self.num_inputs + self.num_outputs
    if len(values) != total:
      raise ValueError(
          f"expected {total} values ({self.num_inputs} inputs + "
          f"{self.num_outputs} outputs), got {len(values)}"
      )
    inputs = values[: self.num_inputs]
    outputs = values[self.num_inputs :]
    return inputs, outputs

  def emit_from_values(self, values):
    inputs, outputs = self._split_values(values)
    return self.emit(inputs, outputs)

  def emit(self, inputs, outputs):
    assert len(inputs) == self.num_inputs, "input count mismatch"
    assert len(outputs) == self.num_outputs, "output count mismatch"

    chan_lines = ["  terminator: chan<bool> out;"]
    for idx, dtype in enumerate(self.input_types):
      chan_lines.append(f"  in{idx}_s: chan<{dtype}> out;")
    for idx, dtype in enumerate(self.output_types):
      chan_lines.append(f"  out{idx}_r: chan<{dtype}> in;")
    chan_block = "\n".join(chan_lines)

    cfg_lines = []
    for idx, dtype in enumerate(self.input_types):
      cfg_lines.append(f"    let (in{idx}_s, in{idx}_r) = chan<{dtype}>(\"in{idx}\");")
    for idx, dtype in enumerate(self.output_types):
      cfg_lines.append(f"    let (out{idx}_s, out{idx}_r) = chan<{dtype}>(\"out{idx}\");")
    spawn_args = [f"in{idx}_r" for idx in range(self.num_inputs)]
    spawn_args += [f"out{idx}_s" for idx in range(self.num_outputs)]
    cfg_lines.append(f"    spawn {self.func_name}({', '.join(spawn_args)});")
    handles = ["terminator"]
    handles += [f"in{idx}_s" for idx in range(self.num_inputs)]
    handles += [f"out{idx}_r" for idx in range(self.num_outputs)]
    cfg_lines.append(f"    ({', '.join(handles)})")
    cfg_block = "\n".join(cfg_lines)

    next_lines = ["    let tok = join();", "    // single invocation"]
    for idx, value in enumerate(inputs):
      literal = self._literal(value, self.input_types[idx])
      next_lines.append(f"    let tok = send(tok, in{idx}_s, {literal});")
    for idx, value in enumerate(outputs):
      recv_name = f"result_{idx}"
      next_lines.append(f"    let (tok, {recv_name}) = recv(tok, out{idx}_r);")
      literal = self._literal(value, self.output_types[idx])
      next_lines.append(f"    assert_eq({recv_name}, {literal});")
    next_lines.append("    send(tok, terminator, true);")
    next_block = "\n".join(next_lines)

    return (
        f"#[test_proc]\n"
        f"proc {self.func_name}_test {{\n"
        f"{chan_block}\n\n"
        f"  config(terminator: chan<bool> out) {{\n"
        f"{cfg_block}\n"
        f"  }}\n\n"
        f"  init {{ () }}\n\n"
        f"  next(state: ()) {{\n"
        f"{next_block}\n"
        f"  }}\n"
        f"}}"
    )
