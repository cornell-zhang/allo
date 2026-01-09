# allo/allo/backend/xls/tester.py
# generate dslx test harnesses for function verification.

from allo._mlir.dialects import func as func_d

from ...utils import get_func_inputs_outputs
from .memory import discover_memory_bindings
from .utils import allo_dtype_to_dslx_type, is_float_type, float_to_dslx_literal


class DslxTestBuilder:
  # Builder for generating dslx #[test_proc] test harnesses.
  def __init__(self, module, func_name):
    self.module = module
    self.func_name = func_name
    self.func = self._find_func()
    self.float_types_used = set()
    inputs, outputs = get_func_inputs_outputs(self.func)
    self.inputs_meta = inputs
    self.outputs_meta = outputs
    self.scalar_input_indices = [i for i, (_, shape) in enumerate(inputs) if shape == ()]
    self.scalar_output_indices = [i for i, (_, shape) in enumerate(outputs) if shape == ()]
    # track float types used
    for i in self.scalar_input_indices:
      dtype = inputs[i][0]
      if is_float_type(dtype):
        self.float_types_used.add(dtype)
    for i in self.scalar_output_indices:
      dtype = outputs[i][0]
      if is_float_type(dtype):
        self.float_types_used.add(dtype)
    self.scalar_input_types = [
        allo_dtype_to_dslx_type(inputs[i][0]) for i in self.scalar_input_indices
    ]
    self.scalar_output_types = [
        allo_dtype_to_dslx_type(outputs[i][0]) for i in self.scalar_output_indices
    ]
    self.scalar_input_dtypes = [inputs[i][0] for i in self.scalar_input_indices]
    self.scalar_output_dtypes = [outputs[i][0] for i in self.scalar_output_indices]
    self.func_args = [arg for arg in self.func.arguments if "!allo.stream" not in str(arg.type)]
    self.memory_bindings, self.memory_map = discover_memory_bindings(self.func)
    self.binding_by_arg = {binding.value: binding for binding in self.memory_bindings}
    self.mem_value_map = {binding.key: binding for binding in self.memory_bindings}
    self.arg_index_to_binding = self._map_arg_bindings()
    self.output_binding_map = self._map_output_bindings()
    self.num_inputs = len(inputs)
    self.num_outputs = len(outputs)

  def _find_func(self):
    for op in self.module.body.operations:
      if isinstance(op, func_d.FuncOp) and op.name.value == self.func_name:
        return op
    raise RuntimeError(f"function {self.func_name} not found")

  def _map_arg_bindings(self):
    return {
        binding.arg_index: binding
        for binding in self.memory_bindings
        if binding.arg_index is not None
    }

  def _map_output_bindings(self):
    mapping = {}
    ret_vals = []
    for op in self.func.body.blocks[0].operations:
      if isinstance(op, func_d.ReturnOp):
        ret_vals = list(op.operands)
        break
    for idx, (_, shape) in enumerate(self.outputs_meta):
      if not shape or idx >= len(ret_vals):
        continue
      binding = self.mem_value_map.get(str(ret_vals[idx]))
      if binding:
        mapping[idx] = binding
    return mapping

  def _literal(self, value, dtype, raw_dtype=None):
    if isinstance(value, str):
      return value if ":" in value else f"{dtype}:{value}"
    if isinstance(value, bool):
      return f"{dtype}:{1 if value else 0}"
    if raw_dtype and is_float_type(raw_dtype):
      return float_to_dslx_literal(float(value), raw_dtype)
    return f"{dtype}:{value}"

  def _literal_for_binding(self, binding, value):
    return self._literal(value, binding.dslx_type)

  def _flatten_array(self, value):
    if hasattr(value, "tolist"):
      value = value.tolist()
    if isinstance(value, (list, tuple)):
      flat = []
      for elem in value:
        flat.extend(self._flatten_array(elem))
      return flat
    return [value]

  def _normalize_array(self, raw, binding, label, idx):
    if raw is None:
      raise ValueError(f"{label} {idx} must be provided as an array")
    values = self._flatten_array(raw)
    if len(values) != binding.size:
      raise ValueError(f"{label} {idx} expected {binding.size} values, got {len(values)}")
    return [self._literal_for_binding(binding, v) for v in values]

  def _gather_array_inputs(self, inputs):
    return {
        binding.base: self._normalize_array(inputs[binding.arg_index], binding, "input", binding.arg_index)
        for binding in self.memory_bindings if binding.arg_index is not None
    }

  def _gather_array_outputs(self, outputs):
    return {
        binding.base: self._normalize_array(outputs[out_idx], binding, "output", out_idx)
        for out_idx, binding in self.output_binding_map.items()
    }

  def _split_values(self, values):
    total = self.num_inputs + self.num_outputs
    if len(values) != total:
      raise ValueError(f"expected {total} values ({self.num_inputs} inputs + {self.num_outputs} outputs), got {len(values)}")
    return values[:self.num_inputs], values[self.num_inputs:]

  def emit_from_values(self, test_cases):
    """Generate test harness. Accepts single case (tuple) or list of cases."""
    if not test_cases:
      raise ValueError("no test cases provided")
    # Normalize: single tuple becomes list of one
    if not isinstance(test_cases[0], (list, tuple)):
      test_cases = [test_cases]
    cases = [self._split_values(v) for v in test_cases]
    return self._emit(cases)

  def _emit(self, cases):
    """Generate test harness for list of (inputs, outputs) pairs."""
    for inputs, outputs in cases:
      assert len(inputs) == self.num_inputs, "input count mismatch"
      assert len(outputs) == self.num_outputs, "output count mismatch"
    if self.memory_bindings:
      return self._emit_memory(cases)
    return self._emit_scalar(cases)

  def _memory_write_stmt(self, binding, addr, literal):
    return (
        f"    let tok = send(tok, {binding.write_req_chan}_s, "
        f"SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}> "
        f"{{ addr: uN[{binding.addr_width}]:{addr}, data: ({literal} as uN[{binding.data_width}]) }});\n"
        f"    let (tok, _) = recv(tok, {binding.write_resp_chan}_r);"
    )

  def _memory_read_stmt(self, binding, addr, literal, idx):
    val_name = f"{binding.base}_val_{idx}"
    return (
        f"    let tok = send(tok, {binding.read_req_chan}_s, "
        f"SimpleReadReq<{binding.addr_param()}> {{ addr: uN[{binding.addr_width}]:{addr} }});\n"
        f"    let (tok, resp_{val_name}) = recv(tok, {binding.read_resp_chan}_r);\n"
        f"    let {val_name} = (resp_{val_name}.data as {binding.dslx_type});\n"
        f"    assert_eq({val_name}, {literal});"
    )

  def _wrap_proc(self, chan_block, cfg_block, next_block):
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

  def _emit_scalar(self, cases):
    # Channels
    chan_lines = ["  terminator: chan<bool> out;"]
    for idx, dtype in enumerate(self.scalar_input_types):
      chan_lines.append(f"  in{idx}_s: chan<{dtype}> out;")
    for idx, dtype in enumerate(self.scalar_output_types):
      chan_lines.append(f"  out{idx}_r: chan<{dtype}> in;")

    # Config
    cfg_lines = []
    for idx, dtype in enumerate(self.scalar_input_types):
      cfg_lines.append(f"    let (in{idx}_s, in{idx}_r) = chan<{dtype}>(\"in{idx}\");")
    for idx, dtype in enumerate(self.scalar_output_types):
      cfg_lines.append(f"    let (out{idx}_s, out{idx}_r) = chan<{dtype}>(\"out{idx}\");")
    spawn_args = [f"in{i}_r" for i in range(len(self.scalar_input_types))]
    spawn_args += [f"out{i}_s" for i in range(len(self.scalar_output_types))]
    cfg_lines.append(f"    spawn {self.func_name}({', '.join(spawn_args)});")
    handles = ["terminator"] + [f"in{i}_s" for i in range(len(self.scalar_input_types))]
    handles += [f"out{i}_r" for i in range(len(self.scalar_output_types))]
    cfg_lines.append(f"    ({', '.join(handles)})")

    # Next block - iterate over all cases
    next_lines = ["    let tok = join();"]
    for case_idx, (inputs, outputs) in enumerate(cases):
      if len(cases) > 1:
        next_lines.append(f"    // test case {case_idx}")
      for idx, global_idx in enumerate(self.scalar_input_indices):
        literal = self._literal(inputs[global_idx], self.scalar_input_types[idx], self.scalar_input_dtypes[idx])
        next_lines.append(f"    let tok = send(tok, in{idx}_s, {literal});")
      for idx, global_idx in enumerate(self.scalar_output_indices):
        recv_name = f"result_{case_idx}_{idx}" if len(cases) > 1 else f"result_{idx}"
        next_lines.append(f"    let (tok, {recv_name}) = recv(tok, out{idx}_r);")
        literal = self._literal(outputs[global_idx], self.scalar_output_types[idx], self.scalar_output_dtypes[idx])
        next_lines.append(f"    assert_eq({recv_name}, {literal});")
    next_lines.append("    send(tok, terminator, true);")

    return self._wrap_proc("\n".join(chan_lines), "\n".join(cfg_lines), "\n".join(next_lines))

  def _emit_memory(self, cases):
    # Channels
    chan_lines = ["  terminator: chan<bool> out;"]
    handle_order = ["terminator"]
    for idx, dtype in enumerate(self.scalar_input_types):
      chan_lines.append(f"  in{idx}_s: chan<{dtype}> out;")
      handle_order.append(f"in{idx}_s")
    for idx, dtype in enumerate(self.scalar_output_types):
      chan_lines.append(f"  out{idx}_r: chan<{dtype}> in;")
      handle_order.append(f"out{idx}_r")
    for binding in self.memory_bindings:
      chan_lines.append(f"  {binding.read_req_chan}_s: chan<SimpleReadReq<{binding.addr_param()}>> out;")
      chan_lines.append(f"  {binding.read_resp_chan}_r: chan<SimpleReadResp<{binding.data_param()}>> in;")
      handle_order.extend([f"{binding.read_req_chan}_s", f"{binding.read_resp_chan}_r"])
      chan_lines.append(f"  {binding.write_req_chan}_s: chan<SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}>> out;")
      chan_lines.append(f"  {binding.write_resp_chan}_r: chan<SimpleWriteResp> in;")
      handle_order.extend([f"{binding.write_req_chan}_s", f"{binding.write_resp_chan}_r"])
    chan_lines.extend(["  go_s: chan<bool> out;", "  done_r: chan<bool> in;"])
    handle_order.extend(["go_s", "done_r"])

    # Config
    cfg_lines = []
    for idx, dtype in enumerate(self.scalar_input_types):
      cfg_lines.append(f"    let (in{idx}_s, in{idx}_r) = chan<{dtype}>(\"in{idx}\");")
    for idx, dtype in enumerate(self.scalar_output_types):
      cfg_lines.append(f"    let (out{idx}_s, out{idx}_r) = chan<{dtype}>(\"out{idx}\");")

    for binding in self.memory_bindings:
      cfg_lines.append(f"    let ({binding.read_req_chan}_s, {binding.read_req_chan}_r) = chan<SimpleReadReq<{binding.addr_param()}>>(\"{binding.read_req_chan}\");")
      cfg_lines.append(f"    let ({binding.read_resp_chan}_s, {binding.read_resp_chan}_r) = chan<SimpleReadResp<{binding.data_param()}>>(\"{binding.read_resp_chan}\");")
      cfg_lines.append(f"    let ({binding.write_req_chan}_s, {binding.write_req_chan}_r) = chan<SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}>>(\"{binding.write_req_chan}\");")
      cfg_lines.append(f"    let ({binding.write_resp_chan}_s, {binding.write_resp_chan}_r) = chan<SimpleWriteResp>(\"{binding.write_resp_chan}\");")
      cfg_lines.append(f"    spawn Simple1R1WRam<u32:{binding.addr_width}, u32:{binding.data_width}, u32:{binding.size}>({binding.read_req_chan}_r, {binding.read_resp_chan}_s, {binding.write_req_chan}_r, {binding.write_resp_chan}_s);")

    cfg_lines.extend(["    let (go_s, go_r) = chan<bool>(\"go\");", "    let (done_s, done_r) = chan<bool>(\"done\");"])

    spawn_args = [f"in{i}_r" for i in range(len(self.scalar_input_types))]
    spawn_args += [f"out{i}_s" for i in range(len(self.scalar_output_types))]
    for binding in self.memory_bindings:
      if binding.needs_read:
        spawn_args.extend([f"{binding.read_req_chan}_s", f"{binding.read_resp_chan}_r"])
      if binding.needs_write:
        spawn_args.extend([f"{binding.write_req_chan}_s", f"{binding.write_resp_chan}_r"])
    spawn_args.extend(["go_r", "done_s"])
    cfg_lines.append(f"    spawn {self.func_name}({', '.join(spawn_args)});")
    cfg_lines.append(f"    ({', '.join(handle_order)})")

    # Next block - iterate over all cases
    next_lines = ["    let tok = join();"]
    for case_idx, (inputs, outputs) in enumerate(cases):
      binding_inputs = self._gather_array_inputs(inputs)
      binding_outputs = self._gather_array_outputs(outputs)
      prefix = f"// case {case_idx}: " if len(cases) > 1 else "// "

      next_lines.append(f"    {prefix}preload memories")
      for binding in self.memory_bindings:
        array_vals = binding_inputs.get(binding.base)
        if array_vals:
          for addr, literal in enumerate(array_vals):
            next_lines.append(self._memory_write_stmt(binding, addr, literal))

      next_lines.append(f"    {prefix}drive scalar inputs")
      for idx, global_idx in enumerate(self.scalar_input_indices):
        literal = self._literal(inputs[global_idx], self.scalar_input_types[idx], self.scalar_input_dtypes[idx])
        next_lines.append(f"    let tok = send(tok, in{idx}_s, {literal});")

      next_lines.append(f"    {prefix}start DUT and wait")
      next_lines.append("    let tok = send(tok, go_s, bool:1);")

      for idx, global_idx in enumerate(self.scalar_output_indices):
        recv_name = f"result_{case_idx}_{idx}" if len(cases) > 1 else f"result_{idx}"
        next_lines.append(f"    let (tok, {recv_name}) = recv(tok, out{idx}_r);")
        literal = self._literal(outputs[global_idx], self.scalar_output_types[idx], self.scalar_output_dtypes[idx])
        next_lines.append(f"    assert_eq({recv_name}, {literal});")

      next_lines.append("    let (tok, done_flag) = recv(tok, done_r);")
      next_lines.append("    assert_eq(done_flag, bool:1);")

      for binding in self.memory_bindings:
        expected_vals = binding_outputs.get(binding.base)
        if expected_vals:
          next_lines.append(f"    {prefix}verify {binding.base}")
          for addr, literal in enumerate(expected_vals):
            next_lines.append(self._memory_read_stmt(binding, addr, literal, f"{case_idx}_{addr}" if len(cases) > 1 else addr))

    next_lines.append("    send(tok, terminator, true);")

    return self._wrap_proc("\n".join(chan_lines), "\n".join(cfg_lines), "\n".join(next_lines))
