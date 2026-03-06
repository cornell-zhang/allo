# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import re

import pytest

from allo.bindings import ir, allo as allo_d
from allo.schedule import HandleState, Schedule


MLIR_ONE_LOOP_SCF = """
module {
  func.func @kernel(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      memref.store %c0_i32, %arg0[%i] : memref<8xi32>
    }
    return
  }
}
"""


MLIR_TWO_LOOPS_SCF = """
module {
  func.func @kernel(%arg0: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %c0_i32 = arith.constant 0 : i32
        memref.store %c0_i32, %arg0[%i, %j] : memref<8x8xi32>
      }
    }
    return
  }
}
"""


MLIR_TWO_LOOPS_AFFINE = """
module {
  func.func @kernel(%arg0: memref<8x8xi32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %c0_i32 = arith.constant 0 : i32
        affine.store %c0_i32, %arg0[%i, %j] : memref<8x8xi32>
      }
    }
    return
  }
}
"""


MLIR_ALLOC_ROOT = """
module {
  func.func @kernel(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %buf = memref.alloc() : memref<8xi32>
    scf.for %i = %c0 to %c8 step %c1 {
      %v = memref.load %arg0[%i] : memref<8xi32>
      memref.store %v, %buf[%i] : memref<8xi32>
    }
    return
  }
}
"""


def _iter_nodes(node):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _valid_handles(sched: Schedule, identifier: str):
    return [
        h
        for h in sched.handles
        if h.identifier == identifier and h.state == HandleState.VALID
    ]


def _valid_ids(sched: Schedule):
    return {h.identifier for h in sched.handles if h.state == HandleState.VALID}


def test_example_proxy_tree_parse_and_completion():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_TWO_LOOPS_SCF)
    stats = ir.complete_op_identifiers(mod)
    root = ir.parse_proxy_tree(mod)

    loop_ids = [
        node.op_identifier for node in _iter_nodes(root) if node.op_kind == "scf.for"
    ]
    assert stats["visited"] > 0
    assert stats["assigned"] > 0
    assert loop_ids == ["kernel.L0", "kernel.L0.L0"]


def test_example_sched_constructors(tmp_path: Path):
    ctx = ir.Context()
    sched_from_string = Schedule.from_string(ctx, MLIR_ONE_LOOP_SCF)
    assert len(sched_from_string.handles) > 0

    ir_file = tmp_path / "example.mlir"
    ir_file.write_text(MLIR_ONE_LOOP_SCF, encoding="utf-8")
    sched_from_file = Schedule.from_file(ctx, str(ir_file))
    assert len(sched_from_file.handles) > 0

    ctx2 = ir.Context()
    ctx2.load_dialects()
    mod = ir.parse_from_string(ctx2, MLIR_ONE_LOOP_SCF)
    sched_from_module = Schedule.from_module(mod)
    assert len(sched_from_module.handles) > 0


def test_example_select_query_and_active():
    ctx = ir.Context()
    sched = Schedule.from_string(ctx, MLIR_TWO_LOOPS_SCF)

    loops = sched.query(op_kind="scf.for")
    assert len(loops) == 2

    sched.select("kernel.L0")
    assert sched.active is not None
    assert sched.active.identifier == "kernel.L0"

    nested = sched.query(op_kind="scf.for", under="kernel.L0")
    nested_ids = {handle.identifier for handle in nested}
    assert nested_ids == {"kernel.L0", "kernel.L0.L0"}

    with pytest.raises(ValueError, match="not loop-like"):
        sched.pipeline("kernel")


def test_example_split_then_pipeline_then_commit():
    ctx = ir.Context()
    sched = Schedule.from_string(ctx, MLIR_ONE_LOOP_SCF)

    old_loop = _valid_handles(sched, "kernel.L0")[0]
    sched.split("kernel.L0", factor=2)
    assert old_loop.state == HandleState.CONSUMED
    assert len(_valid_handles(sched, "kernel.L0::outer")) == 1
    assert len(_valid_handles(sched, "kernel.L0::inner")) == 1

    inner = _valid_handles(sched, "kernel.L0::inner")[0]
    sched.pipeline("kernel.L0::inner", ii=2)
    assert inner.meta["pipeline_ii"] == 2
    assert sched.dirty

    sched.refresh()
    assert not sched.dirty
    assert len(_valid_handles(sched, "kernel.L0::outer")) == 0
    assert len(_valid_handles(sched, "kernel.L0::inner")) == 0
    assert len(_valid_handles(sched, "kernel.L0")) == 1
    assert len(_valid_handles(sched, "kernel.L0.L0")) == 1


def test_example_tile_multi_targets():
    ctx = ir.Context()
    sched = Schedule.from_string(ctx, MLIR_TWO_LOOPS_SCF)

    sched.tile(["kernel.L0", "kernel.L0.L0"], factors=[2, 2])
    valid_ids = _valid_ids(sched)
    assert "kernel.L0::tile" in valid_ids
    assert "kernel.L0::point" in valid_ids
    assert "kernel.L0.L0::tile" in valid_ids
    assert "kernel.L0.L0::point" in valid_ids


def test_example_reorder_on_affine_loops_keeps_identifier():
    ctx = ir.Context()
    sched = Schedule.from_string(ctx, MLIR_TWO_LOOPS_AFFINE)

    sched.reorder(["kernel.L0", "kernel.L0.L0"], [1, 0])
    assert sched.dirty
    sched.refresh()
    assert not sched.dirty

    valid_ids = _valid_ids(sched)
    assert "kernel.L0" in valid_ids
    assert "kernel.L0.L0" in valid_ids


def test_example_partition_value_targets():
    ctx = ir.Context()
    sched_arg = Schedule.from_string(ctx, MLIR_ONE_LOOP_SCF)
    sched_arg.partition("kernel:arg0", dim=0)
    sched_arg.refresh()
    text_arg = str(sched_arg.module)
    assert "allo.part" in text_arg
    assert re.search(r"%arg0: memref<8xi32>\s*\{[^}]*allo.part", text_arg) is not None

    sched_alloc = Schedule.from_string(ctx, MLIR_ALLOC_ROOT)
    alloc_values = [
        value
        for value in sched_alloc.values
        if value.root_kind == "alloc" and value.source_kind == "res"
    ]
    assert len(alloc_values) > 0
    sched_alloc.partition(alloc_values[0], dim=0, kind=allo_d.Block, factor=2)
    sched_alloc.refresh()
    text_alloc = str(sched_alloc.module)
    assert "allo.part" in text_alloc
    assert re.search(r"memref.alloc\(\)\s*\{[^}]*allo.part", text_alloc) is not None


def test_example_outline_flatten_and_dump_smoke():
    ctx = ir.Context()
    sched_outline = Schedule.from_string(ctx, MLIR_ONE_LOOP_SCF)
    old_loop = _valid_handles(sched_outline, "kernel.L0")[0]
    sched_outline.outline("kernel.L0", func_name="outlined_kernel")
    assert old_loop.state == HandleState.CONSUMED
    call = _valid_handles(sched_outline, "kernel.L0::call")
    assert len(call) == 1
    assert sched_outline.active is not None
    assert sched_outline.active.instance_id == call[0].instance_id

    sched_flatten = Schedule.from_string(ctx, MLIR_TWO_LOOPS_AFFINE)
    sched_flatten.flatten(["kernel.L0", "kernel.L0.L0"])
    flat = _valid_handles(sched_flatten, "kernel.L0::flat")
    assert len(flat) == 1

    state_text = sched_flatten.dump_state(include_meta=True, last_n_effects=1)
    assert "=== Sched State ===" in state_text
    assert "effect_log" in state_text
    assert "kernel.L0::flat" in state_text
