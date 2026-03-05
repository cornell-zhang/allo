# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

from allo.bindings import ir, allo as allo_d, transform as tran_d
from allo.sched import HandleState, Sched


MLIR_ONE_LOOP = """
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


MLIR_ONE_LOOP_LEGACY_IDENTIFIER = """
module {
  func.func @kernel(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      memref.store %c0_i32, %arg0[%i] : memref<8xi32>
    } {sym_name = "legacy.loop"}
    return
  }
}
"""


MLIR_TWO_LOOPS = """
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


MLIR_THREE_NESTED_LOOPS = """
module {
  func.func @kernel() {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        scf.for %k = %c0 to %c8 step %c1 { }
      }
    }
    return
  }
}
"""


MLIR_AFFINE_TWO_LOOPS = """
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


MLIR_LOOP_RESULT_MEMREF = """
module {
  func.func @kernel(%arg0: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %arg0) -> (memref<8xi32>) {
      scf.yield %acc : memref<8xi32>
    }
    memref.copy %out, %arg0 : memref<8xi32> to memref<8xi32>
    return
  }
}
"""


def _iter_nodes(node):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _find_valid_by_identifier(sched: Sched, identifier: str):
    return [
        h
        for h in sched.handles
        if h.identifier == identifier and h.state == HandleState.VALID
    ]


def test_parse_proxy_tree_and_identifier_completion_core():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_THREE_NESTED_LOOPS)
    stats = ir.complete_op_identifiers(mod)
    root = ir.parse_proxy_tree(mod)

    assert root.op_identifier == "__allo_module__"
    assert stats["visited"] > 0
    assert stats["assigned"] > 0

    all_nodes = list(_iter_nodes(root))
    all_ids = [node.op_identifier for node in all_nodes if node.op_identifier != ""]
    loop_ids = [node.op_identifier for node in all_nodes if node.op_kind == "scf.for"]
    assert len(all_ids) == len(set(all_ids))
    assert loop_ids == ["kernel.L0", "kernel.L0.L0", "kernel.L0.L0.L0"]


def test_parse_proxy_tree_exposes_value_metadata_core():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_LOOP_RESULT_MEMREF)
    ir.complete_op_identifiers(mod)
    root = ir.parse_proxy_tree(mod)

    all_values = []
    for node in _iter_nodes(root):
        all_values.extend(list(node.values))

    func_arg = next(
        value for value in all_values if value.value_identifier == "kernel:arg0"
    )
    assert func_arg.source_kind == "arg"
    assert func_arg.is_memref
    assert func_arg.root_kind == "block_arg"


def test_sched_constructors_from_string_and_module():
    ctx = ir.Context()
    sched_from_string = Sched.from_string(ctx, MLIR_ONE_LOOP)
    assert len(sched_from_string.handles) > 0

    ctx2 = ir.Context()
    ctx2.load_dialects()
    mod = ir.parse_from_string(ctx2, MLIR_ONE_LOOP)
    sched_from_module = Sched.from_module(mod)
    assert len(sched_from_module.handles) > 0


def test_split_pipeline_lifecycle_core():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_ONE_LOOP)
    sched.select("kernel.L0")
    original = sched.active
    assert original is not None

    sched.split(factor=4)
    assert original.state == HandleState.CONSUMED
    assert len(_find_valid_by_identifier(sched, "kernel.L0::outer")) == 1
    assert len(_find_valid_by_identifier(sched, "kernel.L0::inner")) == 1

    inner = _find_valid_by_identifier(sched, "kernel.L0::inner")[0]
    sched.pipeline("kernel.L0::inner", ii=2)
    assert inner.state == HandleState.VALID
    assert inner.meta["pipeline_ii"] == 2


def test_stale_descendants_on_split_core():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_TWO_LOOPS)
    nested = _find_valid_by_identifier(sched, "kernel.L0.L0")[0]
    sched.split("kernel.L0", factor=2)
    assert nested.state == HandleState.STALE


def test_refresh_from_module_does_not_rewrite_identifiers():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_ONE_LOOP_LEGACY_IDENTIFIER)
    ir.complete_op_identifiers(mod)

    sched = Sched.from_module(mod)
    assert len(_find_valid_by_identifier(sched, "legacy.loop")) == 1

    sched.refresh_from_module()
    assert len(_find_valid_by_identifier(sched, "legacy.loop")) == 1
    assert len(_find_valid_by_identifier(sched, "kernel.L0")) == 0


def test_finalize_transform_removes_non_symbol_identifiers_only():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_ONE_LOOP_LEGACY_IDENTIFIER)
    ir.complete_op_identifiers(mod, overwrite=True)
    before = str(mod)
    assert "legacy.loop" not in before
    assert "module @__allo_module__" in before
    assert "func.func @kernel" in before

    stats = ir.finalize_transform(mod)
    after = str(mod)
    assert stats["visited"] > 0
    assert stats["removed"] > 0
    assert stats["kept_symbol"] > 0
    assert "module @__allo_module__" not in after
    assert "func.func @kernel" in after
    assert "sym_name" not in after


def test_commit_always_rewrites_identifiers_once():
    ctx = ir.Context()
    ctx.load_dialects()
    mod = ir.parse_from_string(ctx, MLIR_ONE_LOOP_LEGACY_IDENTIFIER)
    ir.complete_op_identifiers(mod)

    sched = Sched.from_module(mod)
    sched.pipeline("legacy.loop", ii=2)
    sched.commit()

    assert len(_find_valid_by_identifier(sched, "legacy.loop")) == 0
    assert len(_find_valid_by_identifier(sched, "kernel.L0")) == 1


def test_reorder_updates_hierarchy_for_stale_propagation():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_THREE_NESTED_LOOPS)

    sched.to_affine("kernel.L0")
    sched.to_affine("kernel.L0.L0")
    sched.to_affine("kernel.L0.L0.L0")
    sched.reorder(["kernel.L0", "kernel.L0.L0", "kernel.L0.L0.L0"], [2, 1, 0])

    sched.unroll("kernel.L0", factor=2, tag_only=False)
    sched.unroll("kernel.L0.L0", factor=2, tag_only=True)

    middle = _find_valid_by_identifier(sched, "kernel.L0.L0")
    assert len(middle) == 1
    assert middle[0].state == HandleState.VALID
    assert middle[0].meta["unroll_factor"] == 2


def test_to_affine_keeps_descendants_valid():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_TWO_LOOPS)

    sched.to_affine("kernel.L0")
    inner = _find_valid_by_identifier(sched, "kernel.L0.L0")[0]
    sched.pipeline("kernel.L0.L0", ii=2)

    assert inner.state == HandleState.VALID
    assert inner.meta["pipeline_ii"] == 2


def test_partition_handles_block_argument_root():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_ONE_LOOP)

    sched.partition("kernel:arg0", dim=0)
    sched.commit()
    text = str(sched.module)
    assert "allo.part" in text
    assert re.search(r"%arg0: memref<8xi32>\s*\{[^}]*allo.part", text) is not None


def test_partition_rejects_non_memref_value():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_LOOP_RESULT_MEMREF)
    with pytest.raises(ValueError, match="not memref-typed"):
        sched.partition("kernel.L0:arg0", dim=0)


def test_dump_methods_include_internal_state():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_LOOP_RESULT_MEMREF)
    sched.select("kernel.L0")
    sched.pipeline(ii=2)

    state_text = sched.dump_state(include_meta=True, last_n_effects=1)
    assert "=== Sched State ===" in state_text
    assert "active=kernel.L0" in state_text
    assert "dirty=True" in state_text
    assert "kernel.L0:res0" in state_text
    assert "pipeline" in state_text

    indexes_text = sched.dump_indexes()
    assert "by_identifier:" in indexes_text
    assert "by_value_identifier:" in indexes_text

    effects_text = sched.dump_effect_log()
    assert "'op': 'pipeline'" in effects_text


def test_dev_rebind_payloads_smoke():
    ctx = ir.Context()
    sched = Sched.from_string(ctx, MLIR_AFFINE_TWO_LOOPS)

    resolved = sched.dev_resolve_targets(
        ["kernel.L0", "kernel.L0.L0"],
        action="dev_reorder_like",
        require_loop=True,
        deduplicate=False,
    )
    payloads = sched.dev_materialize_payloads(resolved)
    merged = tran_d.MergeHandlesOp.create(
        sched.builder, payloads, deduplicate=False
    ).get_result_at(0)
    allo_d.LoopReorderOp.create(sched.builder, merged, [1, 0])
    split_back = tran_d.SplitHandleOp.create(sched.builder, merged, len(resolved))
    sched._mark_dirty()

    sched.dev_rebind_payloads(
        resolved,
        [split_back.get_result_at(0), split_back.get_result_at(1)],
        select=0,
        meta_updates={"dev_reordered": True},
    )
    sched.dev_finalize_transform(
        transform="dev_reorder_like",
        log={"targets": [p.identifier for p in resolved], "order": [1, 0]},
    )

    assert resolved[0].state == HandleState.VALID
    assert resolved[1].state == HandleState.VALID
    assert resolved[0].meta["dev_reordered"] is True
