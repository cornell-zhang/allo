# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import Stateful, int32


def _functions(schedule):
    return {
        op.attributes["sym_name"].value: op
        for op in schedule.module.body.operations
        if op.operation.name == "func.func"
    }


def _pipelined(schedule):
    return {
        name
        for name, function in _functions(schedule).items()
        if "pipeline_ii" in str(function)
    }


def test_source_numeric_suffix_does_not_collide_with_auto_id():
    def bump[N](x: int32) -> int32:
        return x + N

    def bump_1(x: int32) -> int32:
        return x + 100

    def top(x: int32) -> int32:
        return bump[1](x) + bump[2](x) + bump[1](x) + bump_1(x)

    # Repeated calls reuse an ID; a new compilation starts numbering at zero.
    for _ in range(2):
        s = allo.customize(top)
        assert set(_functions(s)) == {"top", "bump__0", "bump__1", "bump_1"}
        module = s.build(target="llvm")
        for value in [-4, 0, 5]:
            assert module(value) == 4 * value + 104


@pytest.mark.parametrize("label", ["", "0", "_fn_x", "clone_0"])
def test_unicode_function_and_underscore_label_boundaries(label):
    def _π_[Offset](A: int32[4], B: int32[4]):
        for i in allo.grid(4, name="update"):
            B[i] = A[i] + Offset

    def top(A: int32[4], B: int32[4]):
        _π_[7, label](A, B)

    schedule = allo.customize(top)
    expected = "_π___id_" + label
    assert set(_functions(schedule)) == {"top", expected}
    assert len(schedule.get_functions(_π_)) == 1
    loops = schedule.get_loops(schedule.get_functions(_π_)[0])
    schedule.pipeline(loops.update.i)
    assert _pipelined(schedule) == {expected}
    A = np.array([-4, 0, 2, 19], dtype=np.int32)
    B = np.zeros(4, dtype=np.int32)
    schedule.build(target="llvm")(A, B)
    np.testing.assert_array_equal(B, A + 7)


@pytest.mark.parametrize("label", ["a__b", "é", "a-b", "A\n"])
def test_labels_reject_non_ascii_and_separator_characters(label, capsys):
    def worker[Offset](x: int32) -> int32:
        return x + Offset

    def top(x: int32) -> int32:
        return worker[1, label](x)

    with pytest.raises((RuntimeError, ValueError, SystemExit)) as error:
        allo.customize(top)
    diagnostic = str(error.value) + capsys.readouterr().out
    assert "label" in diagnostic.lower()


def test_source_function_name_cannot_contain_the_field_separator(capsys):
    def worker__x(x: int32) -> int32:
        return x + 1

    with pytest.raises((RuntimeError, ValueError, SystemExit)) as error:
        allo.customize(worker__x)
    diagnostic = str(error.value) + capsys.readouterr().out
    assert "worker__x" in diagnostic


def _callees(function):
    return [
        op.callee.value
        for op in function.entry_block.operations
        if op.operation.name == "func.call"
    ]


@pytest.mark.parametrize(
    "parent_label, override_label",
    [("A", "B"), ("A", "")],
)
def test_override_propagates_through_helpers_and_restores_sibling_label(
    parent_label,
    override_label,
):
    def leaf[N](x: int32) -> int32:
        return x + N

    def bridge(x: int32) -> int32:
        return leaf[4](x)

    def helper(x: int32) -> int32:
        return bridge(x)

    def parent(x: int32) -> int32:
        first = helper(x)
        overridden = helper[override_label](x)
        last = helper(x)
        return first + overridden + last

    def top(x: int32) -> int32:
        return parent[parent_label](x) + leaf[4, parent_label](x)

    schedule = allo.customize(top)
    labels = {parent_label, override_label}
    assert len(schedule.get_functions(parent)) == 1
    assert _callees(schedule.get_functions(parent)[0]) == [
        f"helper__id_{parent_label}",
        f"helper__id_{override_label}",
        f"helper__id_{parent_label}",
    ]
    for function, child in [(helper, "bridge"), (bridge, "leaf"), (leaf, None)]:
        handles = schedule.get_functions(function)
        assert sorted(handle.name.value for handle in handles) == sorted(
            f"{function.__name__}__id_{label}" for label in labels
        )
        if child is not None:
            for handle in handles:
                label = handle.name.value.split("__id_", 1)[1]
                assert _callees(handle) == [f"{child}__id_{label}"]
    module = schedule.build(target="llvm")
    for value in [-4, 0, 5]:
        assert module(value) == 4 * value + 16


@pytest.mark.parametrize("direct_first", [False, True])
def test_inherited_and_explicit_labels_reject_incompatible_parameters(
    direct_first,
    capsys,
):
    def leaf[N](x: int32) -> int32:
        return x + N

    def helper(x: int32) -> int32:
        return leaf[4](x)

    def parent(x: int32) -> int32:
        return helper(x)

    def inherited_first(x: int32) -> int32:
        return parent["A"](x) + leaf[8, "A"](x)

    def explicit_first(x: int32) -> int32:
        return leaf[8, "A"](x) + parent["A"](x)

    with pytest.raises(SystemExit):
        allo.customize(explicit_first if direct_first else inherited_first)
    assert "conflicting parameters" in capsys.readouterr().out


def test_label_closure_value_conflict_is_an_error(capsys):
    def parent[Offset](x: int32) -> int32:
        def leaf(y: int32) -> int32:
            return y + Offset

        return leaf["A"](x)

    def top(x: int32) -> int32:
        return parent[1](x) + parent[2](x)

    with pytest.raises(SystemExit):
        allo.customize(top)
    assert "conflicting parameters" in capsys.readouterr().out


def test_factory_callable_identity_and_aliases_survive_composition():
    def make_worker(offset):
        def worker(A: int32[4], B: int32[4]):
            for i in allo.grid(4, name="update"):
                B[i] = A[i] + offset

        return worker

    first = make_worker(1)
    alias = first
    second = make_worker(10)

    def top(A: int32[4], B: int32[4], C: int32[4], D: int32[4]):
        first["same"](A, B)
        alias["same"](A, C)
        second["same"](A, D)

    schedule = allo.customize(top)
    assert set(_functions(schedule)) == {
        "top",
        "worker__id_same",
        "worker__def_1__id_same",
    }
    assert len(schedule.get_functions(first)) == 1
    assert schedule.get_functions(alias) == schedule.get_functions(first)
    assert len(schedule.get_functions(second)) == 1
    source = allo.customize(second)
    source.pipeline("i")
    schedule.compose(source, id="same")
    assert _pipelined(schedule) == {"worker__def_1__id_same"}
    A = np.array([-10, 0, 4, 8], dtype=np.int32)
    B, C, D = [np.zeros_like(A) for _ in range(3)]
    schedule.build(target="llvm")(A, B, C, D)
    np.testing.assert_array_equal(B, A + 1)
    np.testing.assert_array_equal(C, A + 1)
    np.testing.assert_array_equal(D, A + 10)


def test_contextual_nongeneric_instances_have_separate_automatic_ids():
    def leaf(A: int32[4], B: int32[4]):
        for i in allo.grid(4, name="update"):
            B[i] = A[i] + 1

    def parent[Offset](A: int32[4], B: int32[4]):
        leaf(A, B)

    def top(A: int32[4], B: int32[4], C: int32[4]):
        parent[1](A, B)
        parent[2](A, C)

    schedule = allo.customize(top)
    assert len(schedule.get_functions(leaf)) == 2
    assert {"leaf__0", "leaf__1"}.issubset(_functions(schedule))
    source = allo.customize(leaf)
    source.pipeline("i")
    schedule.compose(source, id=1)
    assert _pipelined(schedule) == {"leaf__1"}
    A = np.arange(4, dtype=np.int32)
    B, C = np.zeros_like(A), np.zeros_like(A)
    schedule.build(target="llvm")(A, B, C)
    np.testing.assert_array_equal(B, A + 1)
    np.testing.assert_array_equal(C, A + 1)


def _composition_kernels():
    def worker[Offset](A: int32[4], B: int32[4]):
        for i in allo.grid(4, name="update"):
            B[i] = A[i] + Offset

    def top(
        A: int32[4],
        B0: int32[4],
        B1: int32[4],
        S0: int32[4],
        Sempty: int32[4],
        SA: int32[4],
    ):
        worker[1](A, B0)
        worker[2, "0"](A, S0)
        worker[3, ""](A, Sempty)
        worker[4, "A"](A, SA)
        worker[5](A, B1)

    return worker, top


@pytest.mark.parametrize(
    "selector,offset,selected",
    [
        (None, 1, "worker__0"),
        (0, 1, "worker__0"),
        (1, 5, "worker__1"),
        ("0", 2, "worker__id_0"),
        ("", 3, "worker__id_"),
        ("A", 4, "worker__id_A"),
    ],
)
def test_composition_preserves_selector_types_and_targets_only_matches(
    selector, offset, selected
):
    worker, top = _composition_kernels()
    source = allo.customize(worker, instantiate=[offset])
    assert set(_functions(source)) == {"worker"}
    source.pipeline("i")
    schedule = allo.customize(top)
    assert set(_functions(schedule)) == {
        "top",
        "worker__0",
        "worker__1",
        "worker__id_0",
        "worker__id_",
        "worker__id_A",
    }
    schedule.compose(source, id=selector)
    assert _pipelined(schedule) == {selected}
    A = np.array([-3, 0, 5, 11], dtype=np.int32)
    outputs = [np.zeros_like(A) for _ in range(5)]
    schedule.build(target="llvm")(A, *outputs)
    for output, value in zip(outputs, (1, 5, 2, 3, 4)):
        np.testing.assert_array_equal(output, A + value)


@pytest.mark.parametrize("selector", [2, "missing"])
def test_missing_composition_selector_raises_without_fallback(selector):
    worker, top = _composition_kernels()
    source = allo.customize(worker, instantiate=[1])
    source.pipeline("i")
    schedule = allo.customize(top)
    with pytest.raises(RuntimeError):
        schedule.compose(source, id=selector)
    assert not _pipelined(schedule)


@pytest.mark.parametrize("selector", [False, True])
def test_boolean_composition_selector_cannot_select_automatic_id(selector):
    worker, top = _composition_kernels()
    source = allo.customize(worker, instantiate=[1])
    source.pipeline("i")
    schedule = allo.customize(top)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        schedule.compose(source, id=selector)
    assert not _pipelined(schedule)


def test_default_composition_does_not_choose_a_labeled_instance():
    def worker[Offset](A: int32[4], B: int32[4]):
        for i in allo.grid(4, name="update"):
            B[i] = A[i] + Offset

    def top(A: int32[4], B: int32[4]):
        worker[1, ""](A, B)

    source = allo.customize(worker, instantiate=[1])
    source.pipeline("i")
    schedule = allo.customize(top)
    with pytest.raises(RuntimeError):
        schedule.compose(source)
    assert not _pipelined(schedule)


def test_compose_default_prefers_ordinary_instance():
    def leaf(a: int32[4]):
        for i in range(4):
            a[i] += 1

    def wrapper[N](a: int32[4]):
        leaf(a)

    def top(a: int32[4]):
        leaf(a)
        wrapper[1](a)

    source = allo.customize(leaf)
    source.pipeline("i")
    destination = allo.customize(top)
    destination.compose(source)
    functions = {fn.name.value: str(fn) for fn in destination.get_functions(leaf)}
    assert "pipeline_ii" in functions["leaf"]
    assert "pipeline_ii" not in functions["leaf__0"]
    a = np.zeros(4, np.int32)
    destination.build(target="llvm")(a)
    np.testing.assert_array_equal(a, [2, 2, 2, 2])


def test_same_label_has_independent_generic_values_in_each_pe_scope():
    def helper_[Offset](A: int32[4], B: int32[4]):
        for i in allo.grid(4, name="update"):
            B[i] = A[i] + Offset

    @df.region()
    def top(A: int32[2, 2, 4], B: int32[2, 2, 4]):
        @df.kernel(mapping=[2, 2], args=[A, B])
        def worker_2(X: int32[2, 2, 4], Y: int32[2, 2, 4]):
            p, q = df.get_pid()
            helper_[p * 10 + q, "_fn_x"](X[p, q], Y[p, q])

    schedule = df.customize(top)
    expected = {
        f"worker_2__pe_{p}_{q}__fn_helper___id__fn_x"
        for p in range(2)
        for q in range(2)
    }
    assert expected.issubset(_functions(schedule))
    assert len(schedule.get_functions(helper_)) == 4
    source = allo.customize(helper_, instantiate=[0])
    source.pipeline("i")
    schedule.compose(source, id="_fn_x")
    assert _pipelined(schedule) == expected
    # The IR above checks composition; the simulator checks scoped execution.
    A = np.arange(16, dtype=np.int32).reshape(2, 2, 4)
    B = np.zeros_like(A)
    df.build(top, target="simulator")(A, B)
    np.testing.assert_array_equal(
        B, A + np.array([[0, 1], [10, 11]], dtype=np.int32)[:, :, None]
    )


def test_unfold_refreshes_handles_and_preserves_other_callers():
    def leaf(A: int32[4], B: int32[4]):
        for k in allo.grid(4, name="update"):
            B[k] = A[k] * 3 - 2

    def top(A: int32[2, 2, 4], B: int32[2, 2, 4], C: int32[1, 4]):
        for i, j in allo.grid(2, 2, name="tile"):
            leaf(A[i, j], B[i, j])
        leaf(A[0, 0], C[0])

    unfolded = allo.customize(top)
    untouched = allo.customize(top)
    assert len(unfolded.get_functions(leaf)) == 1
    unfolded.unfold("tile", [0])
    assert set(_functions(unfolded)) == {
        "top",
        "leaf",
        "leaf__clone_0",
        "leaf__clone_1",
    }
    for band_name, _ in list(unfolded.get_loops()):
        unfolded.unfold(band_name, [0])
    expected = {"leaf"} | {f"leaf__clone_{i}" for i in range(2, 6)}
    assert set(_functions(unfolded)) == {"top"} | expected
    assert set(_functions(untouched)) == {"top", "leaf"}

    # Refresh lookup after unfolding; all handles must refer to live functions.
    functions = unfolded.get_functions(leaf)
    assert len(functions) == len(expected)
    assert {function.name.value for function in functions} == expected
    for function in functions:
        loops = unfolded.get_loops(function)
        unfolded.pipeline(loops.update.k)
    assert _pipelined(unfolded) == expected
    assert not _pipelined(untouched)

    source = allo.customize(leaf)
    source.unroll("k", factor=2)
    unfolded.compose(source)
    assert all("unroll = 2" in str(fn) for fn in unfolded.get_functions(leaf))
    assert "unroll =" not in str(untouched.get_functions(leaf)[0])
    A = np.arange(-8, 8, dtype=np.int32).reshape(2, 2, 4)
    for schedule in (unfolded, untouched):
        B, C = np.zeros_like(A), np.zeros((1, 4), dtype=np.int32)
        schedule.build(target="llvm")(A, B, C)
        np.testing.assert_array_equal(B, A * 3 - 2)
        np.testing.assert_array_equal(C[0], A[0, 0] * 3 - 2)


def test_composition_keeps_each_labeled_state_and_nested_state():
    label = "A"

    def leaf[N](x: int32) -> int32:
        count: int32 @ Stateful = 0
        count += N
        return count + x

    def worker[N](a: int32[2], b: int32[2]):
        visits: int32 @ Stateful = 0
        visits += 1
        for k in range(2):
            b[k] = leaf[N](a[k]) + visits

    def top(a: int32[2], b: int32[2], c: int32[2]):
        worker[1, label](a, b)
        worker[10, "other"](a, c)

    source = allo.customize(worker, instantiate=[1])
    source.pipeline("k")
    destination = allo.customize(top)
    destination.compose(source, id=label)
    destination.compose(source, id="other")
    globals_before = {
        op.attributes["sym_name"].value: str(op)
        for op in destination.module.body.operations
        if op.name == "memref.global"
    }
    assert len(globals_before) == 4
    # Reapplying a schedule must not duplicate, reset, or join persistent globals.
    destination.compose(source, id=label)
    assert globals_before == {
        op.attributes["sym_name"].value: str(op)
        for op in destination.module.body.operations
        if op.name == "memref.global"
    }
    assert destination.module.operation.verify()
    module = destination.build(target="llvm")
    inputs = np.array([3, 8], dtype=np.int32)
    left, right = np.zeros(2, np.int32), np.zeros(2, np.int32)
    module(inputs, left, right)
    np.testing.assert_array_equal(left, inputs + [2, 3])
    np.testing.assert_array_equal(right, inputs + [11, 21])
    module(inputs, left, right)
    np.testing.assert_array_equal(left, inputs + [5, 6])
    np.testing.assert_array_equal(right, inputs + [32, 42])


def test_nested_generic_arguments_do_not_change_the_parent_instantiation():
    def leaf[N](x: int32) -> int32:
        return x + N

    def pair[N](x: int32, y: int32) -> int32:
        return leaf[N](x) + y

    def parent[N](x: int32) -> int32:
        return pair[100](leaf[2](x), leaf[3](x)) + leaf[N](x)

    def top(x: int32) -> int32:
        return parent[7](x) + parent[11](x)

    schedule = allo.customize(top)
    assert schedule.module.operation.verify()
    assert schedule.build(target="llvm")(5) == 258


if __name__ == "__main__":
    pytest.main([__file__])
