# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from gurobipy import GurobiError
import numpy as np
import pytest
from allo.ir.types import float32, int32
from allo.autoscheduler.passes import dataflow_optimization_pass, DEBUG_POINTS
from allo.autoscheduler.config import AutoschedulerConfig
from tests.autoscheduler.polybench import get_polybench
import allo
from allo.backend.hls import is_available

kinds = ["graph", "node", "combined"]

MODE = "sw_emu"


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", kinds)
def test_simple(debug_point, kind):
    def simple(v: int32[10, 10]) -> int32[10, 10]:
        def stageA(v: int32[10, 10]) -> int32[10, 10]:
            A: int32[10, 10]
            for j in range(10):
                for i in range(10):
                    A[i, j] = i + j + v[i, j]
            return A

        def stageB(A: int32[10, 10]) -> int32[10, 10]:
            B: int32[10, 10]
            for i in range(10):
                for j in range(10):
                    B[i, j] = A[i, j] + 1
            return B

        A = stageA(v)
        B = stageB(A)
        return B

    s = allo.customize(simple)

    # Create AutoschedulerConfig with the specified parameters
    cfg = AutoschedulerConfig.builder().with_debug_point(debug_point).with_kind(kind)
    optimized_schedule = dataflow_optimization_pass(s, cfg)

    expected = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            expected[i, j] = i + j + 1

    if debug_point is not None:
        print(optimized_schedule.module)
        mod = optimized_schedule.build()
        input = np.zeros((10, 10), dtype=np.int32)
        np.testing.assert_allclose(mod(input), expected)

    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_simple.prj", wrap_io=True
        )
        input = np.zeros((10, 10), dtype=np.int32)

        output = np.zeros((10, 10))
        mod(input, output)
        np.testing.assert_allclose(output, expected)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", kinds)
def test_conv2d(debug_point, kind):
    IR, IC = 30, 30  # Input rows and columns
    FR, FC = 6, 6  # Filter rows and columns
    OR, OC = 25, 25  # Output rows and columns

    def conv2D(A: float32[IR, IC], B: float32[FR, FC]) -> float32[OR, OC]:
        C: float32[OR, OC] = 0.0
        for y, x in allo.grid(OR, OC):  # these are the output dimensions
            for r, c in allo.reduction(FR, FC):  # this is the filter dimensions
                C[y, x] += A[y + r, x + c] * B[r, c]
        return C

    s = allo.customize(conv2D)
    # Create AutoschedulerConfig with the specified parameters
    cfg = (
        AutoschedulerConfig.builder()
        .with_debug_point(debug_point)
        .with_kind(kind)
        .with_verbose(True)
    )
    try:
        optimized_schedule = dataflow_optimization_pass(s, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e

    input_A = np.random.rand(IR, IC).astype(np.float32)
    input_B = np.random.rand(FR, FC).astype(np.float32)
    expected = np.zeros((OR, OC), dtype=np.float32)
    for y in range(OR):
        for x in range(OC):
            for r in range(FR):
                for c in range(FC):
                    expected[y, x] += input_A[y + r, x + c] * input_B[r, c]
    expected = expected.astype(np.float32)
    print(optimized_schedule.module)

    if debug_point is not None:
        mod = optimized_schedule.build()
        output = mod(input_A, input_B)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_conv2d.prj", wrap_io=True
        )
        output = np.zeros((OR, OC), dtype=np.float32)
        mod(input_A, input_B, output)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", kinds)
def test_three_mm(debug_point, kind):
    schedule, inputs, expected = get_polybench(
        "three_mm", size="small", concrete_type=float32
    )
    try:
        # Create AutoschedulerConfig with the specified parameters
        cfg = (
            AutoschedulerConfig.builder()
            .with_debug_point(debug_point)
            .with_kind(kind)
            .with_verbose(True)
        )
        optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e

    if debug_point is not None:
        mod = optimized_schedule.build()
        np.testing.assert_allclose(mod(*inputs), expected, rtol=1e-5, atol=1e-5)

    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_three_mm.prj", wrap_io=False
        )
        output = np.zeros_like(expected)
        mod(*inputs, output)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", kinds)
def test_two_mm(debug_point, kind):
    schedule, inputs, expected = get_polybench(
        "two_mm", size="medium", concrete_type=float32
    )
    try:
        # Create AutoschedulerConfig with the specified parameters
        cfg = (
            AutoschedulerConfig.builder()
            .with_debug_point(debug_point)
            .with_kind(kind)
            .with_verbose(True)
        )
        optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )

    if debug_point is not None:
        mod = optimized_schedule.build()
        np.testing.assert_allclose(mod(*inputs), expected, rtol=1e-5, atol=1e-5)

    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_two_mm.prj"
        )
        output = np.zeros_like(expected)
        mod(*inputs, output)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", kinds)
def test_atax(debug_point, kind):
    schedule, inputs, expected = get_polybench(
        "atax", size="small", concrete_type=float32
    )
    try:
        # Create AutoschedulerConfig with the specified parameters
        cfg = (
            AutoschedulerConfig.builder()
            .with_debug_point(debug_point)
            .with_kind(kind)
            .with_verbose(True)
        )
        optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e

    A, x, y = inputs
    if debug_point is not None:
        mod = optimized_schedule.build()
        mod(A, x, y)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    elif is_available("vitis_hls"):
        print(optimized_schedule.module)
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_atax.prj"
        )
        print(mod.hls_code)
        mod(A, x, y)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


# @pytest.mark.parametrize("debug_point", DEBUG_POINTS)
# def test_bicg(debug_point):
#     schedule, inputs, expected = get_polybench(
#         "bicg", size="small", concrete_type=float32
#     )
#     cfg = AutoschedulerConfig.builder().with_debug_point(debug_point)
#     optimized_schedule = dataflow_optimization_pass(schedule, cfg)
#     mod = optimized_schedule.build()

#     A, A_copy, s, q, p, r = inputs
#     q_out = np.zeros_like(q)
#     s_out = np.zeros_like(s)
#     mod(A, A_copy, p, r, q_out, s_out)

#     expected_q, expected_s = expected
#     np.testing.assert_allclose(q_out, expected_q)
#     np.testing.assert_allclose(s_out, expected_s)


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", ["graph", "node"])
def test_gemm(debug_point, kind):
    schedule, inputs, expected = get_polybench(
        "gemm", size="small", concrete_type=float32
    )
    try:
        # Create AutoschedulerConfig with the specified parameters
        cfg = (
            AutoschedulerConfig.builder().with_debug_point(debug_point).with_kind(kind)
        )
        optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e

    A, B, C = inputs
    if debug_point is not None:
        mod = optimized_schedule.build()
        output = np.zeros_like(expected)
        mod(A, B, C, output)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_gemm.prj"
        )
        output = np.zeros_like(expected)
        mod(A, B, C, output)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


@pytest.mark.parametrize("debug_point", DEBUG_POINTS)
@pytest.mark.parametrize("kind", ["graph", "node"])
def test_gesummv(debug_point, kind):
    schedule, inputs, expected = get_polybench(
        "gesummv", size="small", concrete_type=float32
    )
    try:
        # Create AutoschedulerConfig with the specified parameters
        cfg = (
            AutoschedulerConfig.builder().with_debug_point(debug_point).with_kind(kind)
        )
        optimized_schedule = dataflow_optimization_pass(schedule, cfg)
    except GurobiError as e:
        if "Model too large for size-limited license" in str(e):
            pytest.skip(
                "Skipping test: model too large for size-limited Gurobi license"
            )
        else:
            raise e

    A, B, x = inputs
    if debug_point is not None:
        mod = optimized_schedule.build()
        y = np.zeros_like(expected)
        mod(A, B, x, y)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    elif is_available("vitis_hls"):
        mod = optimized_schedule.build(
            target="vitis_hls", mode=MODE, project="test_gesummv.prj"
        )
        y = np.zeros_like(expected)
        mod(A, B, x, y)
        np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    else:
        pytest.skip("Skipping test: vitis_hls not available")


# @pytest.mark.parametrize("debug_point", DEBUG_POINTS)
# def test_mvt(debug_point):
#     schedule, inputs, expected = get_polybench(
#         "mvt", size="small", concrete_type=float32
#     )
#     cfg = AutoschedulerConfig.builder().with_debug_point(debug_point)
#     optimized_schedule = dataflow_optimization_pass(schedule, cfg)
#     mod = optimized_schedule.build()

#     A, A_copy, y1, y2, x1, x2, x1_out, x2_out = inputs
#     expected_x1, expected_x2 = expected
#     out_x1 = np.zeros_like(x1)
#     out_x2 = np.zeros_like(x2)
#     mod(A, A_copy, y1, y2, x1, x2, out_x1, out_x2)

#     np.testing.assert_allclose(out_x1, expected_x1, rtol=1e-5, atol=1e-5)
#     np.testing.assert_allclose(out_x2, expected_x2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
