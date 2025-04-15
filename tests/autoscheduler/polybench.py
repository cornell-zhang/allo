# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import int32, float32
import allo.ir.types as T
import allo
from allo.customize import Schedule
import numpy as np
import os
import json
from typing import Dict, Any, Callable, Tuple, List
from examples.polybench.three_mm import kernel_3mm, three_mm_np
from examples.polybench.two_mm import kernel_2mm, two_mm_np
from examples.polybench.atax import kernel_atax, atax_np
from examples.polybench.bicg import kernel_bicg, bicg_np
from examples.polybench.gemm import kernel_gemm, gemm_np
from examples.polybench.gesummv import kernel_gesummv, gesummv_np
from examples.polybench.mvt import kernel_mvt, mvt_np


polybench_registry: dict[str, tuple[Callable, Callable, Callable]] = {}


def add_benchmark(name: str, sch_fn: Callable, np_fn: Callable, gen_fn: Callable):
    """Add a benchmark to the registry"""
    polybench_registry[name] = (sch_fn, np_fn, gen_fn)


def get_polybench(
    test_name: str, size: str = "small", concrete_type=float32
) -> Tuple[Schedule, Tuple[np.ndarray, ...], np.ndarray]:
    """Get a polybench test case with inputs and reference output"""
    if test_name not in polybench_registry:
        raise ValueError(f"Unknown benchmark {test_name}")

    setting_path = os.path.join(
        os.path.dirname(__file__), "../../examples/polybench/psize.json"
    )
    with open(setting_path, "r") as fp:
        psize = json.load(fp)

    if test_name not in psize:
        raise ValueError(f"No problem size defined for {test_name}")
    if size not in psize[test_name]:
        valid_sizes = list(psize[test_name].keys())
        raise ValueError(
            f"Size {size} not defined for {test_name}. Valid sizes are: {valid_sizes}"
        )

    params = psize[test_name][size]
    param_values = list(params.values())
    sch_fn, np_fn, gen_fn = polybench_registry[test_name]

    sch = sch_fn(concrete_type, *param_values)
    inputs, ref_out = gen_fn(concrete_type, *param_values)

    return sch, inputs, ref_out


def three_mm(concrete_type, p, r, q, t, s):
    return allo.customize(kernel_3mm, instantiate=[concrete_type, p, q, r, s, t])


def two_mm(concrete_type, p, r, q, s, alpha=0.1, beta=0.5):
    return allo.customize(kernel_2mm, instantiate=[concrete_type, p, r, q, s])


def atax(concrete_type, m, n):
    return allo.customize(kernel_atax, instantiate=[concrete_type, m, n])


def bicg(concrete_type, m, n):
    return allo.customize(kernel_bicg, instantiate=[concrete_type, m, n])


def gemm(concrete_type, p, r, q, beta=0.1):
    return allo.customize(kernel_gemm, instantiate=[concrete_type, p, q, r])


def gesummv(concrete_type, n, alpha=1.0, beta=1.0):
    return allo.customize(kernel_gesummv, instantiate=[concrete_type, n])


def mvt(concrete_type, n):
    return allo.customize(kernel_mvt, instantiate=[concrete_type, n])


def gen_three_mm_inputs(concrete_type, p, r, q, t, s):
    """Generate inputs and reference output for three_mm benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (r, s)).astype(np_dtype)
    D = np.random.randint(-10, 10, (s, t)).astype(np_dtype)

    ref_out = three_mm_np(np.copy(A), np.copy(B), np.copy(C), np.copy(D))

    return (A, B, C, D), ref_out


def gen_two_mm_inputs(concrete_type, p, r, q, s, alpha=0.1, beta=0.5):
    """Generate inputs and reference output for two_mm benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (r, s)).astype(np_dtype)
    D = np.random.randint(-10, 10, (p, s)).astype(np_dtype)

    ref_out = two_mm_np(np.copy(A), np.copy(B), np.copy(C), np.copy(D), alpha, beta)

    return (A, B, C, D), ref_out


def gen_atax_inputs(concrete_type, m, n):
    """Generate inputs and reference output for atax benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (m, n)).astype(np_dtype)
    x = np.random.randint(-10, 10, n).astype(np_dtype)
    y = np.zeros(n, dtype=np_dtype)

    ref_out = atax_np(np.copy(A), np.copy(x))

    return (A, x, y), ref_out


def gen_bicg_inputs(concrete_type, m, n):
    """Generate inputs and reference output for bicg benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (n, m)).astype(np_dtype)
    A_copy = np.copy(A)

    s = np.zeros(m, dtype=np_dtype)
    q = np.zeros(n, dtype=np_dtype)
    p = np.random.randint(-10, 10, m).astype(np_dtype)
    r = np.random.randint(-10, 10, n).astype(np_dtype)

    s_copy, q_copy = np.copy(s), np.copy(q)
    s_out, q_out = bicg_np(np.copy(A), s_copy, q_copy, np.copy(p), np.copy(r))

    return (A, A_copy, s, q, p, r), (q_out, s_out)


def gen_gemm_inputs(concrete_type, p, r, q, beta=0.1):
    """Generate inputs and reference output for gemm benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (p, q)).astype(np_dtype)
    B = np.random.randint(-10, 10, (q, r)).astype(np_dtype)
    C = np.random.randint(-10, 10, (p, r)).astype(np_dtype)

    ref_out = gemm_np(np.copy(A), np.copy(B), np.copy(C), beta)

    return (A, B, C), ref_out


def gen_gesummv_inputs(concrete_type, n, alpha=1.0, beta=1.0):
    """Generate inputs and reference output for gesummv benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    B = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    x = np.random.randint(-10, 10, n).astype(np_dtype)

    y = np.zeros(n, dtype=np_dtype)
    ref_out = gesummv_np(np.copy(A), np.copy(B), np.copy(x), y, alpha, beta)

    return (A, B, x), ref_out


def gen_mvt_inputs(concrete_type, n):
    """Generate inputs and reference output for mvt benchmark"""
    np_dtype = np.float32 if concrete_type == float32 else np.int32

    A = np.random.randint(-10, 10, (n, n)).astype(np_dtype)
    A_copy = np.copy(A)
    y1 = np.random.randint(-10, 10, n).astype(np_dtype)
    y2 = np.random.randint(-10, 10, n).astype(np_dtype)
    x1 = np.random.randint(-10, 10, n).astype(np_dtype)
    x2 = np.random.randint(-10, 10, n).astype(np_dtype)
    x1_out = np.zeros(n, dtype=np_dtype)
    x2_out = np.zeros(n, dtype=np_dtype)

    x1_ref, x2_ref = np.copy(x1), np.copy(x2)
    expected_x1, expected_x2 = mvt_np(
        np.copy(A), x1_ref, x2_ref, np.copy(y1), np.copy(y2)
    )

    return (A, A_copy, y1, y2, x1, x2, x1_out, x2_out), (expected_x1, expected_x2)


# Register all benchmarks
add_benchmark("three_mm", three_mm, three_mm_np, gen_three_mm_inputs)
add_benchmark("two_mm", two_mm, two_mm_np, gen_two_mm_inputs)
add_benchmark("gemm", gemm, gemm_np, gen_gemm_inputs)
add_benchmark("mvt", mvt, mvt_np, gen_mvt_inputs)
add_benchmark("atax", atax, atax_np, gen_atax_inputs)
add_benchmark("bicg", bicg, bicg_np, gen_bicg_inputs)
add_benchmark("gesummv", gesummv, gesummv_np, gen_gesummv_inputs)
