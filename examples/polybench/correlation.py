# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def correlation_np(data, mean, stddev, corr, M, N, N_float, epsilon):
    # Compute mean
    for x in range(M):
        total = 0.0
        for k in range(N):
            total += data[k, x]
        mean[x] = total / N

    # Compute stddev
    for x in range(M):
        variance = 0.0
        for m in range(N):
            variance += (data[m, x] - mean[x]) * (data[m, x] - mean[x])
        stddev[x] = np.sqrt(variance / N_float)
        # This is to avoid a division by zero situation
        if stddev[x] <= epsilon:
            stddev[x] = 1.0

    # Center and reduce the column vectors.
    for x in range(N):
        for y in range(M):
            data[x, y] -= mean[y]
            data[x, y] /= np.sqrt(N_float) * stddev[y]

    # Calculate the m * m correlation matrix.
    for i in range(M - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, M):
            corr[i, j] = 0.0
            for k in range(N):
                corr[i, j] += data[k, i] * data[k, j]
            corr[j, i] = corr[i, j]

    corr[M - 1, M - 1] = 1.0
    return data, mean, stddev, corr


def compute_mean[
    T: (float32, int32), M: int32, N: int32
](data: "T[N, M]", mean: "T[M]"):
    for x in allo.grid(M):
        total: T = 0.0
        for k in allo.grid(N):
            total += data[k, x]
        mean[x] = total / N


def compute_stddev[
    T: (float32, int32), M: int32, N: int32
](data: "T[N, M]", mean: "T[M]", mean_passed_on: "T[M]", stddev: "T[M]"):
    for x in allo.grid(M):
        variance: T = 0.0
        for m in allo.grid(N):
            variance += (data[m, x] - mean[x]) * (data[m, x] - mean[x])
        stddev[x] = allo.sqrt(variance / N_float)
        mean_passed_on[x] = mean[x]
        # This is to avoid a division by zero situation
        if stddev[x] <= epsilon:
            stddev[x] = 1.0


def center_reduce[
    T: (float32, int32), M: int32, N: int32
](data: "T[N, M]", data_out: "T[N, M]", mean: "T[M]", stddev: "T[M]"):
    for x in allo.grid(N):
        for y in allo.grid(M):
            d: T = data[x, y]
            d -= mean[y]
            d /= allo.sqrt(N_float) * stddev[y]
            data_out[x, y] = d


def compute_corr[
    T: (float32, int32), M: int32, N: int32
](data: "T[N, M]", corr: "T[M, M]"):
    for i in range(M - 1):
        corr[i, i] = 1.0
        for j in range(M):
            if j > i:
                corr_v: T = 0.0
                for k in range(N):
                    corr_v += data[k, i] * data[k, j]
                corr[j, i] = corr_v
                corr[i, j] = corr_v

    corr[M - 1, M - 1] = 1.0


def kernel_correlation[
    T: (float32, int32), M: int32, N: int32
](
    data_mean: "T[N, M]",
    data_stddev: "T[N, M]",
    data_for_center: "T[N, M]",
    corr: "T[M, M]",
):
    mean: T[M] = 0.0
    mean_passed_on: T[M] = 0.0
    stddev: T[M] = 0.0
    compute_mean(data_mean, mean)
    compute_stddev(data_stddev, mean, mean_passed_on, stddev)
    data_centered: T[N, M] = 0.0
    center_reduce(data_for_center, data_centered, mean_passed_on, stddev)
    compute_corr(data_centered, corr)


# Global constants for kernel functions
N_float = 0.0
epsilon = 0.0


def correlation(concrete_type, M, N, N_float_val, epsilon_val):
    global N_float, epsilon
    N_float = N_float_val
    epsilon = epsilon_val

    s0 = allo.customize(compute_mean, instantiate=[concrete_type, M, N])
    s0.pipeline("x")
    s0.partition(s0.data, dim=1)

    s1 = allo.customize(compute_stddev, instantiate=[concrete_type, M, N])
    s1.pipeline("x")
    s1.partition(s1.data, dim=1)

    s2 = allo.customize(center_reduce, instantiate=[concrete_type, M, N])
    s2.pipeline("x")
    s2.partition(s2.data, dim=2)
    s2.partition(s2.data_out, dim=2)
    s2.partition(s2.mean, dim=1)
    s2.partition(s2.stddev, dim=1)

    s3 = allo.customize(compute_corr, instantiate=[concrete_type, M, N])
    s3.pipeline("j")

    sch = allo.customize(kernel_correlation, instantiate=[concrete_type, M, N])
    sch.compose(s0)
    sch.compose(s1)
    sch.compose(s2)
    sch.compose(s3)

    return sch


def test_correlation():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    M = psize["correlation"][test_psize]["M"]
    N = psize["correlation"][test_psize]["N"]
    N_float = float(N)
    epsilon = 1e-5
    concrete_type = float32
    sch = correlation(concrete_type, M, N, N_float, epsilon)
    mod = sch.build()
    data = np.random.rand(N, M).astype(np.float32)
    mean = np.zeros((M,), dtype=np.float32)
    stddev = np.zeros((M,), dtype=np.float32)
    corr = np.zeros((M, M), dtype=np.float32)
    corr_ref = np.zeros((M, M), dtype=np.float32)
    data_ref, mean_ref, stddev_ref, corr_ref = correlation_np(
        data.copy(), mean.copy(), stddev.copy(), corr_ref, M, N, N_float, epsilon
    )
    mod = sch.build()
    mod(data.copy(), data.copy(), data.copy(), corr)
    np.testing.assert_allclose(corr, corr_ref, rtol=1e-5, atol=1e-5)
    print("test_correlation passed")


if __name__ == "__main__":
    pytest.main([__file__])
