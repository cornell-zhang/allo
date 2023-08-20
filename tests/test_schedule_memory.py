# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import numpy as np
import pytest
from allo.ir.types import int1, int32, float32, index


def test_reuse_blur_x():
    def reuse_blur_x(A: int32[10, 10]) -> int32[10, 8]:
        B: int32[10, 8] = 0
        for y, x in allo.grid(10, 8):
            B[y, x] = A[y, x] + A[y, x + 1] + A[y, x + 2]
        return B

    s = allo.customize(reuse_blur_x)
    RB = s.reuse_at(s.A, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x + 1] + np_A[y][x + 2]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_and_partition():
    def reuse_and_partition(A: int32[10, 10], F: int32[3, 3]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                # B[y, x] += A[y + r, x + c] * F[r, c]
                v += A[y + r, x + c] * F[r, c]
            B[y, x] = v
        return B

    s = allo.customize(reuse_and_partition)
    LB = s.reuse_at(s.A, "y")
    WB = s.reuse_at(LB, "x")
    s.partition(LB, dim=1)
    s.partition(WB)
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_F = np.random.randint(0, 10, size=(3, 3)).astype(np.int32)
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y + r][x + c] * np_F[r][c]

    np_B = mod(np_A, np_F)
    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_tensor():
    def reuse_blur_x_tensor(A: int32[10, 10]) -> int32[10, 8]:
        X: int32[10, 10] = 0
        for y, x in allo.grid(10, 10):
            X[y, x] = A[y, x]

        B: int32[10, 8] = 0
        for yy, xx in allo.grid(10, 8):
            B[yy, xx] = X[yy, xx] + X[yy, xx + 1] + X[yy, xx + 2]
        return B

    s = allo.customize(reuse_blur_x_tensor)
    RB = s.reuse_at(s.X, "xx")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x + 1] + np_A[y][x + 2]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_y():
    def reuse_blur_y(A: int32[10, 10]) -> int32[8, 10]:
        B: int32[8, 10] = 0
        for y, x in allo.grid(8, 10):
            B[y, x] = A[y, x] + A[y + 1, x] + A[y + 2, x]
        return B

    s = allo.customize(reuse_blur_y)
    RB = s.reuse_at(s.A, "y")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[y][x] + np_A[y + 1][x] + np_A[y + 2][x]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_y():
    def reuse_blur_x_y(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            B[y, x] = A[y, x] + A[y + 1, x + 1] + A[y + 2, x + 2]
        return B

    s = allo.customize(reuse_blur_x_y)
    RB_y = s.reuse_at(s.A, "y")
    RB_x = s.reuse_at(RB_y, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y + 1][x + 1] + np_A[y + 2][x + 2]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_3D():
    def reuse_blur_x_3D(A: int32[10, 10, 2]) -> int32[10, 8, 2]:
        B: int32[10, 8, 2] = 0
        for y, x, c in allo.grid(10, 8, 2):
            B[y, x, c] = A[y, x, c] + A[y, x + 1, c] + A[y, x + 2, c]
        return B

    s = allo.customize(reuse_blur_x_3D)
    RB = s.reuse_at(s.A, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10, 2)).astype(np.int32)
    np_C = np.zeros((10, 8, 2), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            for c in range(0, 2):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y][x + 1][c] + np_A[y][x + 2][c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_y_3D():
    def reuse_blur_y_3D(A: int32[10, 10, 2]) -> int32[8, 10, 2]:
        B: int32[8, 10, 2] = 0
        for y, x, c in allo.grid(8, 10, 2):
            B[y, x, c] = A[y, x, c] + A[y + 1, x, c] + A[y + 2, x, c]
        return B

    s = allo.customize(reuse_blur_y_3D)
    RB = s.reuse_at(s.A, "y")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10, 2)).astype(np.int32)
    np_C = np.zeros((8, 10, 2), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            for c in range(0, 2):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y + 1][x][c] + np_A[y + 2][x][c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_y_3D():
    def reuse_blur_x_y_3D(A: int32[10, 10, 2]) -> int32[8, 8, 2]:
        B: int32[8, 8, 2] = 0
        for y, x, c in allo.grid(8, 8, 2):
            B[y, x, c] = A[y, x, c] + A[y + 1, x + 1, c] + A[y + 2, x + 2, c]
        return B

    s = allo.customize(reuse_blur_x_y_3D)
    RB_y = s.reuse_at(s.A, "y")
    RB_x = s.reuse_at(RB_y, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10, 2)).astype(np.int32)
    np_C = np.zeros((8, 8, 2), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for c in range(0, 2):
                np_C[y][x][c] = (
                    np_A[y][x][c] + np_A[y + 1][x + 1][c] + np_A[y + 2][x + 2][c]
                )

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_y_z_3D():
    def reuse_blur_x_y_z_3D(A: int32[10, 8, 6]) -> int32[8, 6, 4]:
        B: int32[8, 6, 4] = 0
        for y, x, z in allo.grid(8, 6, 4):
            B[y, x, z] = A[y, x, z] + A[y + 1, x + 1, z + 1] + A[y + 2, x + 2, z + 2]
        return B

    s = allo.customize(reuse_blur_x_y_z_3D)
    RB_y = s.reuse_at(s.A, "y")
    RB_x = s.reuse_at(RB_y, "x")
    RB_z = s.reuse_at(RB_x, "z")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 8, 6)).astype(np.int32)
    np_C = np.zeros((8, 6, 4), dtype="int")

    for y in range(0, 8):
        for x in range(0, 6):
            for z in range(0, 4):
                np_C[y][x][z] = (
                    np_A[y][x][z]
                    + np_A[y + 1][x + 1][z + 1]
                    + np_A[y + 2][x + 2][z + 2]
                )

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_conv2D_lb():
    def conv2D_lb(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb)
    LB = s.reuse_at(s.A, "y")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y + r][x + c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_conv2D_wb():
    def conv2D_wb(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_wb)
    WB = s.reuse_at(s.A, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y + r][x + c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_conv2D_lb_wb():
    def conv2D_lb_wb(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb_wb)
    LB = s.reuse_at(s.A, "y")
    WB = s.reuse_at(LB, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y + r][x + c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_partition_basic():
    def partition_basic(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10] = 0
        for x, y in allo.grid(10, 10):
            B[x, y] = A[x, y]
        return B

    s = allo.customize(partition_basic)
    s.partition(s.A)
    print(s.module)
    ir = str(s.module)
    assert "affine_map<(d0, d1) -> (d0, d1, 0, 0)>" in ir


def test_partition_type():
    def partition_type(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10] = 0
        for x, y in allo.grid(10, 10):
            B[x, y] = A[x, y]
        return B

    def test_1():
        s0 = allo.customize(partition_type)
        s0.partition(s0.A)
        print(s0.module)
        ir = str(s0.module)
        assert "affine_map<(d0, d1) -> (d0, d1, 0, 0)>" in ir

    def test_2():
        s1 = allo.customize(partition_type)
        s1.partition(s1.A, partition_type=1, dim=0, factor=2)
        print(s1.module)
        ir = str(s1.module)
        assert (
            "affine_map<(d0, d1) -> (d0 floordiv 5, d1 floordiv 5, d0 mod 5, d1 mod 5)>"
            in ir
        )

    def test_3():
        s2 = allo.customize(partition_type)
        s2.partition(s2.A, partition_type=2)
        print(s2.module)
        ir = str(s2.module)
        assert (
            "affine_map<(d0, d1) -> (d0 mod 0, d1 mod 0, d0 floordiv 0, d1 floordiv 0)>"
            in ir
        )

    test_1()
    test_2()
    test_3()


def test_partition_dim_factor():
    def partition_dim_factor(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10] = 0
        for x, y in allo.grid(10, 10):
            B[x, y] = A[x, y]
        return B

    s = allo.customize(partition_dim_factor)
    s.partition(s.A, dim=1, factor=2)
    print(s.module)
    ir = str(s.module)
    assert "affine_map<(d0, d1) -> (d0, 0, 0, d1)>" in ir


def test_reshape():
    def reshape(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10] = 0
        for x, y in allo.grid(10, 10):
            B[x, y] = A[x, y]
        C: int32[10, 10] = 0
        for xx, yy in allo.grid(10, 10):
            C[xx, yy] = B[xx, yy]
        return C

    s = allo.customize(reshape)
    s.reshape(s.B, (2, 5, 2, 5))
    print(s.module)
    ir = str(s.module)
    assert "memref<2x5x2x5xi32>" in ir


def test_conv2D_lb_wb_schedule():
    def conv2D_lb_wb_schedule(A: int32[10, 10]) -> int32[8, 8]:
        B: int32[8, 8] = 0
        for y, x in allo.grid(8, 8):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y + r, x + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb_wb_schedule)
    # xo, xi = s.split("x", 4)
    s.split("x", 4)
    s.reorder("x.outer", "y", "x.inner")
    LB = s.reuse_at(s.A, "y")
    WB = s.reuse_at(LB, "x.inner")
    s.partition(LB, dim=2)
    s.partition(WB)
    s.reshape(s.B, (8, 2, 4))
    s.pipeline("y")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((8, 2, 4), dtype="int")

    for y in range(0, 8):
        for xo in range(0, 2):
            for xi in range(0, 4):
                for r in range(0, 3):
                    for c in range(0, 3):
                        np_C[y][xo][xi] += np_A[y + r][xi + xo * 4 + c]

    np_B = mod(np_A)

    assert np.array_equal(np_B, np_C)


def test_conv2D_lb_wb_stride_2():
    def conv2D_lb_wb_schedule(A: int32[10, 10]) -> int32[4, 4]:
        B: int32[4, 4] = 0
        for y, x in allo.grid(4, 4):
            v: int32 = 0
            for r, c in allo.reduction(3, 3):
                v += A[y * 2 + r, x * 2 + c]
            B[y, x] = v
        return B

    s = allo.customize(conv2D_lb_wb_schedule)
    LB = s.reuse_at(s.A, "y")
    WB = s.reuse_at(LB, "x")
    print(s.module)
    mod = s.build()

    np_A = np.random.randint(0, 10, size=(10, 10)).astype(np.int32)
    np_C = np.zeros((4, 4), dtype="int")

    for y in range(0, 4):
        for x in range(0, 4):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y * 2 + r][x * 2 + c]

    np_B = mod(np_A)
    assert np.array_equal(np_B, np_C)


def test_avgpool_nchw():
    bs = 4
    ic, oc = 16, 16
    ih, iw = 8, 8
    kh, kw = 2, 2
    stride = 1
    oh, ow = (ih - kh) // stride + 1, (iw - kw) // stride + 1
    dtype = float32

    def avgpool_nchw(A: dtype[bs, ic, ih, iw]) -> dtype[bs, oc, oh, ow]:
        B: dtype[bs, oc, oh, ow] = 0.0
        stride: index = 1
        for n, c, h, w in allo.grid(bs, oc, oh, ow):
            v: dtype = 0.0
            for rh, rw in allo.reduction(kh, kw):
                v += A[n, c, h * stride + rh, w * stride + rw]
            B[n, c, h, w] = v / (kh * kw)
        return B

    s = allo.customize(avgpool_nchw)
    LB = s.reuse_at(s.A, "h")
    WB = s.reuse_at(LB, "w")
    print(s.module)
    mod = s.build()

    np_A = np.random.random((bs, ic, ih, iw)).astype(np.float32)
    np_C = np.zeros((bs, oc, oh, ow), dtype="float")

    for n in range(0, bs):
        for c in range(0, oc):
            for y in range(0, oh):
                for x in range(0, ow):
                    for rh in range(0, kh):
                        for rw in range(0, kw):
                            np_C[n][c][y][x] += (
                                np_A[n][c][y * stride + rh][x * stride + rw]
                            ) / (kh * kw)

    np_B = mod(np_A)
    assert np.allclose(np_B, np_C)


if __name__ == "__main__":
    pytest.main([__file__])
