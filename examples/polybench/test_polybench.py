# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def test_adi():
    from . import adi

    adi.test_adi()


def test_atax():
    from . import atax

    atax.test_atax()


def test_bicg():
    from . import bicg

    bicg.test_bicg()


def test_cholesky():
    from . import cholesky

    cholesky.test_cholesky()


def test_correlation():
    from . import correlation

    correlation.test_correlation()


def test_covariance():
    from . import covariance

    covariance.test_covariance()


def test_deriche():
    from . import deriche

    deriche.test_deriche()


def test_doitgen():
    from . import doitgen

    doitgen.test_doitgen()


def test_durbin():
    from . import durbin

    durbin.test_durbin()


def test_fdtd_2d():
    from . import fdtd_2d

    fdtd_2d.test_fdtd_2d()


def test_floyd_warshall():
    from . import floyd_warshall

    floyd_warshall.test_floyd_warshall()


def test_gemm():
    from . import gemm

    gemm.test_gemm()


def test_gemver():
    from . import gemver

    gemver.test_gemver()


def test_gesummv():
    from . import gesummv

    gesummv.test_gesummv()


def test_gramschmidt():
    from . import gramschmidt

    gramschmidt.test_gramschmidt()


def test_heat_3d():
    from . import heat_3d

    heat_3d.test_heat_3d()


def test_jacobi_1d():
    from . import jacobi_1d

    jacobi_1d.test_jacobi_1d()


def test_jacobi_2d():
    from . import jacobi_2d

    jacobi_2d.test_jacobi_2d()


def test_lu():
    from . import lu

    lu.test_lu()


def test_ludcmp():
    from . import ludcmp

    ludcmp.test_ludcmp()


def test_mvt():
    from . import mvt

    mvt.test_mvt()


def test_nussinov():
    from . import nussinov

    nussinov.test_nussinov()


def test_seidel_2d():
    from . import seidel_2d

    seidel_2d.test_seidel_2d()


def test_symm():
    from . import symm

    symm.test_symm()


def test_syr2k():
    from . import syr2k

    syr2k.test_syr2k()


def test_syrk():
    from . import syrk

    syrk.test_syrk()


def test_three_mm():
    from . import three_mm

    three_mm.test_three_mm()


def test_trisolv():
    from . import trisolv

    trisolv.test_trisolv()


def test_trmm():
    from . import trmm

    trmm.test_trmm()


def test_two_mm():
    from . import two_mm

    two_mm.test_two_mm()
