# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import allo
from allo.primitives.unify import unify
from allo.ir.types import int8


def _make_unified_simple(L=4):
    def f(A: int8[L], B: int8[L]):
        for i in range(L):
            A[i] = B[i]

    unified = unify(f, f, 1)
    return unified


def test_vivado_hls_project_emits_kernel_only():
    unified = _make_unified_simple()
    with tempfile.TemporaryDirectory() as tmp:
        proj = os.path.join(tmp, "proj_vivado")
        # vivado_hls does not require func_args at construction
        mod = allo.HLSModule(unified, "f1_f2_unified", platform="vivado_hls", mode="csim", project=proj)

        # kernel.cpp must always exist
        assert os.path.exists(os.path.join(proj, "kernel.cpp"))
        # vivado_hls does not generate a host.cpp by default in the constructor
        assert not os.path.exists(os.path.join(proj, "host.cpp"))


def test_pynq_project_emits_kernel_and_header():
    unified = _make_unified_simple()
    with tempfile.TemporaryDirectory() as tmp:
        proj = os.path.join(tmp, "proj_pynq")
        # pynq constructor requires func_args (we don't depend on its value here)
        mod = allo.HLSModule(
            unified,
            "f1_f2_unified",
            platform="pynq",
            mode="csim",
            project=proj,
            func_args=[],
        )

        # kernel.cpp and kernel.h must be emitted for PYNQ
        assert os.path.exists(os.path.join(proj, "kernel.cpp"))
        assert os.path.exists(os.path.join(proj, "kernel.h"))
        # host.py / host.cpp are not generated during construction (host is generated during packaging)
        assert not os.path.exists(os.path.join(proj, "host.cpp"))
