# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin


import os
import glob
import ctypes

def _preload_shared_libraries():
    """
    Preloads shared libraries to avoid ImportError due to missing RPATH.
    This is necessary because the default RPATH settings might not propagate correct
    origin paths in all environments.
    """
    current_dir = os.path.dirname(__file__)
    # The library is located in the _mlir subdirectory
    mlir_dir = os.path.join(current_dir, "_mlir")
    libs = glob.glob(os.path.join(mlir_dir, "libAlloMLIRAggregateCAPI.so*"))
    for lib in libs:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

_preload_shared_libraries()

from . import frontend, backend, ir, passes, library, _mlir
from .customize import customize, Partition
from .backend.llvm import invoke_mlir_parser, LLVMModule
from .backend.hls import HLSModule
from .backend.ip import IPModule
from .dsl import *
from .template import *
from .verify import verify
