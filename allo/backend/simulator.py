# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, super-init-not-called, too-many-nested-blocks

import os
from ..backend.llvm import LLVMModule
from .._mlir.ir import (
    Location,
    UnitAttr,
    InsertionPoint,
    Module,
    Context,
    Region,
    Block,
    OpView,
)
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    openmp as openmp_d,
)
from .._mlir.passmanager import PassManager
from .._mlir.execution_engine import ExecutionEngine
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function
from ..utils import get_func_inputs_outputs


def build_dataflow_simulator(module: Module, top_func_name: str):
    with module.context, Location.unknown():
        func = find_func_in_module(module, top_func_name)
        assert isinstance(func.body, Region)
        top_func_ops = func.body.blocks[0].operations
        pe_call_define_ops: dict[func_d.CallOp, func_d.FuncOp] = {}
        for op in top_func_ops:
            if isinstance(op, memref_d.AllocOp):
                continue
            if isinstance(op, func_d.CallOp):
                callee_name = str(op.callee)[1:]
                if not callee_name.startswith(("load_buf", "store_res")):
                    for mod_op in module.body.operations:
                        if isinstance(mod_op, func_d.FuncOp):
                            if callee_name == str(mod_op.sym_name).strip('"'):
                                pe_call_define_ops[op] = mod_op
                                break

        # Add the outmost `omp.parallel`
        assert len(pe_call_define_ops) > 0
        omp_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])
        omp_parallel_op = openmp_d.ParallelOp([], [], [], [], ip=omp_ip)
        assert isinstance(omp_parallel_op.region, Region)
        omp_parallel_block = Block.create_at_start(omp_parallel_op.region, [])

        # Add `omp.sections`
        ip_omp_parallel = InsertionPoint(omp_parallel_block)
        omp_sections_op = openmp_d.SectionsOp([], [], [], [], ip=ip_omp_parallel)
        omp_sections_block = Block.create_at_start(omp_sections_op.region, [])
        openmp_d.TerminatorOp(ip=ip_omp_parallel)

        # Add `omp.section`s for PE calls
        ip_omp_sections = InsertionPoint(omp_sections_block)
        for call_op in pe_call_define_ops:
            assert isinstance(call_op, OpView)
            omp_section_op = openmp_d.SectionOp(ip=ip_omp_sections)
            omp_section_block = Block.create_at_start(omp_section_op.region, [])
            ip_omp_section = InsertionPoint(omp_section_block)
            omp_term_op = openmp_d.TerminatorOp(ip=ip_omp_section)
            call_op.operation.move_before(omp_term_op.operation)
        openmp_d.TerminatorOp(ip=ip_omp_sections)


class LLVMOMPModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_dataflow_simulator(self.module, self.top_func_name)
            print(self.module)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # Start lowering
            # Reference: https://discourse.llvm.org/t/help-lowering-affine-loop-to-openmp/72441/9
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-cf-to-llvm,"
                "convert-openmp-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(os.getenv("LLVM_BUILD_DIR"), "lib", "libomp.so"),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )
