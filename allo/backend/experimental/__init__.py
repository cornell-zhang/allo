# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-nested-blocks, too-many-instance-attributes
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import shutil

import aie.ir as aie_ir

import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.dialects import func as allo_func_d

from ...passes import analyze_read_write_patterns
from ...memory import DTensor
from .external_kernel import ExternalModule

from ..._mlir.passmanager import PassManager as mlir_pass_manager
from .mlir_codegen import CodeGenerator, Argument, Stream
from .utils import (
    inject_external_kernels,
    classify_aie_functions,
    codegen_external_kernels,
    read_tensor_from_file,
    codegen_host,
)


class AIE_MLIRModule:
    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name: str,
        func_args: dict,
        project_dir: str,
        stream_info: dict,
        ext_libs: list = None,
    ):
        self.allo_module: allo_ir.ir.Module = module
        self.top_func_name: str = top_func_name
        self.streams: dict[str, Stream] = {}

        tmp_map: dict = {}
        self.func_args: dict[str, list[Argument]] = {}
        for func_name, args in func_args.items():
            self.func_args[func_name] = []
            for arg in args:
                if arg in tmp_map:
                    self.func_args[func_name].append(tmp_map[arg])
                elif isinstance(arg, DTensor):
                    argument = Argument(arg, None)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                elif isinstance(arg, str):
                    stream = Stream(arg)
                    self.streams[arg] = stream
                    argument = Argument(None, stream)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                else:
                    raise ValueError(f"Unresolved function argument {arg}")

        self.stream_info: dict[str, dict[str, bool]] = {}
        for func_name, info_list in stream_info.items():
            self.stream_info[func_name] = {}
            for name, io in info_list:
                if io == "in":
                    self.streams[name].dst = func_name
                    self.stream_info[func_name][name] = True
                else:
                    self.streams[name].src = func_name
                    self.stream_info[func_name][name] = False

        self.project_dir: str = project_dir

        self.external_kernel_lib: dict[str, ExternalModule] = {}
        for ext_kernel in ext_libs:
            if isinstance(ext_kernel, ExternalModule):
                self.external_kernel_lib[ext_kernel.top] = ext_kernel

        # index in top fucntion argument list -> DTensor
        self.global_inputs: dict[int, DTensor] = {}
        self.global_outputs: dict[int, DTensor] = {}

        self.core_func_args: dict[str, dict[int, tuple[Argument, bool]]] = (
            {}
        )  # core func name -> a list of (dtensors, is_in)

        self.aie_module: aie_ir.Module = None

    def collect_io(
        self,
        func_groups: dict[str, list[allo_func_d.FuncOp]],
    ) -> tuple[dict, dict]:
        """
        Analyze input/output tensors of each function in the groups.
        Returns dictionaries of input/output DTensors for each function group and core.
        """
        inputs = {}
        outputs = {}
        for func_name, funcs in func_groups.items():
            inputs[func_name] = {}
            outputs[func_name] = {}
            inputs[func_name]["_global"] = []
            outputs[func_name]["_global"] = []
            for func in funcs:
                func_name_w_id = func.attributes["sym_name"].value
                self.core_func_args[func_name_w_id] = {}
                # [NOTE]: function name implies some kind of mapping from io tensor to 'core's
                func_id = tuple(
                    int(x) for x in func_name_w_id.split(func_name + "_")[-1].split("_")
                )
                # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
                in_idx, out_idx = analyze_read_write_patterns(
                    func, self.external_kernel_lib
                )
                for io_lst, io_idx, io in (
                    (inputs, in_idx, "in"),
                    (outputs, out_idx, "out"),
                ):
                    io_lst[func_name][func_id] = []
                    for idx in io_idx:
                        argument: Argument = self.func_args[func_name_w_id][idx]
                        self.core_func_args[func_name_w_id][idx] = (
                            argument,
                            io == "in",
                        )
                        if not argument.dtensor is None:
                            argument.dtensor.type_as_param = func.arguments[
                                idx
                            ].type.shape
                            if argument.dtensor not in io_lst[func_name]["_global"]:
                                io_lst[func_name]["_global"].append(argument.dtensor)
                                if io == "in":
                                    self.global_inputs[
                                        self.func_args[self.top_func_name].index(
                                            argument
                                        )
                                    ] = argument.dtensor
                                else:
                                    self.global_outputs[
                                        self.func_args[self.top_func_name].index(
                                            argument
                                        )
                                    ] = argument.dtensor
                            io_lst[func_name][func_id].append(argument.dtensor)
                # streams
                for i, _ in enumerate(func.arguments):
                    func_arg = self.func_args[func_name_w_id][i]
                    if (
                        i in self.core_func_args[func_name_w_id]
                        or func_arg.stream is None  # unused
                    ):
                        continue
                    self.core_func_args[func_name_w_id][i] = (
                        func_arg,
                        self.stream_info[func_name_w_id][func_arg.stream.name],
                    )
        # validity check
        for i in range(len(self.global_inputs)):
            assert (
                i in self.global_inputs
            ), "inputs should be the starting arguments of the function"
        for i in range(len(self.global_outputs)):
            assert (
                i + len(self.global_inputs) in self.global_outputs
            ), "outputs should be the ending arguments of the function"

        return inputs, outputs

    def build(
        self,
        device_type="npu1_4col",
        profile: bool = False,
        warmup: int = 20,
        num_iters: int = 100,
    ):
        self.profile = profile
        self.warmup = warmup
        self.num_iters = num_iters
        build_dir = os.path.join(self.project_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        # TODO: maybe use other ways to capture the relationship between DTensor, function group
        _, core_func_groups, _ = classify_aie_functions(
            self.allo_module, self.top_func_name
        )
        inputs, outputs = self.collect_io(core_func_groups)

        # - extract external kernels
        use_external_kernels, injected_kernels, include_src = inject_external_kernels(
            self.allo_module, self.top_func_name, self.external_kernel_lib
        )
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        # - lower tensor to memref with registered pass
        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-affine",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        top_func, core_func_groups, external_funcs = classify_aie_functions(
            self.allo_module, self.top_func_name
        )
        code_generator = CodeGenerator(
            device_type, self.global_inputs, self.global_outputs, top_func
        )
        self.aie_module = code_generator.aie_codegen(
            core_func_groups,
            external_funcs,
            inputs,
            outputs,
            use_external_kernels,
            self.core_func_args,
            self.streams,
        )
        with open(
            os.path.join(self.project_dir, "top.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.aie_module))
        if len(injected_kernels) > 0:
            paths = set()
            # user defined external kernels
            for ext_module in self.external_kernel_lib.values():
                paths.add(ext_module.impl_path)
            for src_path in paths:
                target_path = os.path.join(self.project_dir, os.path.basename(src_path))
                if os.path.exists(target_path) and os.path.samefile(
                    src_path, target_path
                ):
                    continue
                shutil.copy(src_path, target_path)
            kernel_code = codegen_external_kernels(injected_kernels, include_src)
            with open(
                os.path.join(self.project_dir, "external.cc"), "w", encoding="utf-8"
            ) as f:
                f.write(kernel_code)
            cmd = f"cd {self.project_dir} && $PEANO_INSTALL_DIR/bin/clang++ -O2 -v -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I $MLIR_AIE_INSTALL_DIR/include -I $MLIR_AIE_EXTERNAL_KERNEL_DIR/aie2 -I. -c external.cc -o external.o"
            with subprocess.Popen(cmd, shell=True) as process:
                process.wait()
            if process.returncode != 0:
                raise RuntimeError("Failed to compile external kernels.")
        # TODO
        # build mlir-aie
        cmd = f"cd {self.project_dir} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to compile the MLIR-AIE code")
        # generate host code
        path = os.path.dirname(__file__)
        path = os.path.join(path, "../../harness/aie")
        os.system(f"cp -r {path}/* {self.project_dir}")
        host_code = codegen_host(self.global_inputs, self.global_outputs)
        with open(
            os.path.join(self.project_dir, "test.cpp"), "w", encoding="utf-8"
        ) as f:
            f.write(host_code)
        cmd = f"cd {self.project_dir}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to build AIE project.")
        return self

    def __call__(self, *args):
        for i in range(len(self.global_inputs)):
            with open(
                os.path.join(self.project_dir, f"input{i}.data"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join([str(i) for i in args[i].flatten()]))
        cmd = f"cd {self.project_dir} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE {f'-p true --warmup {self.warmup} --test_iter {self.num_iters}' if self.profile else ''}"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to execute AIE code.")
        # TODO: need to complete multiple outputs rules
        result = read_tensor_from_file(
            self.global_outputs[len(args) - 1].dtype,
            args[-1].shape,
            f"{self.project_dir}/output.data",
        )
        # suppose the last argument is output
        args[-1][:] = result
