# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-nested-blocks, too-many-instance-attributes
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import shutil

try:
    import aie.ir as aie_ir
    import aie.passmanager as aie_pass_manager
except ImportError:
    pass

import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.dialects import func as allo_func_d
from ..._mlir.ir import Type

from ...passes import analyze_read_write_patterns
from ...memory import DTensor
from .external_kernel import ExternalModule, ExternalModuleBase

from ..._mlir.passmanager import PassManager as mlir_pass_manager
from .mlir_codegen import CodeGenerator, Argument, Stream
from .utils import (
    Argument,
    Stream,
    inject_external_kernels,
    get_df_kernels,
    classify_aie_functions,
    classify_aie_functions_experimental,
    codegen_external_kernels,
    lib_kernel_replacement,
    read_tensor_from_file,
    codegen_host,
)
from .mapping import ComputationGraph, OrderedDTensorTileGroup


class AIE_MLIRModule:
    # ############################################################
    # Construction
    # ############################################################
    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name: str,
        parameter_list: dict[str, int],
        func_args: dict,
        project_dir: str,
        stream_info: dict,
        stream_types_dict: dict[str, Type],
        ext_libs: list = None,
    ):
        """
        Note: the module is data-driven,
            we need to carefully manage data transfer between 'functions' to avoid deadlocks.
            For example, launching the kernels in topological order.
        """
        # module metadata
        # print(module)
        self.project_dir: str = project_dir
        self.allo_module: allo_ir.ir.Module = module
        self.top_func_name: str = top_func_name
        self.module_parameter_list = [
            k for k, _ in sorted(parameter_list.items(), key=lambda item: item[1])
        ]

        self.external_kernel_lib: dict[str, ExternalModule] = {}
        for ext_kernel in ext_libs:
            if isinstance(ext_kernel, ExternalModule):
                self.external_kernel_lib[ext_kernel.name] = ext_kernel

        self.func_args: dict[str, list[Argument]] = {}
        self.streams: dict[str, Stream] = {}
        self.stream_info: dict[str, dict[str, bool]] = {}
        self._init_func_args(func_args)
        self._init_streams(stream_info, stream_types_dict)

        # index in top fucntion argument list -> DTensor
        self.global_inputs: dict[int, DTensor] = None
        self.global_outputs: dict[int, DTensor] = None
        # function name -> (argument index -> (argument, is_input))
        self.core_func_args: dict[str, dict[int, tuple[Argument, bool]]] = None

        self.aie_module: aie_ir.Module = None

    def _init_func_args(self, func_args: dict):
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

    def _init_streams(self, stream_info: dict, stream_types_dict: dict[str, Type]):
        """
        Collect allo.stream information for each function.
        """
        for func_name, info_list in stream_info.items():
            self.stream_info[func_name] = {}
            for name, io in info_list:
                self.streams[name].set_element_type(
                    str(stream_types_dict[name]), self.allo_module.context
                )
                if io == "in":
                    self.streams[name].dst = func_name
                    self.stream_info[func_name][name] = True
                else:
                    self.streams[name].src = func_name
                    self.stream_info[func_name][name] = False

    # ############################################################
    # Build
    # ############################################################
    def _init_virtual_graph(self, use_external_kernels: dict[str, bool]):
        assert (
            self.core_func_args is not None
            and self.global_inputs is not None
            and self.global_outputs is not None
        ), "Analysis of kernel parameters should be done before initializing virtual graph"

        self.virtual_computation_graph: ComputationGraph = ComputationGraph(
            self.allo_module, self.streams, self.core_func_args, use_external_kernels
        )

    def analyze_kernel_parameters(
        self, injected_external_kernels: dict[str:ExternalModuleBase]
    ):
        """
        Analyze the parameters of each df.kernel.

        Collected information:
            - self.core_func_args: function name -> (argument index -> (argument, is_input))
            - self.global_inputs: global input argument index -> DTensor
            - self.global_outputs: global output argument index -> DTensor
        """
        # init
        self.core_func_args = {}
        self.global_inputs = {}
        self.global_outputs = {}
        # analyze
        df_kernels = get_df_kernels(self.allo_module)
        for kernel in df_kernels:
            kernel_name = kernel.attributes["sym_name"].value
            self.core_func_args[kernel_name] = {}
            # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
            in_idx_list, out_idx_list = analyze_read_write_patterns(
                kernel, injected_external_kernels
            )
            for io_idx_list, io_type in (
                (in_idx_list, "in"),
                (out_idx_list, "out"),
            ):
                for io_idx in io_idx_list:
                    argument: Argument = self.func_args[kernel_name][io_idx]
                    self.core_func_args[kernel_name][io_idx] = (
                        argument,
                        io_type == "in",
                    )
                    if not argument.dtensor is None:
                        argument.dtensor.set_access_pattern()
                        argument.dtensor.type_as_param = kernel.arguments[
                            io_idx
                        ].type.shape
                        global_idx = self.func_args[self.top_func_name].index(argument)
                        argument.dtensor.set_global_id(global_idx)
                        if io_type == "in":
                            self.global_inputs[global_idx] = argument.dtensor
                        else:
                            self.global_outputs[global_idx] = argument.dtensor
            # streams
            for i, _ in enumerate(kernel.arguments):
                func_arg = self.func_args[kernel_name][i]
                if (
                    i in self.core_func_args[kernel_name]
                    or func_arg.stream is None  # unused
                ):
                    continue
                self.core_func_args[kernel_name][i] = (
                    func_arg,
                    self.stream_info[kernel_name][func_arg.stream.name],
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

    def allo_opt(self):
        """
        run optimized passes on allo mlir
        """
        # pipeline = f"builtin.module(copy-on-write)"
        # with self.allo_module.context:
        #     mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)

        for func in self.allo_module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and "df.kernel" in func.attributes:
                lib_kernel_replacement(func)

        pipeline = f"builtin.module(copy-on-write, canonicalize)"
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)

        # record optimized allo mlir
        with open(
            os.path.join(self.project_dir, "original_opt.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))

    def build_experimental(
        self,
        device_type="npu1_4col",
        enable_virtual_mapping: bool = False,
        mapping_primitives: list[tuple[str, list]] = [],
        profile: bool = False,
        warmup: int = 20,
        num_iters: int = 100,
    ):
        if "npu1" in device_type:
            self.device = "npu1"
        elif "npu2" in device_type:
            self.device = "npu2"
        else:
            raise ValueError("Unsupported device type.")

        self.profile = profile
        self.warmup = warmup
        self.num_iters = num_iters
        build_dir = os.path.join(self.project_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original_virtual.mlir"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(self.allo_module))

        # inject external kernels
        # (inject before virtual mapping since using external kernel may require layout transformation when transferring data)
        use_external_kernels, injected_external_kernels, include_src = (
            inject_external_kernels(
                self.allo_module,
                self.top_func_name,
                self.external_kernel_lib,
                "aie2" if self.device == "npu1" else "aie2p",
            )
        )
        self.analyze_kernel_parameters(injected_external_kernels)
        self._init_virtual_graph(use_external_kernels)
        if enable_virtual_mapping:
            for mapping in mapping_primitives:
                primitive = mapping[0]
                arg_list = mapping[1]
                if primitive == "chain":
                    assert len(arg_list) == 2
                    self.virtual_computation_graph.chain(arg_list[0], arg_list[1])
                if primitive == "bundle":
                    self.virtual_computation_graph.bundle(arg_list)

        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))

        pipeline = f"builtin.module(lower-view-with-layout-ops, canonicalize)"
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        
        with open(
            os.path.join(self.project_dir, "raw.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))

        self.allo_opt()

        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-affine",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        # code generation
        top_func, core_funcs, external_funcs = classify_aie_functions_experimental(
            self.allo_module, self.top_func_name
        )
        code_generator = CodeGenerator(
            device_type,
            self.global_inputs,
            self.global_outputs,
            top_func,
            self.core_func_args,
            self.streams,
            self.virtual_computation_graph,
        )
        self.aie_module = code_generator.aie_codegen_nightly(
            core_funcs,
            external_funcs,
        )

        # TODO: opt passes on aie-mlir
        passes = [
            "func.func(affine-loop-unroll), canonicalize",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.aie_module.context:
            aie_pass_manager.PassManager.parse(pipeline).run(self.aie_module.operation)

        self.post_codegen_build(injected_external_kernels, include_src)
        return self

    def post_codegen_build(
        self, injected_kernels: dict[str, ExternalModuleBase], include_src: set[str]
    ):
        with open(
            os.path.join(self.project_dir, "top.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.aie_module))
        if len(injected_kernels) > 0:
            paths = set()
            # user defined external kernels
            for ext_module in self.external_kernel_lib.values():
                paths.add((ext_module.impl_path, ext_module.filename))
            for src_path, dst_file in paths:
                target_path = os.path.join(self.project_dir, dst_file)
                if os.path.exists(target_path) and os.path.samefile(
                    src_path, target_path
                ):
                    continue
                shutil.copy(src_path, target_path)
            kernel_code = codegen_external_kernels(
                injected_kernels,
                include_src,
                "aie2" if self.device == "npu1" else "aie2p",
            )
            with open(
                os.path.join(self.project_dir, "external.cc"), "w", encoding="utf-8"
            ) as f:
                f.write(kernel_code)
            cmd = f"cd {self.project_dir} && $PEANO_INSTALL_DIR/bin/clang++ -O2 -v -std=c++20 --target=aie2{"p" if self.device == "npu2" else ""}-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I $MLIR_AIE_INSTALL_DIR/include -I $MLIR_AIE_EXTERNAL_KERNEL_DIR/ -I. -c external.cc -o external.o"
            with subprocess.Popen(cmd, shell=True) as process:
                process.wait()
            if process.returncode != 0:
                raise RuntimeError("Failed to compile external kernels.")
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

    def collect_io(
        self,
        func_groups: dict[str, list[allo_func_d.FuncOp]],
    ) -> tuple[dict, dict]:
        """
        Analyze input/output tensors of each function in the groups.
        Returns dictionaries of input/output DTensors for each function group and core.
        """
        # init
        self.core_func_args = {}
        self.global_inputs = {}
        self.global_outputs = {}
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
                            argument.dtensor.set_access_pattern()
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
        use_external_kernels, injected_external_kernels, include_src = (
            inject_external_kernels(
                self.allo_module,
                self.top_func_name,
                self.external_kernel_lib,
                "aie2" if self.device == "npu1" else "aie2p",
            )
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
        self.post_codegen_build(injected_external_kernels, include_src)
        return self

    def help(self):
        # print the parameter list of the module
        print("Parameter reference:", self.module_parameter_list)

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
