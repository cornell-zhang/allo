# pylint: disable=import-error, c-extension-no-member, too-many-nested-blocks, too-many-instance-attributes, pointless-exception-statement
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import shutil

try:
    import aie.ir as aie_ir
    import aie.passmanager as aie_pass_manager
except ImportError:
    pass

from allo._mlir.exceptions import APIWarning
import allo._mlir._mlir_libs._mlir as allo_ir
from allo._mlir.dialects import (
    allo as allo_d,
    func as allo_func_d,
    _memref_ops_gen as allo_memref_d,
)
from allo._mlir.ir import (
    Type,
    StringAttr,
    InsertionPoint,
    FlatSymbolRefAttr,
    BlockArgument,
    MemRefType,
)
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from ...passes import analyze_read_write_patterns
from ...memory import DTensor
from .external_kernel import ExternalModule, ExternalModuleBase
from .mlir_codegen import CodeGenerator
from .utils import (
    Argument,
    Stream,
    inject_external_kernels,
    matmul_external_kernel_config_map,
    get_df_kernels,
    classify_aie_functions,
    classify_aie_functions_experimental,
    codegen_external_kernels,
    simplify_matmul_accumulate,
    collect_lib_func_call,
    read_tensor_from_file,
    codegen_host,
)
from .mapping import ComputationGraph


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
        self.project_dir: str = project_dir
        self.allo_module: allo_ir.ir.Module = module
        self.top_func_name: str = top_func_name
        self.module_parameter_list = [
            k for k, _ in sorted(parameter_list.items(), key=lambda item: item[1])
        ]

        self.external_kernel_lib: dict[str, ExternalModule] = {}
        for ext_kernel in ext_libs:
            if isinstance(ext_kernel, ExternalModule):
                self.external_kernel_lib[ext_kernel.top] = ext_kernel

        self.func_args: dict[str, list[Argument]] = {}
        self.streams: dict[str, Stream] = {}
        self.stream_info: dict[str, dict[str, bool]] = {}
        self._init_func_args(func_args)
        self.computation_is_dag = self._init_streams(stream_info, stream_types_dict)

        # index in top function argument list -> DTensor
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
        edge_map = {src: set() for src in stream_info.keys()}
        for stream in self.streams.values():
            edge_map[stream.src].add(stream.dst)
        visited = set()
        rec_stack = set()

        def dfs(node):
            if node in rec_stack:
                return False  # found a cycle
            if node in visited:
                return True

            visited.add(node)
            rec_stack.add(node)

            for neighbor in edge_map.get(node, set()):
                if not dfs(neighbor):
                    return False

            rec_stack.remove(node)
            return True

        return all(dfs(node) for node in edge_map.keys())

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
            self.allo_module,
            self.top_func_name,
            self.streams,
            self.core_func_args,
            use_external_kernels,
        )

    def assign_tag_to_kernel(self):
        """
        Assign tag to df kernels (serve as some kind of rerolling)
        """

        class Tagger:
            def __init__(self) -> None:
                self.tag_map: dict[str, str] = {}
                self.counter = 0

            def get_tag(self, key: str) -> str:
                """Return existing tag or assign a new one if not present."""
                if key not in self.tag_map:
                    tag = f"tag_{self.counter}"
                    self.tag_map[key] = tag
                    self.counter += 1
                return self.tag_map[key]

        tagger = Tagger()
        df_kernels = get_df_kernels(self.allo_module)
        for kernel in df_kernels:
            tag_key = re.sub(
                r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(kernel.operation)
            )
            tag = tagger.get_tag(tag_key)
            with kernel.context:
                kernel.attributes["tag"] = StringAttr.get(tag)
        return df_kernels

    def analyze_kernel_parameters(
        self,
        df_kernels: list[allo_func_d.FuncOp],
        injected_external_kernels: dict[str:ExternalModuleBase],
    ):
        """
        Analyze the parameters of each df.kernel.

        Collected information:
            - self.core_func_args: function name -> (argument index -> (argument, is_input))
            - self.global_inputs: global input argument index -> DTensor
            - self.global_outputs: global output argument index -> DTensor
        """
        tag_to_read_write_pattern: dict[str, tuple[list, list]] = {}
        # init
        self.core_func_args = {}
        self.global_inputs = {}
        self.global_outputs = {}
        # analyze
        for kernel in df_kernels:
            kernel_name = kernel.attributes["sym_name"].value
            self.core_func_args[kernel_name] = {}
            tag = kernel.attributes["tag"].value
            if tag in tag_to_read_write_pattern:
                in_idx_list, out_idx_list = tag_to_read_write_pattern[tag]
            else:
                # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
                in_idx_list, out_idx_list = analyze_read_write_patterns(
                    kernel, injected_external_kernels
                )
                tag_to_read_write_pattern[tag] = (in_idx_list, out_idx_list)
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

        def vectorize_matmul(function: allo_func_d.FuncOp):
            """
            Using vectorized matmul to replace scalar version.
            Layout transform is required before and after using vectorized matmul.

            * All external kernels are from https://github.com/Xilinx/mlir-aie/tree/v1.0/aie_kernels.
            """
            matmul_ops: list[allo_func_d.CallOp] = collect_lib_func_call(
                function, "matmul_scalar"
            )
            for call_matmul_op in matmul_ops:
                input_a = call_matmul_op.operands[0]
                input_b = call_matmul_op.operands[1]
                output = call_matmul_op.operands[-1]
                M, K = MemRefType(input_a.type).shape
                _, N = MemRefType(input_b.type).shape
                dtype = str(input_a.type.element_type)
                out_dtype = str(output.type.element_type)
                matmul_configs = matmul_external_kernel_config_map[(dtype, out_dtype)]
                if self.device == "npu1":
                    m, n, k = matmul_configs["aie2"]
                else:
                    m, n, k = matmul_configs["aie2p"]
                with function.context, allo_ir.ir.Location.unknown():
                    new_input_0 = allo_d.transform_layout(
                        input_a.type,
                        input_a,
                        [0, 0, 0, 0],
                        [M // m, K // k, m, k],
                        [m * K, k, K, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_input_0.owner.attributes["layout_hint"] = StringAttr.get(
                        f"A_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    new_input_1 = allo_d.transform_layout(
                        input_b.type,
                        input_b,
                        [0, 0, 0, 0],
                        [K // k, N // n, k, n],
                        [N * k, n, N, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_input_1.owner.attributes["layout_hint"] = StringAttr.get(
                        f"B_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    new_output = allo_d.transform_layout(
                        output.type,
                        output,
                        [0, 0, 0, 0],
                        [M // m, N // n, m, n],
                        [m * N, n, N, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    new_output.owner.attributes["layout_hint"] = StringAttr.get(
                        f"C_to_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    vectorized_kernel_name = call_matmul_op.attributes[
                        "lib"
                    ].value.replace("matmul_scalar_", "matmul_")
                    allo_func_d.CallOp(
                        [],
                        FlatSymbolRefAttr.get(vectorized_kernel_name),
                        [new_input_0, new_input_1, new_output],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    matmul_output = allo_d.transform_layout(
                        new_output.type,
                        new_output,
                        [0, 0, 0, 0],
                        [M // m, m, N // n, n],
                        [m * N, n, m * n, 1],
                        ip=InsertionPoint(call_matmul_op),
                    )
                    matmul_output.owner.attributes["layout_hint"] = StringAttr.get(
                        f"C_from_{M}x{N}x{K}_{m}x{n}x{k}"
                    )
                    allo_memref_d.copy(
                        matmul_output, output, ip=InsertionPoint(call_matmul_op)
                    )
                    if vectorized_kernel_name not in self.injected_external_kernels:
                        scalar_kernel: ExternalModuleBase = (
                            self.injected_external_kernels[
                                call_matmul_op.attributes["lib"].value
                            ]
                        )
                        self.injected_external_kernels[vectorized_kernel_name] = (
                            ExternalModuleBase(
                                vectorized_kernel_name,
                                scalar_kernel.input_idx,
                                scalar_kernel.output_idx,
                                scalar_kernel.kernel_code,
                                scalar_kernel.kernel_header,
                            )
                        )
                        operand_types = [x.type for x in call_matmul_op.operands]
                        func_type = allo_func_d.FunctionType.get(
                            operand_types,
                            [],
                        )
                        vectorized_kernel = allo_func_d.FuncOp(
                            vectorized_kernel_name,
                            func_type,
                            ip=InsertionPoint(function),
                        )
                        vectorized_kernel.attributes["sym_visibility"] = StringAttr.get(
                            "private"
                        )
                    call_matmul_op.erase()

        def optimize_layout_transformation(function: allo_func_d.FuncOp):
            """
            Optimize layout transformation operations
                - some can be 'push out of the function' and done at transfer time (e.g. with dma)
                - some contiguous inverse transformation can be safely removed.
            """
            node = self.virtual_computation_graph.nodes[
                func.attributes["sym_name"].value
            ]
            dead_ops = []
            op_stack_map: dict = {}
            # no need to transform if the result is unchanged
            excuse_operands = set()

            def optimize_layout_transformation_recursive(op):
                # op.operands[0] is whole zero
                if (
                    "lib" in op.attributes
                    and "fill_zeros" in op.attributes["lib"].value
                ):
                    excuse_operands.add(op.operands[0])
                    return
                if op.name == "allo.transform_layout":
                    if op.operands[0] in excuse_operands:
                        op.result.replace_all_uses_with(op.operands[0])
                        excuse_operands.remove(op.operands[0])
                        dead_ops.append(op)
                        return
                    if op.operands[0] in func.arguments:
                        arg = BlockArgument(op.operands[0])
                        if arg.arg_number in node.global_interfaces:
                            # only used once
                            transform_layout_cnt = 0
                            for use in arg.uses:
                                if (
                                    use.owner.name == "allo.transform_layout"
                                    and use.owner not in dead_ops
                                ):
                                    transform_layout_cnt += 1
                            if transform_layout_cnt == 1:
                                node.interface_layout[arg.arg_number] = (
                                    list(op.attributes["offsets"]),
                                    list(op.attributes["sizes"]),
                                    list(op.attributes["strides"]),
                                )
                                op.result.replace_all_uses_with(arg)
                                dead_ops.append(op)
                                return
                    if "layout_hint" in op.attributes:
                        layout_hint = op.attributes["layout_hint"].value
                        parts = re.findall(r"[^_]+", layout_hint)
                        if len(parts) == 4 and parts[1] == "from":
                            result_uses = list(op.result.uses)
                            if (
                                len(result_uses) == 1
                                and result_uses[0].owner.name == "memref.copy"
                                and result_uses[0].owner.operands[1] == op.operands[0]
                            ):
                                op_stack_map[op.operands[0]] = (
                                    parts[0] + parts[2] + parts[3],
                                    op,
                                    result_uses[0].owner,
                                )
                        if (
                            len(parts) == 4
                            and parts[1] == "to"
                            and op.operands[0] in op_stack_map
                        ):
                            if (
                                parts[0] + parts[2] + parts[3]
                                == op_stack_map[op.operands[0]][0]
                            ):
                                op.result.replace_all_uses_with(op.operands[0])
                                dead_ops.append(op)
                                dead_ops.append(op_stack_map[op.operands[0]][2])
                                dead_ops.append(op_stack_map[op.operands[0]][1])
                                op_stack_map.pop(op.operands[0])
                                return

                for region in op.regions:
                    for block in region.blocks:
                        # fixme: using 'stack' for nested blocks can be better
                        excuse_operands.clear()
                        op_stack_map.clear()
                        for inner_op in block.operations:
                            optimize_layout_transformation_recursive(inner_op)

            optimize_layout_transformation_recursive(function)
            for op in dead_ops:
                op.erase()

        for func in self.allo_module.body.operations:
            if isinstance(func, allo_func_d.FuncOp) and "df.kernel" in func.attributes:
                simplify_matmul_accumulate(func)
                allo_d.copy_on_write_on_function(func)
                vectorize_matmul(func)
                optimize_layout_transformation(func)

        pipeline = "builtin.module(canonicalize)"
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
        mapping_primitives: list[tuple[str, list]] = None,
        profile: bool = False,
        warmup: int = 20,
        num_iters: int = 100,
    ):
        # virtual mapping can only be applied to DAG
        if not self.computation_is_dag:
            if enable_virtual_mapping and len(mapping_primitives) > 0:
                raise ValueError(
                    "The input computation graph is not a DAG. Do not support virtual mapping now."
                )
            APIWarning(
                "The input computation graph is not a DAG. Fallback to default build."
            )
            return self.build(
                device_type=device_type,
                profile=profile,
                warmup=warmup,
                num_iters=num_iters,
            )
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

        # inject external kernels
        # (inject before virtual mapping since using external kernel may require layout transformation when transferring data)
        use_external_kernels, self.injected_external_kernels, include_src = (
            inject_external_kernels(
                self.allo_module,
                self.top_func_name,
                self.external_kernel_lib,
                "aie2" if self.device == "npu1" else "aie2p",
            )
        )
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "raw.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        self.analyze_kernel_parameters(
            self.assign_tag_to_kernel(), self.injected_external_kernels
        )
        # ------------------------- virtual mapping -------------------------
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

        # ------------------------- code optimization -------------------------
        self.allo_opt()

        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-transform-layout-ops,lower-affine",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        # ------------------------- mlir-aie code generation -------------------------
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
        self.aie_module = code_generator.aie_codegen_experimental(
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

        # ------------------------- build project -------------------------
        self.post_codegen_build(self.injected_external_kernels, include_src)
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
            device_type,
            self.global_inputs,
            self.global_outputs,
            top_func,
            self.core_func_args,
            self.streams,
        )
        self.aie_module = code_generator.aie_codegen(
            core_func_groups,
            external_funcs,
            inputs,
            outputs,
            use_external_kernels,
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
