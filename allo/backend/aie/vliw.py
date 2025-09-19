# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import os
import io
from ..._mlir.dialects import allo as allo_d
from ..._mlir.ir import (
    Context,
    Location,
    Module,
)
from ..._mlir.passmanager import PassManager

from ..config import DEFAULT_CONFIG
from ...passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    analyze_read_write_patterns,
)
from ...ir.transform import find_func_in_module
from .external_kernel import ExternalModule


class VLIWKernelFunction:
    """Wrapper for functions decorated with @vliw.kernel"""

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.vliw_module = None
        self.external_module = None
        self._created = False

    def _create_vliw_module(self):
        """Lazily create the VLIW module when first needed"""
        if not self._created:
            self.vliw_module = create_vliw_module(self.func, **self.kwargs)
            self.external_module = self.vliw_module.get_external_module()
            self._created = True

    def get_external_module(self):
        """Get the external module, creating it if necessary"""
        self._create_vliw_module()
        return self.external_module

    def __call__(self, *args, **kwargs):
        """Direct execution of the function (for testing/simulation)"""
        # For direct calls, just use the original function
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"VLIWKernelFunction({self.func.__name__})"


class VLIWDecorator:
    """Decorator class for @vliw.kernel"""

    def kernel(self, **kwargs):
        """
        Decorator to mark a function as a VLIW kernel.

        Args:
            **kwargs: Optional arguments passed to VLIWModule creation:
                     - input_idx: List of input argument indices
                     - output_idx: List of output argument indices
                     - configs: Configuration dictionary
                     - project: Project directory path
                     - save_code: Whether to save generated code

        Returns:
            VLIWKernelFunction wrapper
        """

        def decorator(func):
            return VLIWKernelFunction(func, **kwargs)

        return decorator


# Create a module-level instance that users can import
vliw = VLIWDecorator()


class VLIWModule:
    """
    AIE VLIW Module for generating C code for AMD AIE processors.

    This module takes Allo Python code, processes it through the normal frontend
    to get an MLIR module, then calls allo_d.emit_vhls to generate C code suitable
    for AMD AIE VLIW processors. The generated C code is then wrapped in an
    ExternalModule for use in AIE kernels.
    """

    def __init__(
        self,
        mod,
        top_func_name,
        input_idx=None,
        output_idx=None,
        configs=None,
        project=None,
        save_code=True,
        auto_infer_io=True,
    ):
        """
        Initialize AIE VLIW Module

        Args:
            mod: MLIR module from Allo frontend
            top_func_name: Name of the top-level function
            input_idx: List of input argument indices for ExternalModule (optional, auto-inferred if None)
            output_idx: List of output argument indices for ExternalModule (optional, auto-inferred if None)
            configs: Configuration dictionary (optional)
            project: Project directory path (optional)
            save_code: Whether to save generated code to file
            auto_infer_io: Whether to automatically infer input/output indices
        """
        self.top_func_name = top_func_name
        self.project = (
            project if project is not None else f"aie_{top_func_name}_project"
        )
        self.save_code = save_code

        # Setup configuration
        if configs is not None:
            new_configs = DEFAULT_CONFIG.copy()
            new_configs.update(configs)
            configs = new_configs
        else:
            configs = DEFAULT_CONFIG

        # Process MLIR module through standard pipeline
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.func = find_func_in_module(self.module, top_func_name)

            # Auto-infer input/output indices if not provided
            if auto_infer_io and (input_idx is None or output_idx is None):
                inferred_input_idx, inferred_output_idx = analyze_read_write_patterns(
                    self.func
                )

                # If function has a return value, the output should be the last argument
                # (since return values become additional arguments in generated C code)
                num_args = len(self.func.arguments)
                has_return = len(self.func.type.results) > 0

                if has_return and output_idx is None:
                    # Return value becomes the last argument in C function
                    self.output_idx = [num_args]
                    # All original arguments are inputs (except those identified as outputs by analysis)
                    if input_idx is None:
                        self.input_idx = list(range(num_args))
                else:
                    # No return value, use analysis results
                    self.input_idx = (
                        input_idx if input_idx is not None else inferred_input_idx
                    )
                    self.output_idx = (
                        output_idx if output_idx is not None else inferred_output_idx
                    )

                print(f"Auto-inferred indices for {top_func_name}:")
                print(f"  Input indices: {self.input_idx}")
                print(f"  Output indices: {self.output_idx}")
                if has_return:
                    print(
                        f"  (Return value detected - output index set to last argument: {self.output_idx})"
                    )
            else:
                self.input_idx = input_idx if input_idx is not None else []
                self.output_idx = output_idx if output_idx is not None else []

            # Apply standard transformations
            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)

            # Run through lowering passes (similar to HLS backend)
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                # "one-shot-bufferize{bufferize-function-boundaries},"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)

        # Generate C code using emit_vhls
        buf = io.StringIO()
        allo_d.emit_vhls(self.module, buf)
        buf.seek(0)
        self.c_code = buf.read()

        # Save code to file if requested
        self.code_file_path = None
        if save_code:
            os.makedirs(self.project, exist_ok=True)
            self.code_file_path = os.path.join(self.project, f"{top_func_name}_aie.cc")

            with open(self.code_file_path, "w", encoding="utf-8") as f:
                f.write(self._postprocess_c_code())

        # Create ExternalModule wrapper
        if self.code_file_path and (self.input_idx or self.output_idx):
            self.external_module = ExternalModule(
                top=top_func_name,
                impl_path=self.code_file_path,
                input_idx=self.input_idx,
                output_idx=self.output_idx,
            )
        else:
            self.external_module = None

    def _postprocess_c_code(self):
        """
        Post-process the generated C code for AIE VLIW processors.
        Add necessary headers and extern "C" wrapper.
        """
        processed_code = f"""/*
 * Generated VLIW C code for AMD AIE processors
 * Function: {self.top_func_name}
 */
#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {{

{"\n".join(self.c_code.splitlines()[15:])}

}} // extern "C"
"""
        return processed_code

    def get_external_module(self):
        """Get the ExternalModule wrapper for this AIE function"""
        if self.external_module is None:
            raise ValueError(
                "ExternalModule not created. Ensure input_idx and output_idx are provided "
                "and save_code is enabled."
            )
        return self.external_module

    def get_c_code(self):
        """Get the generated C code"""
        return self.c_code

    def get_processed_code(self):
        """Get the post-processed C code with AIE headers"""
        return self._postprocess_c_code()

    def __repr__(self):
        return f"VLIWModule({self.top_func_name}, inputs={self.input_idx}, outputs={self.output_idx})"

    def __call__(self, *args):
        """
        Execute the AIE function (for compatibility with HLS interface).
        This would typically be used in simulation or testing contexts.
        """
        if self.external_module is not None:
            # pylint: disable=not-callable
            return self.external_module(*args)
        raise NotImplementedError(
            "Direct execution not supported without ExternalModule wrapper. "
            "Use get_external_module() for AIE execution."
        )


def create_vliw_module(func, **kwargs):
    """
    Ultra-simple function to create an AIE VLIW module from a function.

    Args:
        func (Callable): Python function to convert to AIE (will be processed with allo.customize)
        **kwargs: Optional arguments to override defaults:
                 - input_idx: List of input argument indices (auto-inferred if None)
                 - output_idx: List of output argument indices (auto-inferred if None)
                 - configs: Configuration dictionary
                 - project: Project directory path (auto-generated if None)
                 - save_code: Whether to save generated code to file (default True)

    Returns:
        VLIWModule instance
    """
    from ...customize import customize

    s = customize(func)

    # Get function name from the schedule
    top_func_name = s.top_func_name

    # Auto-generate project name if not provided
    project = kwargs.get("project", f"aie_{top_func_name}_kernel")

    return VLIWModule(
        mod=s.module,
        top_func_name=top_func_name,
        input_idx=kwargs.get("input_idx", None),
        output_idx=kwargs.get("output_idx", None),
        configs=kwargs.get("configs", None),
        project=project,
        save_code=kwargs.get("save_code", True),
        auto_infer_io=True,
    )
