# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLIR Translation Guide
======================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This guide will give some examples on how to invoke the MLIR toolchain to
verify the correctness of a handwritten or generated MLIR program.
"""

import allo
import numpy as np

##############################################################################
# Define an MLIR program with linalg dialect
# ------------------------------------------
# Based on the `MLIR <https://mlir.llvm.org/docs/LangRef/>`_ syntax, we can define
# an MLIR program as follows. Currently our frontend is not able to generate this
# linalg program, but we can still use it to invoke the MLIR toolchain.
#
# Basically, linalg dialect provides lots of high-level operations, and they are
# more like the NumPy operations, so we do not need to explicitly express the
# for loops inside the program, which may be easier to conduct program transformations
# for specific backends.

test_mlir_program = """
func.func @matmul(%A: memref<32x32xi32>, %B: memref<32x32xi32>) -> memref<32x32xi32> {
  %C = memref.alloc() : memref<32x32xi32>
  %c0_i32 = arith.constant 0 : i32
  linalg.fill ins(%c0_i32 : i32) outs(%C : memref<32x32xi32>)
  linalg.matmul ins(%A, %B: memref<32x32xi32>, memref<32x32xi32>)
                outs(%C: memref<32x32xi32>)
  return %C: memref<32x32xi32>
}
"""

# .. note::
#
#    For more linalg examples, please refer to the `linalg test suite <https://github.com/llvm/llvm-project/tree/main/mlir/test/Dialect/Linalg>`_.

# %%
# We wrap the MLIR parser in allo, so we can directly invoke it to parse the MLIR
# program.

mod = allo.invoke_mlir_parser(test_mlir_program)
print(mod)

# %%
# The above result should be exactly the same as what we defined in the MLIR program,
# meaning the MLIR program is valid. Otherwise, for example, if omit the return value
# of ``C``, you can see the following error message:
#
# .. code-block:: python
#
#    loc("-":8:3): error: 'func.return' op has 0 operands, but enclosing function (@matmul) returns 1
#    Traceback (most recent call last):
#      File "tutorials/developer_02_mlir.py", line 47, in <module>
#        mod = allo.invoke_mlir_parser(test_mlir_program)
#      File "/scratch/users/hc676/allo/allo/module.py", line 33, in invoke_mlir_parser
#        module = Module.parse(str(mod), ctx)
#    ValueError: Unable to parse module assembly (see diagnostics)
#
# The first line gives the error message and the exact location (line 8, column 3) of the error.
# Then we know that there is a problem in the return value of our MLIR code, which helps us debug the program.
#
# .. note::
#
#    We can also leverage the HCL-MLIR backend to parse an MLIR program from a text file.
#    For example, we can save the above program to a file named ``matmul.mlir``, and then
#    invoke the following command to parse it: (Suppose you are using our provided server)
#
#    .. code-block:: bash
#
#       $ /work/shared/common/hcl-dialect/build/bin/hcl-opt matmul.mlir
#
# We also wrap the LLVM execution engine in allo, so we can directly invoke it to execute the MLIR program.
# The ``LLVMMoudle`` class takes the MLIR module and the name of the top function as input.
# Then we can directly invoke the module with random inputs, and see if the result is correct.
#
# .. note::
#
#    To execute the MLIR with an LLVM backend, we need to lower the MLIR program to LLVM dialect first.
#    This is done inside the ``LLVMModule`` class, and you can check the details `here <https://github.com/cornell-zhang/allo/blob/main/allo/module.py>`_.
#    However, we only include several lowering passes from commonly used dialects in the module,
#    so not all the programs can be directly lowered. You will see some examples that cannot be lowered later.

llvm_mod = allo.LLVMModule(mod, "matmul")
np_A = np.random.randint(0, 10, size=(32, 32), dtype=np.int32)
np_B = np.random.randint(0, 10, size=(32, 32), dtype=np.int32)
allo_C = llvm_mod(np_A, np_B)
np.testing.assert_array_equal(allo_C, np_A @ np_B)

# %%
# We verify the correctness of our handwritten MLIR program, but we definitely don't want users to write
# these tedious IR code by hand, so we need to think about how to raise the abstraction level and let
# users write programs in a more friendly way. One thing we can do is to provide high-level programming
# abstractions like NumPy that has lots of tensor-based operations instead of elementwise ones.
# Therefore, the frontend interface may look like this:
#
# .. code-block:: python
#
#    def kernel(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
#        C = allo.matmul(A, B)
#        return C
#
# Later, we want to figure out a way to lower this high-level program to the MLIR program we defined above.

##############################################################################
# Define an MLIR program with Tensor dialect
# ------------------------------------------
# Not only for computation, we also need to raise the abstraction level for memory management.
# Currently we explicitly use ``memref`` to allocate memory and pass them to the operations.
# However, as users already write tensor programs, we should generate tensor interfaces instead.
# Thanks to the `tensor dialect <https://mlir.llvm.org/docs/Dialects/TensorOps>`_, we can
# easily leverage it to conduct slicing, reshaping, and other tensor operations. Following
# shows an example of how to use the tensor dialect to define a matmul program:

tensor_program = """
func.func @matmul(%A: tensor<32x32xi32>, %B: tensor<32x32xi32>) -> tensor<32x32xi32> {
  %C = tensor.generate {
      ^bb0(%i : index, %j : index):
          %c0_i32 = arith.constant 0 : i32
          tensor.yield %c0_i32 : i32
  } : tensor<32x32xi32>
  %1 = linalg.matmul ins(%A, %B: tensor<32x32xi32>, tensor<32x32xi32>)
                outs(%C: tensor<32x32xi32>) -> tensor<32x32xi32>
  return %1 : tensor<32x32xi32>
}
"""

# %%
# It is very similar to the original one, but the main difference is that we use ``tensor``
# instead of ``memref`` to define the input and output of the operations.
# Again, we can invoke the MLIR parser to check if the program is valid.

mod = allo.invoke_mlir_parser(tensor_program)
print(mod)

# %%
# It outputs without any error, so we know that the program is valid.
# And we can also invoke the LLVM execution engine trying to execute the program.
#
# .. code-block:: python
#
#    llvm_mod = allo.LLVMModule(mod, "matmul")
#
# You will see the following error message:
#
# .. code-block::
#
#    python3: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Loops.cpp:209: mlir::FailureOr<llvm::SmallVector<mlir::Operation*, 4> > linalgOpToLoopsImpl(mlir::PatternRewriter&, mlir::linalg::LinalgOp) [with LoopTy = mlir::AffineForOp]: Assertion `linalgOp.hasBufferSemantics() && "expected linalg op with buffer semantics"' failed.

# %%
# Unfortunately, the program cannot be lowered to LLVM dialect, because we have not added
# the lowering pass from tensor dialect to LLVM dialect, and that is something we need to do next.
