# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IR Builder Walkthrough
======================

**Author**: Hongzheng Chen (hzchen@cs.cornell.edu)

This guide will walk you through the process of translating a Python-based
Allo program to the internal MLIR representation. We will use the vector
addition example to demonstrate the process.
"""

import allo
from allo.ir.types import int32

##############################################################################
# Algorithm Definition
# --------------------
# We can define a ``matrix_add`` function as follows. In the new frontend, we
# leverage the `parsing <https://en.wikipedia.org/wiki/Parsing>`_ technique to
# translate the Python code to an MLIR program. Therefore, the first
# step is to parse the Python code to the
# `Abstract Syntax Tree (AST) <https://en.wikipedia.org/wiki/Abstract_syntax_tree>`_ representation.

M, N = 1024, 1024


def matrix_add(A: int32[M, N]) -> int32[M, N]:
    B: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        B[i, j] = A[i, j] + 1
    return B


# %%
# Python has a rich set of tools to support
# `reflection <https://en.wikipedia.org/wiki/Reflective_programming>`_.
# One of the most useful tools is the ``inspect`` module, which provides
# an API to access the source code of a Python function. We can call
# ``inspect.getsource`` to get the source code of the ``matrix_add``.

import inspect

src = inspect.getsource(matrix_add)
print(src)

# %%
# After we get the string representation of the source code, we can use
# the ``ast`` module to parse the code to an AST. The ``astpretty`` module
# can be used to print the AST in a human-readable format, which requires to
# be installed through ``pip`` separately. Otherwise, you can just use
# ``ast.dump`` to print the AST in raw format.

import ast, astpretty

tree = ast.parse(src)
astpretty.pprint(tree, indent=2, show_offsets=False)

# %%
# The AST is a tree structure that represents the syntactic structure of the
# source code. Each node is an operator or an annotation in the source code.
# For example, the ``FunctionDef`` node represents a function
# definition, and the ``AnnAssign`` node represents an annotated assignment statement.
#
# .. note::
#
#    We also wrap the above functions in ``allo.customize``, you can
#    directly call ``s = allo.customize(matrix_add, verbose=True)`` to obtain
#    the AST of the function. The entry point of the ``customize`` function is
#    located in `allo/customize.py <https://github.com/cornell-zhang/allo/blob/main/allo/customize.py>`_.

##############################################################################
# Traverse the AST
# ----------------
# After obtaining the AST, we can traverse the tree node one by one to generate the IR.
# The IR builder is inside `allo/ir/builder.py <https://github.com/cornell-zhang/allo/blob/main/allo/ir/builder.py>`_.
# Basically, the builder is a dispatcher that maps the AST node to the corresponding
# IR builder function. For example, the ``FunctionDef`` node will be mapped to
# ``ASTTransformer.build_FunctionDef``.
#
# All the builder function are ``staticmethod`` s that take in two arguments:
# an AST context and an AST node.
# The AST context stores necessary information used to build the IR, including:
#
# - ``ip_stack``: The stack of insertion points. The insertion point is used to
#   denote the current position of the IR builder. For example, when we are
#   building the body of a function, the insertion point is the function body.
# - ``buffers``: The dictionary that stores all the tensors in the program.
# - ``induction_vars``: The list of loop iterators, e.g., ``i``, ``j``, ``k``.
# - ``global_vars``: The global variables defined outside the user-defined function.
# - ``top_func``: The top-level function of the current program.
#
# The first node to traverse is the ``Module`` node, which is the root of the AST.
# We can see the ``build_Module`` function only does one thing: traverse the statements
# inside the body of the module, and recursively call ``build_stmt``.
#
# .. code-block:: python
#
#    @staticmethod
#    def build_Module(ctx, node):
#        for stmt in node.body:
#            build_stmt(ctx, stmt)

##############################################################################
# FunctionDef Node
# ^^^^^^^^^^^^^^^^
# And then we meet the ``FunctionDef`` node, which is the function definition.
# The ``build_FunctionDef`` function first creates the input and output data types
# based on users' annotations. Then, it creates a new MLIR function operation by calling
#
# .. code-block:: python
#
#    func_op = func_d.FuncOp(name=node.name, type=func_type, ip=ip, loc=loc)
#
# Here, ``func_d`` is the `func <https://mlir.llvm.org/docs/Dialects/Func/>`_ dialect defined in MLIR.
# The ``FuncOp`` is the operation that represents a function in MLIR. The function arguments are
# explained below:
#
# - ``name`` is the name of the function, and we directly use the AST ``FunctionDef`` node's name ``matrix_add`` as the operation name.
# - ``type`` is the ``FunctionType`` that defines the input and output types of the function.
# - ``ip`` is the insertion point of the function, which is the current insertion point of the AST context, and we can directly obtain it by calling ``ctx.get_ip()``.
# - ``loc`` is the actual line number of the function, which can be usually omitted.
#
# After creating the function operation, we need to create the function body. We first update the insertion point
# to the function body by calling ``ctx.push_ip(func_op.entry_block)``. Then, we traverse the function body and recursively
# call ``build_stmt``. The function arguments are inserted into the ``buffers`` for further usage.
#
# .. note::
#
#    You may probably notice the ``MockArg`` class. This is a mock class that is used to store the
#    function arguments, which are ``BlockArgument`` s in MLIR. It is different from other operations
#    that inherently have a ``result`` attribute. Therefore, we mock the ``BlockArgument`` to make
#    it consistent with other operations by providing a ``result`` property method.

##############################################################################
# AnnAssign Node
# ^^^^^^^^^^^^^^
# Next, let's visit the ``AnnAssign`` node, which is the annotated assignment statement.
# The ``build_AnnAssign`` function first evaluate the right hand side of the assignment statement
# by calling ``rhs = build_stmt(ctx, node.value)``. Then, it gets the user-defined type annotation
# to generate correct data types for the tensor. Please refer to `memref <https://mlir.llvm.org/docs/Dialects/MemRef/>`_
# dialect for more details. Similarly, we can call ``memref_d.AllocOp`` to create a new memory allocation,
# and you can see the actual ``memref.alloc`` operation in the generated MLIR code.
#
# One more thing to mention is that what we see inside the AST is just **string**, so if we want to
# get the actual value of a literal, we need to retrieve it from the ``ctx.global_vars`` dictionary.
# For example, the ``int32[M, N]`` generates the following annotation:
#
# .. code-block:: python
#
#    slice=Index(
#      value=Tuple(
#        elts=[
#          Name(id='M', ctx=Load()),
#          Name(id='N', ctx=Load()),
#        ],
#        ctx=Load(),
#      ),
#    )
#
# We can see the ``M`` and ``N`` are just the ``Name`` nodes, and we need to retrieve the actual value
# from the ``ctx.global_vars`` dictionary by calling something like ``ctx.global_vars[node.slice.value.elts[0].id]``.

##############################################################################
# For Node
# ^^^^^^^^
# The next operator is the ``For`` node, which is the for-loop statement. We provide different APIs to
# support different loop structures, so we need to further dispatch the ``For`` node to the corresponding
# builder function. For example, here we use ``allo.grid``, so it will be dispatched to ``build_grid_for``.
#
# We provide some helper functions in `allo/ir/transform.py <https://github.com/cornell-zhang/allo/blob/main/allo/ir/transform.py>`_ to make the IR creation easier.
# In this case, we can just call ``build_for_loops`` and pass in the bounds and the names of the loops
# to create a loop nest.
# Before building the loop body, we need to update the insertion point:
#
# .. code-block:: python
#
#    ctx.set_ip(for_loops[-1].body.operations[0])
#
# After calling ``build_stmts(ctx, node.body)``, we also need to recover the insertion point:
#
# .. code-block:: python
#
#    ctx.pop_ip()

##############################################################################
# Other Nodes
# ^^^^^^^^^^^
# The build process is similar for other nodes, so we will not go into them one by one.
# Please refer to the `source code <https://github.com/cornell-zhang/allo/blob/main/allo/ir/builder.py>`_ for more details.
# After building the IR, you can call ``s.module`` to see the effect.
#
# Most of the MLIR operations can be found on this `webpage <https://mlir.llvm.org/docs/Dialects/>`_, and now
# you can follow the definitions and add more amazing facilities to the new Allo compiler!
