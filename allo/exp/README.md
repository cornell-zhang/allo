# Allo IR Generation

This directory contains the core logic for converting Python AST into Allo Intermediate Representation (IR), which relies on MLIR.


The compilation process is divided into two distinct phases to separate concerns:

1.  **Inference Phase (`ASTProcessor`)**:
    - Handles the "dirty work" directly on the AST.
    - Responsible for type inference, semantic checking, and desugaring syntax sugar.
    - Produces a structured, fully-resolved AST where types and shapes are known (and explicitly encoded).

2.  **Builder Phase (`IRBuilder`)**:
    - Consumes the well-formed AST from the Inference Phase.
    - Focuses solely on traversal and IR construction.
    - Does not perform complex analysis; assumes the input AST is correct and fully annotated.

## Components

- **`ast_processor.py`**:
  - Implements the `ASTProcessor` class.
  - Transforms the standard Python AST into a more structured version suitable for IR generation.
  - Handles type annotations, broadcasting, and AST simplification.
  - Manages symbol tables and scoping for variables and constants.

- **`ir_builder.py`**:
  - Implements the `IRBuilder` class.
  - Traverses the processed AST to generate MLIR operations.
  - Utilizes various MLIR dialects (e.g., `arith`, `memref`, `scf`, `affine`, `linalg`) to construct the actual IR.
  - Manages insertion points and MLIR context.

- **`typing_rule.py`**:
  - Defines type inference guidelines, specifically "CPP-style" rules.
  - Handles result type deduction for boolean operations based on operand types.

- **`utils.py`**:
  - Provides utility classes such as `SymbolTable` and `Scope` to manage identifier resolution during compilation.

- **`builtin/`**:
  - A comprehensive library for handling builtin functions and operations, designed to be extensible.
  - Each builtin handler encapsulates its own type inference logic via the `infer` method.
  - Each builtin handler construct the corresponding MLIR operations via the `build` method.
