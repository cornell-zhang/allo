# Allo Builtin Library Extensions

This directory serves as an extensible library for handling builtin functions and operations within the Allo compiler.

## Extensibility

The builtin system is designed to be easily extensible. New builtin functions can be added by implementing a handler class that inherits from `BuiltinHandler` and decorating it with `@register_builtin_handler`. This allows the compiler to support new operations without modifying the core `IRBuilder`.

## Files

- **`handler.py`**:
  - Defines the abstract base class `BuiltinHandler`.
  - Implements the registration mechanism via the `register_builtin_handler` decorator.
  - Maintains the `BUILTIN_HANDLERS` registry mapping function names to their respective handlers.

- **`arith.py`**:
  - Implements handlers for standard arithmetic operations (e.g., `Add`, `Sub`, `Mult`, `Div`, `Mod`, `FloorDiv`).
  - Dispatches operations to appropriate MLIR dialects (`arith`, `allo`, `linalg`) based on input types (scalar vs. tensor, integer vs. float).
  - Handles signed/unsigned attributes and broadcasting requirements.

- **`value.py`**:
  - Provides handlers for value-related operations such as `constant` creation and `broadcast`.
