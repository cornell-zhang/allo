"""Debug utilities for MLIR IR inspection."""


def debug_print_ir(module):
    # just print the module out nicely
    print("MLIR MODULE")
    print(module)
    print("\nSTRUCTURED WALK")

    def walk(op, indent=0):
        print("  " * indent + op.operation.name)
        for region in op.regions:
            for block in region.blocks:
                for nested in block.operations:
                    walk(nested, indent + 1)

    walk(module.operation)
