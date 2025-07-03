#!/bin/bash

set -e

PYTHON=python3

FILES=(
    # basic/test_gemm.py
    mapping/test_mapping_basic.py
    mapping/test_mapping_gemm.py
    mapping/test_mapping_tp.py
    # basic/test_matmul.py
    # basic/test_matrix_add.py
    # basic/test_pingpong_gemm.py
    # basic/test_producer_consumer.py
    # basic/test_summa.py
    # basic/test_tp.py
    # basic/test_vector.py
)

echo "Running ${#FILES[@]} Python scripts..."

for file in "${FILES[@]}"; do
    echo "=== Running $file ==="
    $PYTHON "$file"
    echo "--- Finished $file ---"
done

echo "All tests completed successfully."
