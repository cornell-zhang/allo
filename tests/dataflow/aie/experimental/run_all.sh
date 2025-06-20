#!/bin/bash

set -e

PYTHON=python3

FILES=(
    test_gemm.py
    test_mapping_basic.py
    test_mapping_gemm.py
    test_mapping_tp.py
    test_matmul.py
    test_matrix_add.py
    test_pingpong_gemm.py
    test_producer_consumer.py
    test_summa.py
    test_tp.py
    test_vector.py
)

echo "Running ${#FILES[@]} Python scripts..."

for file in "${FILES[@]}"; do
    echo "=== Running $file ==="
    $PYTHON "$file"
    echo "--- Finished $file ---"
done

echo "All tests completed successfully."
