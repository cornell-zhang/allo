#!/bin/bash

set -e

PYTHON=python3

FILES=(
    test_mapping_basic.py       
    test_mapping_tp.py  
    test_pingpong_gemm.py      
    test_tp.py                   
    test_vector.py
    test_cannon.py                   
    test_mapping_gemm.py        
    test_matrix.py      
    test_producer_consumer.py  
    test_trace_conv.py
    test_gemm_temporal_reduction.py  
    test_mapping_large_gemm.py  
    test_norm.py        
    test_summa.py              
    test_trace_data_transfer.py
)

echo "Running ${#FILES[@]} Python scripts..."

for file in "${FILES[@]}"; do
    echo "=== Running $file ==="
    $PYTHON "$file"
    echo "--- Finished $file ---"
done

echo "All tests completed successfully."