#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

PYTHON=python3

FILES=(
    test_cannon.py                   
    test_gemm_temporal_reduction.py  
    test_low_bit_gemm.py
    test_mapping_basic.py       
    test_mapping_gemm.py        
    # test_mapping_large_gemm.py  
    test_mapping_multi_gemm.py  
    test_mapping_tp.py  
    test_matrix.py      
    test_meta_for.py      
    test_multi_output.py      
    test_norm.py        
    test_pingpong_gemm.py      
    test_producer_consumer.py  
    test_slice.py
    test_summa.py
    test_tp.py                   
    # test_trace_conv.py
    test_trace_data_transfer.py
    test_vector.py
    # test_vliw.py
)

echo "Running ${#FILES[@]} Python scripts..."

for file in "${FILES[@]}"; do
    echo "=== Running $file ==="
    $PYTHON "$file"
    echo "--- Finished $file ---"
done

echo "All tests completed successfully."
