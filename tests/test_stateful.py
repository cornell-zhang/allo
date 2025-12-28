# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import (
    bool,
    Int, 
    int8, 
    int32, 
    uint8, 
    uint32, 
    float32, 
    stateful,
)
import pytest

def test_stateless_scalar():
    """Test stateless scalar accumulator"""
    def acc_stateless(x: int32) -> int32:
        acc: int32 = 0
        acc = acc + x
        return acc

    s = allo.customize(acc_stateless)
    mod = s.build(target="llvm")
    
    # Test: stateless should always return input value (since acc starts at 0)
    result1 = mod(5)
    result2 = mod(10)
    assert result1 == 5, f"Expected 5, got {result1}"
    assert result2 == 10, f"Expected 10, got {result2}"
    print("test_stateless_scalar passed!")


def test_stateful_scalar():
    """Test stateful scalar accumulator"""
    def acc_stateful(x: int32) -> int32:
        acc: stateful(int32) = 0
        acc = acc + x
        return acc

    s = allo.customize(acc_stateful)
    mod = s.build(target="llvm")
    
    # Test: stateful should accumulate across calls
    result1 = mod(5)
    result2 = mod(10)
    result3 = mod(3)
    assert result1 == 5, f"Expected 5, got {result1}"
    assert result2 == 15, f"Expected 15, got {result2}"
    assert result3 == 18, f"Expected 18, got {result3}"
    print("test_stateful_scalar passed!")


def test_stateless_array():
    """Test stateless array buffer"""
    def array_stateless(x: float32) -> float32:
        buffer: float32[10] = 0.0
        buffer[0] = x
        buffer[1] = buffer[0] + 1.0
        return buffer[1]

    s = allo.customize(array_stateless)
    mod = s.build(target="llvm")
    
    # Test: stateless array reinitializes each call
    result1 = mod(5.0)
    result2 = mod(10.0)
    assert np.isclose(result1, 6.0), f"Expected 6.0, got {result1}"
    assert np.isclose(result2, 11.0), f"Expected 11.0, got {result2}"
    print("test_stateless_array passed!")


def test_stateful_array():
    """Test stateful array buffer"""
    def array_stateful(x: float32) -> float32:
        buffer: stateful(float32[10]) = 0.0
        buffer[0] = buffer[0] + x
        return buffer[0]

    s = allo.customize(array_stateful)
    mod = s.build(target="llvm")
    
    # Test: stateful array persists across calls
    result1 = mod(5.0)
    result2 = mod(10.0)
    result3 = mod(3.0)
    assert np.isclose(result1, 5.0), f"Expected 5.0, got {result1}"
    assert np.isclose(result2, 15.0), f"Expected 15.0, got {result2}"
    assert np.isclose(result3, 18.0), f"Expected 18.0, got {result3}"
    print("test_stateful_array passed!")


def test_moving_average():
    """Test moving average with stateful circular buffer"""
    def moving_average(new_value: float32) -> float32:
        window: stateful(float32[4]) = 0.0
        i: stateful(int32) = 0
        count: stateful(int32) = 0
        total: stateful(float32) = 0.0

        # Subtract the old value that's being replaced
        total = total - window[i]

        # Add new value to window
        window[i] = new_value
        total = total + new_value

        # Move to next position (circular buffer)
        i = (i + 1) % 4

        # Track how many values we've seen (max 4)
        if count < 4:
            count = count + 1

        # Return average of values in window
        return total / count

    s = allo.customize(moving_average)
    mod = s.build(target="llvm")
    
    # Test moving average calculation
    result1 = mod(10.0)  # avg = 10/1 = 10.0
    result2 = mod(20.0)  # avg = 30/2 = 15.0
    result3 = mod(30.0)  # avg = 60/3 = 20.0
    result4 = mod(40.0)  # avg = 100/4 = 25.0
    result5 = mod(50.0)  # avg = (20+30+40+50)/4 = 35.0
    
    assert np.isclose(result1, 10.0), f"Expected 10.0, got {result1}"
    assert np.isclose(result2, 15.0), f"Expected 15.0, got {result2}"
    assert np.isclose(result3, 20.0), f"Expected 20.0, got {result3}"
    assert np.isclose(result4, 25.0), f"Expected 25.0, got {result4}"
    assert np.isclose(result5, 35.0), f"Expected 35.0, got {result5}"
    print("test_moving_average passed!")


def test_kernel_rng():
    """Test stateful random number generator"""
    SEED = 12345
    
    def kernel_rng() -> int32:
        seed: stateful(int32) = SEED
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed & 0xFFFF

    s = allo.customize(kernel_rng)
    mod = s.build(target="llvm")
    
    # Test: RNG should produce different values each call
    results = [mod() for _ in range(5)]
    
    # Check that we got 5 different values
    unique_results = set(results)
    assert len(unique_results) == 5, f"Expected 5 unique values, got {len(unique_results)}"
    
    # Check that all values are within valid range
    for result in results:
        assert 0 <= result <= 0xFFFF, f"Result {result} out of range [0, 0xFFFF]"
    
    print(f"test_kernel_rng passed! Generated values: {results}")


def test_update_crc32():
    """Test stateful CRC32 calculation"""
    def update_crc32(data: uint8) -> uint32:
        crc: stateful(uint32) = 0xFFFFFFFF
        crc = crc ^ data
        for j in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)))
        return crc ^ 0xFFFFFFFF

    s = allo.customize(update_crc32)
    mod = s.build(target="llvm")
    
    # Test: CRC should accumulate across calls
    # Process a simple sequence of bytes
    result1 = mod(ord('A'))
    result2 = mod(ord('B'))
    result3 = mod(ord('C'))
    
    # CRC values should be different and non-zero
    assert result1 != 0, f"Expected non-zero CRC, got {result1}"
    assert result2 != result1, f"Expected different CRC values"
    assert result3 != result2, f"Expected different CRC values"
    
    print(f"test_update_crc32 passed! CRC values: {result1}, {result2}, {result3}")


def test_update_histogram():
    """Test stateful histogram"""
    def update_histogram(number: int8) -> int32[256]:
        histogram: stateful(int32[256]) = 0
        histogram[number] += 1
        return histogram

    s = allo.customize(update_histogram)
    mod = s.build(target="llvm")
    
    # Test: histogram should accumulate counts
    result1 = mod(5)
    result2 = mod(10)
    result3 = mod(5)
    result4 = mod(5)
    result5 = mod(10)
    
    # Check specific bin counts
    assert result5[5] == 3, f"Expected count of 3 for bin 5, got {result5[5]}"
    assert result5[10] == 2, f"Expected count of 2 for bin 10, got {result5[10]}"
    assert result5[0] == 0, f"Expected count of 0 for bin 0, got {result5[0]}"
    
    print("test_update_histogram passed!")


def test_simple_counter():
    """Test simple stateful counter"""
    def counter() -> int32:
        count: stateful(int32) = 0
        count = count + 1
        return count

    s = allo.customize(counter)
    mod = s.build(target="llvm")
    
    # Test: counter should increment
    results = [mod() for _ in range(10)]
    expected = list(range(1, 11))
    
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"test_simple_counter passed! Counter values: {results}")


def test_stateful_reset():
    """Test stateful accumulator with reset"""
    def acc_with_reset(x: int32, rst: bool) -> int32:
        _sum: stateful(int32) = 0
        if rst:
            _sum = 0
        else:
            _sum = _sum + x
        return _sum

    s = allo.customize(acc_with_reset)
    mod = s.build(target="llvm")
    
    # Test accumulation
    result1 = mod(5, False)
    result2 = mod(10, False)
    result3 = mod(3, False)
    
    assert result1 == 5, f"Expected 5, got {result1}"
    assert result2 == 15, f"Expected 15, got {result2}"
    assert result3 == 18, f"Expected 18, got {result3}"
    
    # Test reset
    result4 = mod(100, True)  # Reset
    result5 = mod(7, False)
    
    assert result4 == 0, f"Expected 0 after reset, got {result4}"
    assert result5 == 7, f"Expected 7, got {result5}"
    
    print("test_stateful_reset passed!")
