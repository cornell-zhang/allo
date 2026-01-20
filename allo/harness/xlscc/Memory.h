// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// This file provides a software emulation model for XLS on-chip memory.
// The API is inspired by Google's XLS project (https://github.com/google/xls),
// specifically the memory abstractions used in xlscc.
// This is an independent implementation for Allo's sw_emu mode.

#ifndef ALLO_XLS_MEMORY_H_
#define ALLO_XLS_MEMORY_H_

#include <cassert>
#include <cstddef>
#include <vector>

// __xls_memory: On-chip memory model (SRAM/BRAM/ROM)
// Template parameters:
//   T - element type stored in memory
//   Size - number of elements (depth of memory)
template <typename T, int Size>
class __xls_memory {
  static_assert(Size > 0, "Memory size must be positive");

 public:
  __xls_memory() : data_(Size, T{}) {}
  ~__xls_memory() = default;

  // Allow copy and move (vector handles this correctly)
  __xls_memory(const __xls_memory&) = default;
  __xls_memory& operator=(const __xls_memory&) = default;
  __xls_memory(__xls_memory&&) = default;
  __xls_memory& operator=(__xls_memory&&) = default;

  // Read from memory at address
  T read(int addr) const {
    assert(addr >= 0 && addr < Size && "Memory read out of bounds");
    return data_[addr];
  }

  // Write to memory at address
  void write(int addr, const T& val) {
    assert(addr >= 0 && addr < Size && "Memory write out of bounds");
    data_[addr] = val;
  }

  // Array-style access (read/write)
  T& operator[](int addr) {
    assert(addr >= 0 && addr < Size && "Memory access out of bounds");
    return data_[addr];
  }

  const T& operator[](int addr) const {
    assert(addr >= 0 && addr < Size && "Memory access out of bounds");
    return data_[addr];
  }

  // Bulk load from external array (for test harness initialization)
  void load(const T* src, size_t count) {
    size_t n = (count < static_cast<size_t>(Size)) ? count : static_cast<size_t>(Size);
    for (size_t i = 0; i < n; ++i) {
      data_[i] = src[i];
    }
  }

  // Bulk store to external array (for test harness result extraction)
  void store(T* dst, size_t count) const {
    size_t n = (count < static_cast<size_t>(Size)) ? count : static_cast<size_t>(Size);
    for (size_t i = 0; i < n; ++i) {
      dst[i] = data_[i];
    }
  }

  // Memory info
  static constexpr int size() { return Size; }
  static constexpr size_t size_bytes() { return Size * sizeof(T); }

  // Direct data access (for debugging/testing only)
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

 private:
  std::vector<T> data_;
};

// RAM type aliases matching XLS naming conventions
template <typename T, int Size>
using xls_memory = __xls_memory<T, Size>;

// RAM_1RW: Single-port RAM (one read or write per cycle)
template <typename T, int Size>
using RAM_1RW = __xls_memory<T, Size>;

// RAM_1R1W: Dual-port RAM (one read and one write per cycle)
template <typename T, int Size>
using RAM_1R1W = __xls_memory<T, Size>;

#endif  // ALLO_XLS_MEMORY_H_

