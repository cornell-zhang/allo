// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef XLS_MEMORY_H_
#define XLS_MEMORY_H_

#include <vector>
#include <cassert>

// __xls_memory: On-chip memory model (SRAM/BRAM)
template <typename T, int Size>
class __xls_memory {
 public:
  __xls_memory() : data_(Size) {}

  T read(int addr) const {
    assert(addr >= 0 && addr < Size);
    return data_[addr];
  }

  void write(int addr, const T& val) {
    assert(addr >= 0 && addr < Size);
    data_[addr] = val;
  }

  T& operator[](int addr) {
    assert(addr >= 0 && addr < Size);
    return data_[addr];
  }

  const T& operator[](int addr) const {
    assert(addr >= 0 && addr < Size);
    return data_[addr];
  }

  void load(const T* src, int count) {
    for (int i = 0; i < count && i < Size; ++i) {
      data_[i] = src[i];
    }
  }

  void store(T* dst, int count) const {
    for (int i = 0; i < count && i < Size; ++i) {
      dst[i] = data_[i];
    }
  }

 private:
  std::vector<T> data_;
};

#endif  // XLS_MEMORY_H_

