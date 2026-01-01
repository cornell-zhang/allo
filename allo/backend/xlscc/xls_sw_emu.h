// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef XLS_SW_EMU_H_
#define XLS_SW_EMU_H_

#include <cstdint>
#include <queue>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>

// XlsInt: Arbitrary-precision integer model
template <int Width, bool Signed = true>
class XlsInt {
 public:
  static constexpr int kWidth = Width;
  static constexpr bool kSigned = Signed;

  using StorageType = typename std::conditional<
      (Width <= 8), int8_t,
      typename std::conditional<
          (Width <= 16), int16_t,
          typename std::conditional<
              (Width <= 32), int32_t,
              int64_t>::type>::type>::type;

  using UStorageType = typename std::conditional<
      (Width <= 8), uint8_t,
      typename std::conditional<
          (Width <= 16), uint16_t,
          typename std::conditional<
              (Width <= 32), uint32_t,
              uint64_t>::type>::type>::type;

  XlsInt() : value_(0) {}
  XlsInt(int64_t val) : value_(Mask(val)) {}
  XlsInt(const XlsInt& other) : value_(other.value_) {}

  XlsInt& operator=(const XlsInt& other) {
    value_ = other.value_;
    return *this;
  }

  XlsInt& operator=(int64_t val) {
    value_ = Mask(val);
    return *this;
  }

  // Arithmetic operators
  XlsInt operator+(const XlsInt& other) const {
    return XlsInt(value_ + other.value_);
  }

  XlsInt operator-(const XlsInt& other) const {
    return XlsInt(value_ - other.value_);
  }

  XlsInt operator*(const XlsInt& other) const {
    return XlsInt(value_ * other.value_);
  }

  XlsInt operator/(const XlsInt& other) const {
    if (other.value_ == 0) return XlsInt(0);
    return XlsInt(value_ / other.value_);
  }

  XlsInt operator%(const XlsInt& other) const {
    if (other.value_ == 0) return XlsInt(0);
    return XlsInt(value_ % other.value_);
  }

  // Compound assignment
  XlsInt& operator+=(const XlsInt& other) {
    value_ = Mask(value_ + other.value_);
    return *this;
  }

  XlsInt& operator-=(const XlsInt& other) {
    value_ = Mask(value_ - other.value_);
    return *this;
  }

  XlsInt& operator*=(const XlsInt& other) {
    value_ = Mask(value_ * other.value_);
    return *this;
  }

  // Bitwise operators
  XlsInt operator&(const XlsInt& other) const {
    return XlsInt(value_ & other.value_);
  }

  XlsInt operator|(const XlsInt& other) const {
    return XlsInt(value_ | other.value_);
  }

  XlsInt operator^(const XlsInt& other) const {
    return XlsInt(value_ ^ other.value_);
  }

  XlsInt operator~() const {
    return XlsInt(~value_);
  }

  XlsInt operator<<(int shift) const {
    return XlsInt(value_ << shift);
  }

  XlsInt operator>>(int shift) const {
    if (Signed) {
      return XlsInt(static_cast<StorageType>(value_) >> shift);
    }
    return XlsInt(static_cast<UStorageType>(value_) >> shift);
  }

  // Comparison operators
  bool operator==(const XlsInt& other) const { return value_ == other.value_; }
  bool operator!=(const XlsInt& other) const { return value_ != other.value_; }
  bool operator<(const XlsInt& other) const {
    if (Signed) {
      return static_cast<StorageType>(value_) < static_cast<StorageType>(other.value_);
    }
    return static_cast<UStorageType>(value_) < static_cast<UStorageType>(other.value_);
  }
  bool operator<=(const XlsInt& other) const { return !(other < *this); }
  bool operator>(const XlsInt& other) const { return other < *this; }
  bool operator>=(const XlsInt& other) const { return !(*this < other); }

  // Conversion
  operator int64_t() const {
    if (Signed) {
      return SignExtend(value_);
    }
    return value_ & ((1ULL << Width) - 1);
  }

  int64_t to_int64() const { return static_cast<int64_t>(*this); }

  // Bit slicing: slc<W>(start) - extract W bits starting at position start
  template <int W>
  XlsInt<W, false> slc(int start) const {
    uint64_t mask = (1ULL << W) - 1;
    return XlsInt<W, false>((value_ >> start) & mask);
  }

  // Set bits: set_slc(start, val) - set bits starting at position start
  template <int W>
  void set_slc(int start, XlsInt<W, false> val) {
    uint64_t mask = ((1ULL << W) - 1) << start;
    value_ = (value_ & ~mask) | ((static_cast<uint64_t>(val.to_int64()) << start) & mask);
    value_ = Mask(value_);
  }

 private:
  int64_t value_;

  static int64_t Mask(int64_t val) {
    if (Width >= 64) return val;
    uint64_t mask = (1ULL << Width) - 1;
    return val & mask;
  }

  static int64_t SignExtend(int64_t val) {
    if (Width >= 64) return val;
    int64_t sign_bit = 1LL << (Width - 1);
    if (val & sign_bit) {
      return val | (~((1ULL << Width) - 1));
    }
    return val & ((1ULL << Width) - 1);
  }
};

// __xls_channel: FIFO channel model for streaming
template <typename T, int Depth = 1024>
class __xls_channel {
 public:
  __xls_channel() {}

  void write(const T& val) {
    if (fifo_.size() >= Depth) {
      throw std::runtime_error("Channel overflow");
    }
    fifo_.push(val);
  }

  T read() {
    if (fifo_.empty()) {
      throw std::runtime_error("Channel underflow");
    }
    T val = fifo_.front();
    fifo_.pop();
    return val;
  }

  bool nb_read(T& val) {
    if (fifo_.empty()) return false;
    val = fifo_.front();
    fifo_.pop();
    return true;
  }

  bool nb_write(const T& val) {
    if (fifo_.size() >= Depth) return false;
    fifo_.push(val);
    return true;
  }

  bool empty() const { return fifo_.empty(); }
  bool full() const { return fifo_.size() >= Depth; }
  size_t size() const { return fifo_.size(); }

 private:
  std::queue<T> fifo_;
};

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

// Convenience type aliases (matching XLS[cc] naming)
// ac_int<Width, Signed> - matches XLS[cc] generated code signature
template <int W, bool S = true> using ac_int = XlsInt<W, S>;
template <int W> using ac_uint = XlsInt<W, false>;

// Stream aliases for XLS compatibility
template <typename T> using hls_stream = __xls_channel<T>;

// Utility: print XlsInt
template <int W, bool S>
std::ostream& operator<<(std::ostream& os, const XlsInt<W, S>& val) {
  os << static_cast<int64_t>(val);
  return os;
}

#endif  // XLS_SW_EMU_H_

