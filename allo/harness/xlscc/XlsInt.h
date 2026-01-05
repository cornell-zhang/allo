// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef XLS_INT_H_
#define XLS_INT_H_

#include <cstdint>
#include <iostream>

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

// Convenience type aliases (matching XLS[cc] naming)
// ac_int<Width, Signed> - matches XLS[cc] generated code signature
template <int W, bool S = true> using ac_int = XlsInt<W, S>;
template <int W> using ac_uint = XlsInt<W, false>;

// Utility: print XlsInt
template <int W, bool S>
std::ostream& operator<<(std::ostream& os, const XlsInt<W, S>& val) {
  os << static_cast<int64_t>(val);
  return os;
}

#endif  // XLS_INT_H_

