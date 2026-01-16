// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// This file provides a software emulation model for arbitrary-precision integers.
// The API is inspired by Google's XLS project (https://github.com/google/xls),
// specifically the synth_only headers in xls/contrib/xlscc/synth_only/.
// This is an independent implementation for Allo's sw_emu mode.

#ifndef ALLO_XLS_INT_H_
#define ALLO_XLS_INT_H_

#include <cstdint>
#include <iostream>
#include <type_traits>

// XlsInt: Arbitrary-precision integer model for software emulation
// has Width - bit width of the integer (1+, uses 64-bit storage for Width > 64)
// and Signed - whether the integer is signed (default: true)
template <int Width, bool Signed = true>
class XlsInt {
  static_assert(Width >= 1, "Width must be at least 1");

 public:
  static constexpr int width = Width;
  static constexpr bool is_signed = Signed;

  // Storage type selection based on width
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

  // Constructors
  XlsInt() : value_(0) {}
  XlsInt(int64_t val) : value_(Truncate(val)) {}  // NOLINT: implicit conversion
  XlsInt(const XlsInt& other) = default;
  XlsInt& operator=(const XlsInt& other) = default;

  XlsInt& operator=(int64_t val) {
    value_ = Truncate(val);
    return *this;
  }

  // Unary operators
  XlsInt operator-() const { return XlsInt(-value_); }
  XlsInt operator+() const { return *this; }
  XlsInt operator~() const { return XlsInt(~value_); }
  bool operator!() const { return value_ == 0; }

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
    if (Signed) {
      return XlsInt(ToSigned(value_) / ToSigned(other.value_));
    }
    return XlsInt(ToUnsigned(value_) / ToUnsigned(other.value_));
  }
  XlsInt operator%(const XlsInt& other) const {
    if (other.value_ == 0) return XlsInt(0);
    if (Signed) {
      return XlsInt(ToSigned(value_) % ToSigned(other.value_));
    }
    return XlsInt(ToUnsigned(value_) % ToUnsigned(other.value_));
  }

  // Compound assignment operators
  XlsInt& operator+=(const XlsInt& other) { return *this = *this + other; }
  XlsInt& operator-=(const XlsInt& other) { return *this = *this - other; }
  XlsInt& operator*=(const XlsInt& other) { return *this = *this * other; }
  XlsInt& operator/=(const XlsInt& other) { return *this = *this / other; }
  XlsInt& operator%=(const XlsInt& other) { return *this = *this % other; }
  XlsInt& operator&=(const XlsInt& other) { return *this = *this & other; }
  XlsInt& operator|=(const XlsInt& other) { return *this = *this | other; }
  XlsInt& operator^=(const XlsInt& other) { return *this = *this ^ other; }
  XlsInt& operator<<=(int shift) { return *this = *this << shift; }
  XlsInt& operator>>=(int shift) { return *this = *this >> shift; }

  // Increment/decrement
  XlsInt& operator++() { return *this += XlsInt(1); }
  XlsInt operator++(int) { XlsInt tmp = *this; ++*this; return tmp; }
  XlsInt& operator--() { return *this -= XlsInt(1); }
  XlsInt operator--(int) { XlsInt tmp = *this; --*this; return tmp; }

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

  // Shift operators
  XlsInt operator<<(int shift) const {
    if (shift < 0 || shift >= 64) return XlsInt(0);
    return XlsInt(value_ << shift);
  }
  XlsInt operator>>(int shift) const {
    if (shift < 0 || shift >= 64) return Signed ? XlsInt(value_ < 0 ? -1 : 0) : XlsInt(0);
    if (Signed) {
      return XlsInt(ToSigned(value_) >> shift);
    }
    return XlsInt(ToUnsigned(value_) >> shift);
  }

  // Comparison operators
  bool operator==(const XlsInt& other) const { return value_ == other.value_; }
  bool operator!=(const XlsInt& other) const { return value_ != other.value_; }
  bool operator<(const XlsInt& other) const {
    if (Signed) return ToSigned(value_) < ToSigned(other.value_);
    return ToUnsigned(value_) < ToUnsigned(other.value_);
  }
  bool operator<=(const XlsInt& other) const { return !(other < *this); }
  bool operator>(const XlsInt& other) const { return other < *this; }
  bool operator>=(const XlsInt& other) const { return !(*this < other); }

  // Type conversions
  explicit operator bool() const { return value_ != 0; }
  operator int64_t() const {  // NOLINT: implicit conversion for compatibility
    if (Signed) return SignExtend(value_);
    return ToUnsigned(value_);
  }
  int64_t to_int64() const { return static_cast<int64_t>(*this); }
  int to_int() const { return static_cast<int>(to_int64()); }
  uint64_t to_uint64() const { return ToUnsigned(value_); }

  // Bit extraction: slc<W>(start) - extract W bits starting at bit position start
  template <int W>
  XlsInt<W, false> slc(int start) const {
    static_assert(W >= 1, "Slice width must be at least 1");
    if (start < 0) return XlsInt<W, false>(0);
    uint64_t mask = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);
    return XlsInt<W, false>((ToUnsigned(value_) >> start) & mask);
  }

  // Bit insertion: set_slc(start, val) - set W bits starting at bit position start
  template <int W>
  void set_slc(int start, XlsInt<W, false> val) {
    static_assert(W >= 1, "Slice width must be at least 1");
    if (start < 0) return;
    uint64_t mask = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);
    uint64_t clear_mask = ~(mask << start);
    uint64_t insert_val = (static_cast<uint64_t>(val.to_uint64()) & mask) << start;
    value_ = Truncate((ToUnsigned(value_) & clear_mask) | insert_val);
  }

  // Single bit access
  bool operator[](int bit) const {
    if (bit < 0) return false;
    if (bit >= 64) return false;  // Can't access beyond 64-bit storage
    return (ToUnsigned(value_) >> bit) & 1;
  }

 private:
  int64_t value_;

  // Truncate to Width bits
  static int64_t Truncate(int64_t val) {
    if (Width >= 64) return val;
    uint64_t mask = (1ULL << Width) - 1;
    return val & mask;
  }

  // Sign extend from Width bits to 64 bits
  static int64_t SignExtend(int64_t val) {
    if (Width >= 64) return val;
    uint64_t mask = (1ULL << Width) - 1;
    int64_t sign_bit = 1LL << (Width - 1);
    val &= mask;
    if (val & sign_bit) {
      return val | ~mask;
    }
    return val;
  }

  // Convert to signed value (with sign extension)
  static int64_t ToSigned(int64_t val) {
    return SignExtend(val);
  }

  // Convert to unsigned value (zero extended)
  static uint64_t ToUnsigned(int64_t val) {
    if (Width >= 64) return static_cast<uint64_t>(val);
    return val & ((1ULL << Width) - 1);
  }
};

// Type aliases matching XLS[cc] generated code
template <int W, bool S = true> using ac_int = XlsInt<W, S>;
template <int W> using ac_uint = XlsInt<W, false>;

// Stream output
template <int W, bool S>
std::ostream& operator<<(std::ostream& os, const XlsInt<W, S>& val) {
  os << val.to_int64();
  return os;
}

#endif  // ALLO_XLS_INT_H_

