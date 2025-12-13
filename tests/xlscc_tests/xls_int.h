// Mock XLS integer types for testing
// This provides a simplified XlsInt/ac_int that works with g++

#ifndef XLS_INT_H
#define XLS_INT_H

#include <cstdint>
#include <type_traits>

// XlsInt: Arbitrary width integer type
// For testing purposes, we use native int64_t with masking for width
template <int Width, bool Signed = true>
class XlsInt {
private:
    int64_t value;
    
    // Mask to limit to Width bits (capped at 64)
    static constexpr uint64_t mask() {
        constexpr int EffWidth = (Width > 63) ? 63 : Width;
        return (1ULL << EffWidth) - 1;
    }
    
    // Sign extend if signed
    int64_t normalize(int64_t v) const {
        if (Width >= 64) return v;  // No masking needed for 64+ bit
        v &= mask();
        constexpr int EffWidth = (Width > 63) ? 63 : Width;
        if (Signed && (v & (1ULL << (EffWidth - 1)))) {
            // Sign extend
            v |= ~mask();
        }
        return v;
    }
    
public:
    XlsInt() : value(0) {}
    XlsInt(int v) : value(normalize(v)) {}
    XlsInt(int64_t v) : value(normalize(v)) {}
    XlsInt(unsigned int v) : value(normalize(static_cast<int64_t>(v))) {}
    XlsInt(uint64_t v) : value(normalize(static_cast<int64_t>(v))) {}
    
    // Cross-width copy constructor (for width extension/truncation)
    template <int OtherWidth, bool OtherSigned>
    XlsInt(const XlsInt<OtherWidth, OtherSigned>& other) : value(normalize(other.raw())) {}
    
    // Conversion operators
    operator int() const { return static_cast<int>(value); }
    operator int64_t() const { return value; }
    
    // Arithmetic operators (with same type)
    XlsInt operator+(const XlsInt& other) const {
        return XlsInt(value + other.value);
    }
    XlsInt operator-(const XlsInt& other) const {
        return XlsInt(value - other.value);
    }
    XlsInt operator*(const XlsInt& other) const {
        return XlsInt(value * other.value);
    }
    XlsInt operator/(const XlsInt& other) const {
        return XlsInt(value / other.value);
    }
    XlsInt operator%(const XlsInt& other) const {
        return XlsInt(value % other.value);
    }
    
    // Arithmetic operators with int (for constant multiplication etc.)
    XlsInt operator+(int other) const { return XlsInt(value + other); }
    XlsInt operator-(int other) const { return XlsInt(value - other); }
    XlsInt operator*(int other) const { return XlsInt(value * other); }
    XlsInt operator/(int other) const { return XlsInt(value / other); }
    XlsInt operator%(int other) const { return XlsInt(value % other); }
    
    // Cross-width arithmetic operators
    template <int OtherWidth, bool OtherSigned>
    XlsInt operator+(const XlsInt<OtherWidth, OtherSigned>& other) const {
        return XlsInt(value + other.raw());
    }
    template <int OtherWidth, bool OtherSigned>
    XlsInt operator-(const XlsInt<OtherWidth, OtherSigned>& other) const {
        return XlsInt(value - other.raw());
    }
    template <int OtherWidth, bool OtherSigned>
    XlsInt operator*(const XlsInt<OtherWidth, OtherSigned>& other) const {
        return XlsInt(value * other.raw());
    }
    template <int OtherWidth, bool OtherSigned>
    XlsInt operator/(const XlsInt<OtherWidth, OtherSigned>& other) const {
        return XlsInt(value / other.raw());
    }
    
    // Bitwise operators
    XlsInt operator&(const XlsInt& other) const {
        return XlsInt(value & other.value);
    }
    XlsInt operator|(const XlsInt& other) const {
        return XlsInt(value | other.value);
    }
    XlsInt operator^(const XlsInt& other) const {
        return XlsInt(value ^ other.value);
    }
    XlsInt operator~() const {
        return XlsInt(~value);
    }
    XlsInt operator<<(int shift) const {
        return XlsInt(value << shift);
    }
    XlsInt operator>>(int shift) const {
        if (Signed) {
            return XlsInt(value >> shift);
        } else {
            return XlsInt(static_cast<int64_t>(static_cast<uint64_t>(value & mask()) >> shift));
        }
    }
    
    // Comparison operators
    bool operator==(const XlsInt& other) const { return value == other.value; }
    bool operator!=(const XlsInt& other) const { return value != other.value; }
    bool operator<(const XlsInt& other) const { return value < other.value; }
    bool operator<=(const XlsInt& other) const { return value <= other.value; }
    bool operator>(const XlsInt& other) const { return value > other.value; }
    bool operator>=(const XlsInt& other) const { return value >= other.value; }
    
    // Assignment operators
    XlsInt& operator=(int v) { value = normalize(v); return *this; }
    XlsInt& operator+=(const XlsInt& other) { value = normalize(value + other.value); return *this; }
    XlsInt& operator-=(const XlsInt& other) { value = normalize(value - other.value); return *this; }
    XlsInt& operator*=(const XlsInt& other) { value = normalize(value * other.value); return *this; }
    XlsInt& operator/=(const XlsInt& other) { value = normalize(value / other.value); return *this; }
    
    // Pre/post increment/decrement
    XlsInt& operator++() { value = normalize(value + 1); return *this; }
    XlsInt operator++(int) { XlsInt tmp = *this; ++(*this); return tmp; }
    XlsInt& operator--() { value = normalize(value - 1); return *this; }
    XlsInt operator--(int) { XlsInt tmp = *this; --(*this); return tmp; }
    
    // Get raw value
    int64_t raw() const { return value; }
};

// ac_int alias (common in HLS)
template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

#endif // XLS_INT_H

