// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// This file provides a software emulation model for XLS channels.
// The API is inspired by Google's XLS project (https://github.com/google/xls),
// specifically the channel definitions in xls/contrib/xlscc/synth_only/.
// This is an independent implementation for Allo's sw_emu mode.

#ifndef ALLO_XLS_CHANNEL_H_
#define ALLO_XLS_CHANNEL_H_

#include <cstddef>
#include <deque>
#include <stdexcept>

// Channel direction enum (matches XLS[cc] naming)
enum __xls_channel_dir {
  __xls_channel_dir_In,
  __xls_channel_dir_Out,
  __xls_channel_dir_InOut
};

// __xls_channel: FIFO channel model for streaming data between blocks
// Template parameters:
//   T - data type carried by the channel
//   Dir - channel direction (In/Out/InOut), used for synthesis, ignored in sw_emu
//   Depth - maximum FIFO depth (default: 1024 for sw_emu flexibility)
template <typename T, __xls_channel_dir Dir = __xls_channel_dir_InOut, int Depth = 1024>
class __xls_channel {
 public:
  __xls_channel() = default;
  ~__xls_channel() = default;

  // Disable copy/move to match hardware channel semantics
  __xls_channel(const __xls_channel&) = delete;
  __xls_channel& operator=(const __xls_channel&) = delete;
  __xls_channel(__xls_channel&&) = delete;
  __xls_channel& operator=(__xls_channel&&) = delete;

  // Blocking write - sends data to the channel
  // In hardware: blocks until channel has space
  // In sw_emu: throws if channel is full (to catch deadlocks)
  void write(const T& val) {
    if (fifo_.size() >= static_cast<size_t>(Depth)) {
      throw std::runtime_error("__xls_channel: write to full channel (potential deadlock)");
    }
    fifo_.push_back(val);
  }

  // Blocking read - receives data from the channel
  // In hardware: blocks until data is available
  // In sw_emu: throws if channel is empty (to catch deadlocks)
  T read() {
    if (fifo_.empty()) {
      throw std::runtime_error("__xls_channel: read from empty channel (potential deadlock)");
    }
    T val = fifo_.front();
    fifo_.pop_front();
    return val;
  }

  // Non-blocking read - attempts to read without blocking
  // Returns true if successful, false if channel is empty
  bool nb_read(T& val) {
    if (fifo_.empty()) return false;
    val = fifo_.front();
    fifo_.pop_front();
    return true;
  }

  // Non-blocking write - attempts to write without blocking
  // Returns true if successful, false if channel is full
  bool nb_write(const T& val) {
    if (fifo_.size() >= static_cast<size_t>(Depth)) return false;
    fifo_.push_back(val);
    return true;
  }

  // Channel state queries
  bool empty() const { return fifo_.empty(); }
  bool full() const { return fifo_.size() >= static_cast<size_t>(Depth); }
  size_t size() const { return fifo_.size(); }
  static constexpr int depth() { return Depth; }

 private:
  std::deque<T> fifo_;
};

// Convenience aliases
template <typename T, int Depth = 1024>
using xls_channel = __xls_channel<T, __xls_channel_dir_InOut, Depth>;

#endif  // ALLO_XLS_CHANNEL_H_

