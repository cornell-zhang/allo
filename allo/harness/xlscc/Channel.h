// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef XLS_CHANNEL_H_
#define XLS_CHANNEL_H_

#include <queue>
#include <stdexcept>

// Channel direction enum (for XLS[cc] compatibility)
enum __xls_channel_dir {
  __xls_channel_dir_In,
  __xls_channel_dir_Out,
  __xls_channel_dir_InOut
};

// __xls_channel: FIFO channel model for streaming
// Template accepts T (type) and optional Dir (direction, ignored for sw_emu)
template <typename T, __xls_channel_dir Dir = __xls_channel_dir_InOut, int Depth = 1024>
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

// Stream aliases for XLS compatibility
template <typename T> using hls_stream = __xls_channel<T>;

#endif  // XLS_CHANNEL_H_

