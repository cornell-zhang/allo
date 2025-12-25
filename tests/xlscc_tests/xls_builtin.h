// Mock XLS builtin header for testing
// This provides stub implementations that allow g++ to compile the generated code

#ifndef XLS_BUILTIN_H
#define XLS_BUILTIN_H

#include <cstdint>
#include <queue>
#include <stdexcept>

// XLS channel direction markers
enum __xls_channel_dir_In {};
enum __xls_channel_dir_Out {};

// Mock channel implementation for testing
template <typename T, typename Direction>
class __xls_channel {
private:
    std::queue<T> buffer;
    bool is_input;
    
public:
    __xls_channel() : is_input(std::is_same<Direction, __xls_channel_dir_In>::value) {}
    
    // Read from channel (for input channels)
    T read() {
        if (buffer.empty()) {
            throw std::runtime_error("Channel read on empty buffer");
        }
        T val = buffer.front();
        buffer.pop();
        return val;
    }
    
    // Non-blocking read (returns false if empty)
    bool nb_read(T& val) {
        if (buffer.empty()) {
            return false;
        }
        val = buffer.front();
        buffer.pop();
        return true;
    }
    
    // Write to channel (for output channels)
    void write(const T& val) {
        buffer.push(val);
    }
    
    // Test helpers
    void push_input(const T& val) {
        buffer.push(val);
    }
    
    T pop_output() {
        if (buffer.empty()) {
            throw std::runtime_error("No output available");
        }
        T val = buffer.front();
        buffer.pop();
        return val;
    }
    
    bool empty() const { return buffer.empty(); }
    size_t size() const { return buffer.size(); }
};

// Mock memory implementation for testing
template <typename T, int SIZE>
class __xls_memory {
private:
    T data[SIZE];
    
public:
    __xls_memory() {
        for (int i = 0; i < SIZE; i++) {
            data[i] = T();
        }
    }
    
    // Array-style access
    T& operator[](int idx) {
        return data[idx];
    }
    
    const T& operator[](int idx) const {
        return data[idx];
    }
    
    // Non-blocking read (always succeeds in simulation)
    bool nb_read(int addr, T& val) {
        val = data[addr];
        return true;
    }
    
    // Non-blocking write (always succeeds in simulation)
    bool nb_write(int addr, const T& val) {
        data[addr] = val;
        return true;
    }
    
    // Blocking read/write
    T read(int addr) { return data[addr]; }
    void write(int addr, const T& val) { data[addr] = val; }
    
    // Get pointer for testing
    T* get_data() { return data; }
    const T* get_data() const { return data; }
};

#endif // XLS_BUILTIN_H

