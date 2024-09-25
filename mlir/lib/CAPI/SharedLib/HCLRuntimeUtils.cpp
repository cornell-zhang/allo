/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl-c/SharedLib/HCLRuntimeUtils.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// reference:
// https://github.com/llvm/llvm-project/blob/bd672e2fc03823e536866da6721b9f053cfd586b/mlir/lib/ExecutionEngine/CRunnerUtils.cpp#L59

#define MAX_LINE_LENGTH 4096

template <typename T> void readMemref(int64_t rank, void *ptr, char *str) {
  // Open the input file
  FILE *fp = fopen(str, "r");
  if (!fp) {
    perror("Error opening file");
    return;
  }

  // Read the file line by line
  char line[MAX_LINE_LENGTH];
  T *array = NULL;
  int array_size = 0;
  while (fgets(line, MAX_LINE_LENGTH, fp)) {
    // Parse the line and add the values to the array
    char *token = strtok(line, ",");
    while (token) {
      // Resize the array if necessary
      if (array_size % 8 == 0) {
        array = (T *)realloc(array, (array_size + 8) * sizeof(T));
      }
      // Convert the token to a float and add it to the array
      array[array_size++] = atof(token);
      // Get the next token
      token = strtok(NULL, ",");
    }
  }

  // Close the file
  fclose(fp);

  // Print the array
  printf("LoadMemref: array loaded from file (%d elements):\n", array_size);
  for (int i = 0; i < array_size; i++) {
    std::cout << array[i] << " ";
  }
  printf("\n");

  // Copy data from array to memref buffer
  UnrankedMemRefType<T> unranked_memref = {rank, ptr};
  DynamicMemRefType<T> memref(unranked_memref);
  memcpy(memref.data, array, array_size * sizeof(T));

  // Free the array
  free(array);
}

extern "C" void readMemrefI32(int64_t rank, void *ptr, char *str) {
  readMemref<int32_t>(rank, ptr, str);
}

extern "C" void readMemrefI64(int64_t rank, void *ptr, char *str) {
  readMemref<int64_t>(rank, ptr, str);
}

extern "C" void readMemrefF32(int64_t rank, void *ptr, char *str) {
  readMemref<float>(rank, ptr, str);
}

extern "C" void readMemrefF64(int64_t rank, void *ptr, char *str) {
  readMemref<double>(rank, ptr, str);
}

template <typename T>
void writeMemref(int64_t rank, void *ptr, char *filename, std::string fmt) {
  // Define the array and its size
  UnrankedMemRefType<T> unranked_memref = {rank, ptr};
  DynamicMemRefType<T> memref(unranked_memref);
  int array_size = 1;
  for (int i = 0; i < rank; i++) {
    array_size *= memref.sizes[i];
  }
  T *array = (T *)malloc(array_size * sizeof(T));
  memcpy(array, memref.data, array_size * sizeof(T));

  // Print a message saying writing to file
  printf("Writing memref to file: %s\n", filename);

  // Open the file for writing
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    // File opening failed
    printf("Error opening file!\n");
    return;
  }

  // Write the array to the file, with comma separators
  for (int i = 0; i < array_size; i++) {
    fprintf(fp, fmt.c_str(), array[i]);
    if (i < array_size - 1) {
      // Add a comma separator, unless this is the last element
      fprintf(fp, ", ");
    }
  }

  // Close the file
  fclose(fp);

  // Free the array
  free(array);
}

extern "C" void writeMemrefI32(int64_t rank, void *ptr, char *str) {
  writeMemref<int32_t>(rank, ptr, str, "%d");
}

extern "C" void writeMemrefI64(int64_t rank, void *ptr, char *str) {
  writeMemref<int64_t>(rank, ptr, str, "%ld");
}

extern "C" void writeMemrefF32(int64_t rank, void *ptr, char *str) {
  writeMemref<float>(rank, ptr, str, "%.6f ");
}

extern "C" void writeMemrefF64(int64_t rank, void *ptr, char *str) {
  writeMemref<double>(rank, ptr, str, "%.6f ");
}