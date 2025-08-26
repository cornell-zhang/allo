/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

using int8 = int8_t;

#define PACKED_B_SKIP_UNPACK 0

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i8_packedB(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  constexpr int r = 4, s = 16, t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;
  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        // A pointers are the same as in the original version
        const int8 *__restrict pA1 = pA + (z * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * MMUL::size_A;
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * MMUL::size_A;

        // Key: B is now tile-major. Each (s×t) tile is a contiguous 64B block
        // in L1. pB1 points to tile j, pB2 points to tile (j+1); advance by 64B
        // per K-chunk.
        const uint8_t *__restrict pB1 = reinterpret_cast<const uint8_t *>(pB) +
                                        (0 * (n / t) + j) * (MMUL::size_B / 2);
        const uint8_t *__restrict pB2 =
            reinterpret_cast<const uint8_t *>(pB) +
            (0 * (n / t) + (j + 1)) * (MMUL::size_B / 2);

        aie::vector<int8, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        aie::vector<int8, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;

        // === Load a 64B packed tile at once ===
        aie::vector<uint8_t, s * t / 2> Bp0 = aie::load_v<s * t / 2>(pB1);
        aie::vector<uint8_t, s * t / 2> Bp1 = aie::load_v<s * t / 2>(pB2);

        // directly vector_cast it to int4
        auto B01 = aie::vector_cast<int4>(Bp0);
        auto B11 = aie::vector_cast<int4>(Bp1);

        // C path same as the original version
        auto acc_C00 = aie::load_v<MMUL::size_C>(pC1);
        auto acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        auto acc_C10 = aie::load_v<MMUL::size_C>(pC2);
        auto acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        auto acc_C20 = aie::load_v<MMUL::size_C>(pC3);
        auto acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        auto acc_C30 = aie::load_v<MMUL::size_C>(pC4);
        auto acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);
        MMUL C20(acc_C20), C21(acc_C21), C30(acc_C30), C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        // Remaining blocks along K
        for (unsigned i = 1; i < (k / s); ++i) {
          A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;

          pB1 += (MMUL::size_B / 2) * (n / t);
          ; // next tile along K, step 64B
          pB2 += (MMUL::size_B / 2) * (n / t);
          ;

          Bp0 = aie::load_v<s * t / 2>(pB1);
          Bp1 = aie::load_v<s * t / 2>(pB2);

          B01 = aie::vector_cast<int4>(Bp0);
          B11 = aie::vector_cast<int4>(Bp1);

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);
        }

        aie::store_v(pC1, C00.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC3, C20.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC3, C21.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC4, C30.template to_vector<int8>());
        pC4 += MMUL::size_C;
        aie::store_v(pC4, C31.template to_vector<int8>());
        pC4 += MMUL::size_C;
      }
    }
  event1();
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_packedA_i4_packedB(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  constexpr int r = 4, s = 16, t = 8;
  using MMUL = aie::mmul<r, s, t, int8, int4, accauto>;
  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  event0();

  for (unsigned z = 0; z < (m / r); z += 4)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      int8 *__restrict pC1 = pC + (z * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC2 = pC + ((z + 1) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC3 = pC + ((z + 2) * (n / t) + 0) * MMUL::size_C;
      int8 *__restrict pC4 = pC + ((z + 3) * (n / t) + 0) * MMUL::size_C;

      for (unsigned j = 0; j < (n / t); j += 2) {
        // A pointers are the same as in the original version
        const int8 *__restrict pA1 =
            pA + (z * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA2 =
            pA + ((z + 1) * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA3 =
            pA + ((z + 2) * (k / s) + 0) * (MMUL::size_A / 2);
        const int8 *__restrict pA4 =
            pA + ((z + 3) * (k / s) + 0) * (MMUL::size_A / 2);

        // Key: B is tile-major. Each (s×t) tile is a contiguous 64B block in
        // L1. pB1 points to tile j, pB2 points to tile (j+1); advance 64B per
        // K-chunk.
        const uint8_t *__restrict pB1 = reinterpret_cast<const uint8_t *>(pB) +
                                        (0 * (n / t) + j) * (MMUL::size_B / 2);
        const uint8_t *__restrict pB2 =
            reinterpret_cast<const uint8_t *>(pB) +
            (0 * (n / t) + (j + 1)) * (MMUL::size_B / 2);

        // use vector_cast to load (MMUL::size_A / 2) int8 data as
        // (MMUL::size_A) int4; then unpack() to int8
        auto A01_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA1));
        aie::vector<int8, MMUL::size_A> A01 = aie::unpack(A01_i4);
        auto A11_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA2));
        aie::vector<int8, MMUL::size_A> A11 = aie::unpack(A11_i4);
        auto A21_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA3));
        aie::vector<int8, MMUL::size_A> A21 = aie::unpack(A21_i4);
        auto A31_i4 = aie::vector_cast<int4>(aie::load_v<r * s / 2>(pA4));
        aie::vector<int8, MMUL::size_A> A31 = aie::unpack(A31_i4);

        // === Load a 64B packed tile at once ===
        aie::vector<uint8_t, s * t / 2> Bp0 = aie::load_v<s * t / 2>(pB1);
        aie::vector<uint8_t, s * t / 2> Bp1 = aie::load_v<s * t / 2>(pB2);

        // directly vector_cast it to int4
        auto B01 = aie::vector_cast<int4>(Bp0);
        auto B11 = aie::vector_cast<int4>(Bp1);

        // C path same as the original version
        auto acc_C00 = aie::load_v<MMUL::size_C>(pC1);
        auto acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        auto acc_C10 = aie::load_v<MMUL::size_C>(pC2);
        auto acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        auto acc_C20 = aie::load_v<MMUL::size_C>(pC3);
        auto acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        auto acc_C30 = aie::load_v<MMUL::size_C>(pC4);
        auto acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00), C01(acc_C01), C10(acc_C10), C11(acc_C11);
        MMUL C20(acc_C20), C21(acc_C21), C30(acc_C30), C31(acc_C31);

        C00.mac(A01, B01);
        C01.mac(A01, B11);
        C10.mac(A11, B01);
        C11.mac(A11, B11);
        C20.mac(A21, B01);
        C21.mac(A21, B11);
        C30.mac(A31, B01);
        C31.mac(A31, B11);

        // Remaining blocks along K
        for (unsigned i = 1; i < (k / s); ++i) {
          pA1 += MMUL::size_A / 2;
          pA2 += MMUL::size_A / 2;
          pA3 += MMUL::size_A / 2;
          pA4 += MMUL::size_A / 2;

          A01_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA1));
          A01 = aie::unpack(A01_i4);
          A11_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA2));
          A11 = aie::unpack(A11_i4);
          A21_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA3));
          A21 = aie::unpack(A21_i4);
          A31_i4 = aie::vector_cast<int4>(aie::load_v<MMUL::size_A / 2>(pA4));
          A31 = aie::unpack(A31_i4);

          pB1 += (MMUL::size_B / 2) * (n / t);
          ; // next tile along K, step 64B
          pB2 += (MMUL::size_B / 2) * (n / t);
          ;

          Bp0 = aie::load_v<s * t / 2>(pB1);
          Bp1 = aie::load_v<s * t / 2>(pB2);

          B01 = aie::vector_cast<int4>(Bp0);
          B11 = aie::vector_cast<int4>(Bp1);

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);
        }

        aie::store_v(pC1, C00.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<int8>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<int8>());
        pC2 += MMUL::size_C;
        aie::store_v(pC3, C20.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC3, C21.template to_vector<int8>());
        pC3 += MMUL::size_C;
        aie::store_v(pC4, C30.template to_vector<int8>());
        pC4 += MMUL::size_C;
        aie::store_v(pC4, C31.template to_vector<int8>());
        pC4 += MMUL::size_C;
      }
    }

  event1();
}

extern "C" {
#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

// Explicit C binding for packed i4 case (row-major B)
void matmul_i8xi4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  matmul_vectorized_4x16x8_i4_i8_packedB<DIM_M, DIM_K, DIM_N>(a_in, b_in,
                                                              c_out);
}

// int4xint4, both A and B are packed
void matmul_i4_i8(int8 *a_in, int8 *b_in, int8 *c_out) {
  matmul_vectorized_4x16x8_i4_packedA_i4_packedB<DIM_M, DIM_K, DIM_N>(
      a_in, b_in, c_out);
}

} // extern "C"