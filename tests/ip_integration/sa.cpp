/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

void kernel_0_0(hls::stream<int8_t> &v0 /* v0[2] */,
                hls::stream<int8_t> &v1 /* v1[2] */,
                hls::stream<int8_t> &v2 /* v2[2] */,
                hls::stream<int8_t> &v3 /* v3[2] */, int16_t v4[2][2], int v5,
                int v6) { // L3
#pragma HLS stream variable = v0 depth = 2
#pragma HLS stream variable = v1 depth = 2
#pragma HLS stream variable = v2 depth = 2
#pragma HLS stream variable = v3 depth = 2
#pragma HLS array_partition variable = v4 complete dim = 1
#pragma HLS array_partition variable = v4 complete dim = 2

l_S_k_0_k:
  for (int k = 0; k < 2; k++) { // L4
    int8_t v8 = v0.read();      // v0[k];	// L5
    int8_t a;                   // L6
    a = v8;                     // L7
    int8_t v10 = v1.read();     // v1[k];	// L8
    int8_t b;                   // L9
    b = v10;                    // L10
    int8_t v12 = a;             // L11
    int16_t v13 = v12;          // L12
    int8_t v14 = b;             // L13
    int16_t v15 = v14;          // L14
    int16_t v16 = v13 * v15;    // L15
    int16_t v17 = v4[v5][v6];   // L16
    int16_t v18 = v17 + v16;    // L17
    v4[v5][v6] = v18;           // L18
    int8_t v19 = a;             // L19
    v2.write(v19);              // v2[k] = v19;	// L20
    int8_t v20 = b;             // L21
    v3.write(v20);              // v3[k] = v20;	// L22
  }
}

void kernel_1_0(hls::stream<int8_t> &v21 /* v21[2] */,
                hls::stream<int8_t> &v22 /* v22[2] */,
                hls::stream<int8_t> &v23 /* v23[2] */,
                hls::stream<int8_t> &v24 /* v24[2] */, int16_t v25[2][2],
                int v26,
                int v27) { // L26
#pragma HLS stream variable = v21 depth = 2
#pragma HLS stream variable = v22 depth = 2
#pragma HLS stream variable = v23 depth = 2
#pragma HLS stream variable = v24 depth = 2
#pragma HLS array_partition variable = v25 complete dim = 1
#pragma HLS array_partition variable = v25 complete dim = 2

l_S_k_0_k1:
  for (int k1 = 0; k1 < 2; k1++) { // L27
    int8_t v29 = v21.read();       // v21[k1];	// L28
    int8_t a1;                     // L29
    a1 = v29;                      // L30
    int8_t v31 = v22.read();       // v22[k1];	// L31
    int8_t b1;                     // L32
    b1 = v31;                      // L33
    int8_t v33 = a1;               // L34
    int16_t v34 = v33;             // L35
    int8_t v35 = b1;               // L36
    int16_t v36 = v35;             // L37
    int16_t v37 = v34 * v36;       // L38
    int16_t v38 = v25[v26][v27];   // L39
    int16_t v39 = v38 + v37;       // L40
    v25[v26][v27] = v39;           // L41
    int8_t v40 = a1;               // L42
    v23.write(v40);                // v23[k1] = v40;	// L43
    int8_t v41 = b1;               // L44
    v24.write(v41);                // v24[k1] = v41;	// L45
  }
}

void kernel_0_1(hls::stream<int8_t> &v42 /* v42[2] */,
                hls::stream<int8_t> &v43 /* v43[2] */,
                hls::stream<int8_t> &v44 /* v44[2] */,
                hls::stream<int8_t> &v45 /* v45[2] */, int16_t v46[2][2],
                int v47,
                int v48) { // L49
#pragma HLS stream variable = v42 depth = 2
#pragma HLS stream variable = v43 depth = 2
#pragma HLS stream variable = v44 depth = 2
#pragma HLS stream variable = v45 depth = 2
#pragma HLS array_partition variable = v46 complete dim = 1
#pragma HLS array_partition variable = v46 complete dim = 2

l_S_k_0_k2:
  for (int k2 = 0; k2 < 2; k2++) { // L50
    int8_t v50 = v42.read();       // v42[k2];	// L51
    int8_t a2;                     // L52
    a2 = v50;                      // L53
    int8_t v52 = v43.read();       // v43[k2];	// L54
    int8_t b2;                     // L55
    b2 = v52;                      // L56
    int8_t v54 = a2;               // L57
    int16_t v55 = v54;             // L58
    int8_t v56 = b2;               // L59
    int16_t v57 = v56;             // L60
    int16_t v58 = v55 * v57;       // L61
    int16_t v59 = v46[v47][v48];   // L62
    int16_t v60 = v59 + v58;       // L63
    v46[v47][v48] = v60;           // L64
    int8_t v61 = a2;               // L65
    v44.write(v61);                // v44[k2] = v61;	// L66
    int8_t v62 = b2;               // L67
    v45.write(v62);                // v45[k2] = v62;	// L68
  }
}

void kernel_1_1(hls::stream<int8_t> &v63 /* v63[2] */,
                hls::stream<int8_t> &v64 /* v64[2] */,
                hls::stream<int8_t> &v65 /* v65[2] */,
                hls::stream<int8_t> &v66 /* v66[2] */, int16_t v67[2][2],
                int v68,
                int v69) { // L72
#pragma HLS stream variable = v63 depth = 2
#pragma HLS stream variable = v64 depth = 2
#pragma HLS stream variable = v65 depth = 2
#pragma HLS stream variable = v66 depth = 2
#pragma HLS array_partition variable = v67 complete dim = 1
#pragma HLS array_partition variable = v67 complete dim = 2

l_S_k_0_k3:
  for (int k3 = 0; k3 < 2; k3++) { // L73
    int8_t v71 = v63.read();       // v63[k3];	// L74
    int8_t a3;                     // L75
    a3 = v71;                      // L76
    int8_t v73 = v64.read();       // v64[k3];	// L77
    int8_t b3;                     // L78
    b3 = v73;                      // L79
    int8_t v75 = a3;               // L80
    int16_t v76 = v75;             // L81
    int8_t v77 = b3;               // L82
    int16_t v78 = v77;             // L83
    int16_t v79 = v76 * v78;       // L84
    int16_t v80 = v67[v68][v69];   // L85
    int16_t v81 = v80 + v79;       // L86
    v67[v68][v69] = v81;           // L87
    int8_t v82 = a3;               // L88
    v65.write(v82);                // v65[k3] = v82;	// L89
    int8_t v83 = b3;               // L90
    v66.write(v83);                // v66[k3] = v83;	// L91
  }
}

void systolic_array(int8_t v84[2][2], int8_t v85[2][2],
                    int16_t v86[2][2]) { // L95
#pragma HLS dataflow
#pragma HLS array_partition variable = v86 complete dim = 1
#pragma HLS array_partition variable = v86 complete dim = 2

  hls::stream<int8_t> A_fifo[2][3] /* A_fifo[2][3][2] */; // L96
#pragma HLS stream variable = A_fifo depth = 2
  hls::stream<int8_t> B_fifo[2][3] /* B_fifo[2][3][2] */; // L97
#pragma HLS stream variable = B_fifo depth = 2
l_data_load_k4:
  for (int k4 = 0; k4 < 2; k4++) { // L98
  l_S_m_0_m:
    for (int m = 0; m < 2; m++) { // L99
      int8_t v91 = v84[m][k4];    // L100
      A_fifo[m][0].write(v91);    // A_fifo[m][0][k4] = v91;	// L101
    }
  l_S_n_1_n:
    for (int n = 0; n < 2; n++) { // L103
      int8_t v93 = v85[k4][n];    // L104
      B_fifo[n][0].write(v93);    // B_fifo[n][0][k4] = v93;	// L105
    }
  }
  hls::stream<int8_t> &v94 /* v94[2] */ = A_fifo[0][0];   // L109
  hls::stream<int8_t> &v95 /* v95[2] */ = B_fifo[0][0];   // L110
  hls::stream<int8_t> &v96 /* v96[2] */ = A_fifo[0][1];   // L116
  hls::stream<int8_t> &v97 /* v97[2] */ = B_fifo[0][1];   // L117
  kernel_0_0(v94, v95, v96, v97, v86, 0, 0);              // L118
  hls::stream<int8_t> &v98 /* v98[2] */ = A_fifo[0][1];   // L120
  hls::stream<int8_t> &v99 /* v99[2] */ = B_fifo[1][0];   // L121
  hls::stream<int8_t> &v100 /* v100[2] */ = A_fifo[0][2]; // L125
  hls::stream<int8_t> &v101 /* v101[2] */ = B_fifo[1][1]; // L126
  kernel_1_0(v98, v99, v100, v101, v86, 0, 1);            // L127
  hls::stream<int8_t> &v102 /* v102[2] */ = A_fifo[1][0]; // L128
  hls::stream<int8_t> &v103 /* v103[2] */ = B_fifo[0][1]; // L129
  hls::stream<int8_t> &v104 /* v104[2] */ = A_fifo[1][1]; // L130
  hls::stream<int8_t> &v105 /* v105[2] */ = B_fifo[0][2]; // L131
  kernel_0_1(v102, v103, v104, v105, v86, 1, 0);          // L132
  hls::stream<int8_t> &v106 /* v106[2] */ = A_fifo[1][1]; // L133
  hls::stream<int8_t> &v107 /* v107[2] */ = B_fifo[1][1]; // L134
  hls::stream<int8_t> &v108 /* v108[2] */ = A_fifo[1][2]; // L135
  hls::stream<int8_t> &v109 /* v109[2] */ = B_fifo[1][2]; // L136
  kernel_1_1(v106, v107, v108, v109, v86, 1, 1);          // L137
  int8_t A_drain[2];                                      // L138
  int8_t B_drain[2];                                      // L139
l_data_drain_k5:
  for (int k5 = 0; k5 < 2; k5++) { // L140
  l_S_m_4_m1:
    for (int m1 = 0; m1 < 2; m1++) {      // L141
      int8_t v114 = A_fifo[m1][2].read(); // A_fifo[m1][2][k5];	// L142
      A_drain[m1] = v114;                 // L143
    }
  l_S_n_5_n1:
    for (int n1 = 0; n1 < 2; n1++) {      // L145
      int8_t v116 = B_fifo[n1][2].read(); // B_fifo[n1][2][k5];	// L146
      B_drain[n1] = v116;                 // L147
    }
  }
}
