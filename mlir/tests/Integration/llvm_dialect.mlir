// Copyright HeteroCL authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: hcl-opt %s
module {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @top(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr<f32>, %arg22: !llvm.ptr<f32>, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface, top} {
    %0 = llvm.mlir.constant(16 : index) : i64
    %1 = llvm.mlir.constant(24 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(384 : index) : i64
    %4 = llvm.mlir.null : !llvm.ptr<f32>
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %6 = llvm.ptrtoint %5 : !llvm.ptr<f32> to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr<i8>
    %8 = llvm.bitcast %7 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %0, %13[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %1, %14[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %1, %15[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %2, %16[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(18 : index) : i64
    %19 = llvm.mlir.constant(24 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(432 : index) : i64
    %22 = llvm.mlir.null : !llvm.ptr<f32>
    %23 = llvm.getelementptr %22[%21] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<f32> to i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr<i8>
    %26 = llvm.bitcast %25 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %27 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.insertvalue %26, %27[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.insertvalue %30, %29[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %18, %31[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %19, %32[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %19, %33[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %20, %34[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.mlir.constant(16 : index) : i64
    %37 = llvm.mlir.constant(22 : index) : i64
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mlir.constant(352 : index) : i64
    %40 = llvm.mlir.null : !llvm.ptr<f32>
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr<f32> to i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr<i8>
    %44 = llvm.bitcast %43 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %45 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.insertvalue %44, %45[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %36, %49[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.insertvalue %37, %50[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.insertvalue %37, %51[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.insertvalue %38, %52[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.mlir.constant(22 : index) : i64
    %55 = llvm.mlir.constant(18 : index) : i64
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.mlir.constant(396 : index) : i64
    %58 = llvm.mlir.null : !llvm.ptr<f32>
    %59 = llvm.getelementptr %58[%57] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %60 = llvm.ptrtoint %59 : !llvm.ptr<f32> to i64
    %61 = llvm.call @malloc(%60) : (i64) -> !llvm.ptr<i8>
    %62 = llvm.bitcast %61 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %63 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.insertvalue %54, %67[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.insertvalue %55, %68[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.insertvalue %55, %69[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.insertvalue %56, %70[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.mlir.constant(16 : index) : i64
    %73 = llvm.mlir.constant(18 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.mlir.constant(288 : index) : i64
    %76 = llvm.mlir.null : !llvm.ptr<f32>
    %77 = llvm.getelementptr %76[%75] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %78 = llvm.ptrtoint %77 : !llvm.ptr<f32> to i64
    %79 = llvm.call @malloc(%78) : (i64) -> !llvm.ptr<i8>
    %80 = llvm.bitcast %79 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %81 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %82 = llvm.insertvalue %80, %81[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %83 = llvm.insertvalue %80, %82[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %84 = llvm.mlir.constant(0 : index) : i64
    %85 = llvm.insertvalue %84, %83[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %86 = llvm.insertvalue %72, %85[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %87 = llvm.insertvalue %73, %86[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.insertvalue %73, %87[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %89 = llvm.insertvalue %74, %88[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.mlir.constant(16 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%90 : i64)
  ^bb1(%93: i64):  // 2 preds: ^bb0, ^bb8
    %94 = llvm.icmp "slt" %93, %91 : i64
    llvm.cond_br %94, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.mlir.constant(18 : index) : i64
    %97 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%95 : i64)
  ^bb3(%98: i64):  // 2 preds: ^bb2, ^bb7
    %99 = llvm.icmp "slt" %98, %96 : i64
    llvm.cond_br %99, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.mlir.null : !llvm.ptr<f32>
    %103 = llvm.getelementptr %102[%100] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %104 = llvm.ptrtoint %103 : !llvm.ptr<f32> to i64
    %105 = llvm.call @malloc(%104) : (i64) -> !llvm.ptr<i8>
    %106 = llvm.bitcast %105 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %107 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %112 = llvm.insertvalue %100, %111[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %113 = llvm.insertvalue %101, %112[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %114 = llvm.mlir.constant(0 : index) : i64
    %115 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %116 = llvm.extractvalue %113[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %117 = llvm.getelementptr %116[%114] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %115, %117 : !llvm.ptr<f32>
    %118 = llvm.mlir.constant(0 : index) : i64
    %119 = llvm.mlir.constant(22 : index) : i64
    %120 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%118 : i64)
  ^bb5(%121: i64):  // 2 preds: ^bb4, ^bb6
    %122 = llvm.icmp "slt" %121, %119 : i64
    llvm.cond_br %122, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %123 = llvm.extractvalue %53[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %124 = llvm.mlir.constant(22 : index) : i64
    %125 = llvm.mul %93, %124  : i64
    %126 = llvm.add %125, %121  : i64
    %127 = llvm.getelementptr %123[%126] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %128 = llvm.load %127 : !llvm.ptr<f32>
    %129 = llvm.extractvalue %71[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.mlir.constant(18 : index) : i64
    %131 = llvm.mul %121, %130  : i64
    %132 = llvm.add %131, %98  : i64
    %133 = llvm.getelementptr %129[%132] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %134 = llvm.load %133 : !llvm.ptr<f32>
    %135 = llvm.fmul %128, %134  : f32
    %136 = llvm.extractvalue %113[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %137 = llvm.getelementptr %136[%114] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %138 = llvm.load %137 : !llvm.ptr<f32>
    %139 = llvm.fadd %135, %138  : f32
    %140 = llvm.extractvalue %113[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %141 = llvm.getelementptr %140[%114] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %139, %141 : !llvm.ptr<f32>
    %142 = llvm.add %121, %120  : i64
    llvm.br ^bb5(%142 : i64)
  ^bb7:  // pred: ^bb5
    %143 = llvm.mlir.constant(0 : index) : i64
    %144 = llvm.extractvalue %113[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %146 = llvm.load %145 : !llvm.ptr<f32>
    %147 = llvm.extractvalue %89[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %148 = llvm.mlir.constant(18 : index) : i64
    %149 = llvm.mul %93, %148  : i64
    %150 = llvm.add %149, %98  : i64
    %151 = llvm.getelementptr %147[%150] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %146, %151 : !llvm.ptr<f32>
    %152 = llvm.add %98, %97  : i64
    llvm.br ^bb3(%152 : i64)
  ^bb8:  // pred: ^bb3
    %153 = llvm.add %93, %92  : i64
    llvm.br ^bb1(%153 : i64)
  ^bb9:  // pred: ^bb1
    %154 = llvm.mlir.constant(16 : index) : i64
    %155 = llvm.mlir.constant(24 : index) : i64
    %156 = llvm.mlir.constant(1 : index) : i64
    %157 = llvm.mlir.constant(384 : index) : i64
    %158 = llvm.mlir.null : !llvm.ptr<f32>
    %159 = llvm.getelementptr %158[%157] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %160 = llvm.ptrtoint %159 : !llvm.ptr<f32> to i64
    %161 = llvm.call @malloc(%160) : (i64) -> !llvm.ptr<i8>
    %162 = llvm.bitcast %161 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %163 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %164 = llvm.insertvalue %162, %163[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %165 = llvm.insertvalue %162, %164[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %166 = llvm.mlir.constant(0 : index) : i64
    %167 = llvm.insertvalue %166, %165[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %168 = llvm.insertvalue %154, %167[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %169 = llvm.insertvalue %155, %168[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.insertvalue %155, %169[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %171 = llvm.insertvalue %156, %170[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %172 = llvm.mlir.constant(0 : index) : i64
    %173 = llvm.mlir.constant(16 : index) : i64
    %174 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb10(%172 : i64)
  ^bb10(%175: i64):  // 2 preds: ^bb9, ^bb17
    %176 = llvm.icmp "slt" %175, %173 : i64
    llvm.cond_br %176, ^bb11, ^bb18
  ^bb11:  // pred: ^bb10
    %177 = llvm.mlir.constant(0 : index) : i64
    %178 = llvm.mlir.constant(24 : index) : i64
    %179 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb12(%177 : i64)
  ^bb12(%180: i64):  // 2 preds: ^bb11, ^bb16
    %181 = llvm.icmp "slt" %180, %178 : i64
    llvm.cond_br %181, ^bb13, ^bb17
  ^bb13:  // pred: ^bb12
    %182 = llvm.mlir.constant(1 : index) : i64
    %183 = llvm.mlir.constant(1 : index) : i64
    %184 = llvm.mlir.null : !llvm.ptr<f32>
    %185 = llvm.getelementptr %184[%182] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %186 = llvm.ptrtoint %185 : !llvm.ptr<f32> to i64
    %187 = llvm.call @malloc(%186) : (i64) -> !llvm.ptr<i8>
    %188 = llvm.bitcast %187 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %189 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %190 = llvm.insertvalue %188, %189[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %191 = llvm.insertvalue %188, %190[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %192 = llvm.mlir.constant(0 : index) : i64
    %193 = llvm.insertvalue %192, %191[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %194 = llvm.insertvalue %182, %193[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %195 = llvm.insertvalue %183, %194[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %196 = llvm.mlir.constant(0 : index) : i64
    %197 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %198 = llvm.extractvalue %195[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %199 = llvm.getelementptr %198[%196] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %197, %199 : !llvm.ptr<f32>
    %200 = llvm.mlir.constant(0 : index) : i64
    %201 = llvm.mlir.constant(18 : index) : i64
    %202 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb14(%200 : i64)
  ^bb14(%203: i64):  // 2 preds: ^bb13, ^bb15
    %204 = llvm.icmp "slt" %203, %201 : i64
    llvm.cond_br %204, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %205 = llvm.extractvalue %89[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %206 = llvm.mlir.constant(18 : index) : i64
    %207 = llvm.mul %175, %206  : i64
    %208 = llvm.add %207, %203  : i64
    %209 = llvm.getelementptr %205[%208] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %210 = llvm.load %209 : !llvm.ptr<f32>
    %211 = llvm.extractvalue %35[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %212 = llvm.mlir.constant(24 : index) : i64
    %213 = llvm.mul %203, %212  : i64
    %214 = llvm.add %213, %180  : i64
    %215 = llvm.getelementptr %211[%214] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %216 = llvm.load %215 : !llvm.ptr<f32>
    %217 = llvm.fmul %210, %216  : f32
    %218 = llvm.extractvalue %195[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %219 = llvm.getelementptr %218[%196] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %220 = llvm.load %219 : !llvm.ptr<f32>
    %221 = llvm.fadd %217, %220  : f32
    %222 = llvm.extractvalue %195[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %223 = llvm.getelementptr %222[%196] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %221, %223 : !llvm.ptr<f32>
    %224 = llvm.add %203, %202  : i64
    llvm.br ^bb14(%224 : i64)
  ^bb16:  // pred: ^bb14
    %225 = llvm.mlir.constant(0 : index) : i64
    %226 = llvm.extractvalue %195[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %227 = llvm.getelementptr %226[%225] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %228 = llvm.load %227 : !llvm.ptr<f32>
    %229 = llvm.extractvalue %171[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %230 = llvm.mlir.constant(24 : index) : i64
    %231 = llvm.mul %175, %230  : i64
    %232 = llvm.add %231, %180  : i64
    %233 = llvm.getelementptr %229[%232] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %228, %233 : !llvm.ptr<f32>
    %234 = llvm.add %180, %179  : i64
    llvm.br ^bb12(%234 : i64)
  ^bb17:  // pred: ^bb12
    %235 = llvm.add %175, %174  : i64
    llvm.br ^bb10(%235 : i64)
  ^bb18:  // pred: ^bb10
    %236 = llvm.mlir.constant(16 : index) : i64
    %237 = llvm.mlir.constant(24 : index) : i64
    %238 = llvm.mlir.constant(1 : index) : i64
    %239 = llvm.mlir.constant(384 : index) : i64
    %240 = llvm.mlir.null : !llvm.ptr<f32>
    %241 = llvm.getelementptr %240[%239] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %242 = llvm.ptrtoint %241 : !llvm.ptr<f32> to i64
    %243 = llvm.call @malloc(%242) : (i64) -> !llvm.ptr<i8>
    %244 = llvm.bitcast %243 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %245 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %246 = llvm.insertvalue %244, %245[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %247 = llvm.insertvalue %244, %246[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %248 = llvm.mlir.constant(0 : index) : i64
    %249 = llvm.insertvalue %248, %247[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %250 = llvm.insertvalue %236, %249[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %251 = llvm.insertvalue %237, %250[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %252 = llvm.insertvalue %237, %251[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %253 = llvm.insertvalue %238, %252[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %254 = llvm.mlir.constant(0 : index) : i64
    %255 = llvm.mlir.constant(16 : index) : i64
    %256 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%254 : i64)
  ^bb19(%257: i64):  // 2 preds: ^bb18, ^bb23
    %258 = llvm.icmp "slt" %257, %255 : i64
    llvm.cond_br %258, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    %259 = llvm.mlir.constant(0 : index) : i64
    %260 = llvm.mlir.constant(24 : index) : i64
    %261 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb21(%259 : i64)
  ^bb21(%262: i64):  // 2 preds: ^bb20, ^bb22
    %263 = llvm.icmp "slt" %262, %260 : i64
    llvm.cond_br %263, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %264 = llvm.mlir.constant(1.000000e-01 : f32) : f32
    %265 = llvm.extractvalue %171[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %266 = llvm.mlir.constant(24 : index) : i64
    %267 = llvm.mul %257, %266  : i64
    %268 = llvm.add %267, %262  : i64
    %269 = llvm.getelementptr %265[%268] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %270 = llvm.load %269 : !llvm.ptr<f32>
    %271 = llvm.fmul %264, %270  : f32
    %272 = llvm.mlir.constant(1.000000e-01 : f32) : f32
    %273 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %274 = llvm.mlir.constant(24 : index) : i64
    %275 = llvm.mul %257, %274  : i64
    %276 = llvm.add %275, %262  : i64
    %277 = llvm.getelementptr %273[%276] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %278 = llvm.load %277 : !llvm.ptr<f32>
    %279 = llvm.fmul %272, %278  : f32
    %280 = llvm.fadd %271, %279  : f32
    %281 = llvm.extractvalue %253[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %282 = llvm.mlir.constant(24 : index) : i64
    %283 = llvm.mul %257, %282  : i64
    %284 = llvm.add %283, %262  : i64
    %285 = llvm.getelementptr %281[%284] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %280, %285 : !llvm.ptr<f32>
    %286 = llvm.add %262, %261  : i64
    llvm.br ^bb21(%286 : i64)
  ^bb23:  // pred: ^bb21
    %287 = llvm.add %257, %256  : i64
    llvm.br ^bb19(%287 : i64)
  ^bb24:  // pred: ^bb19
    llvm.return %253 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_top(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>) attributes {llvm.emit_c_interface, top} {
    %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.load %arg4 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.extractvalue %24[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.call @top(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %25, %26, %27, %28, %29, %30, %31) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %32, %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.return
  }
}