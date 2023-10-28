module {
  func.func @matmul(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>, %arg2: tensor<512x512xbf16>) {
    %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<512x512xbf16>, tensor<512x512xbf16>) outs(%arg2 : tensor<512x512xbf16>) -> tensor<512x512xbf16>
    return
  }
}