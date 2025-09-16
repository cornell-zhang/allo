module attributes {transform.with_named_sequence} {
  transform.named_sequence @outline_affine_loops(%arg0: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    %loops = transform.structured.match ops{["affine.for"]} 
             attributes{top_level} in %func 
             : (!transform.any_op) -> !transform.any_op
    
    %outlined_funcs, %calls = transform.loop.outline %loops {func_name = "affine_kernel"} 
                             : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}