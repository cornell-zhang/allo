#include "ir.h"

NB_MODULE(_liballo, m) {
  m.doc() = "Python bindings to the C++ Allo API";
  llvm::sys::PrintStackTraceOnErrorSignal("_liballo");
  auto ir = m.def_submodule("ir");
  init_ir(ir);
  auto arith = m.def_submodule("arith");
  init_arith_ops(arith);
  auto math = m.def_submodule("math");
  init_math_ops(math);
  auto scf = m.def_submodule("scf");
  init_scf_ops(scf);
  auto cf = m.def_submodule("cf");
  init_cf_ops(cf);
  auto ub = m.def_submodule("ub");
  init_ub_ops(ub);
  auto func = m.def_submodule("func");
  init_func_ops(func);
  auto affine = m.def_submodule("affine");
  init_affine_ops(affine);
  auto tensor = m.def_submodule("tensor");
  init_tensor_ops(tensor);
  auto memref = m.def_submodule("memref");
  init_memref_ops(memref);
  auto linalg = m.def_submodule("linalg");
  init_linalg_ops(linalg);
  auto allo = m.def_submodule("allo");
  init_allo_ir(allo);

  // lazy load for perforance
  m.def("_initialize_transform_bindings", []() {
    static bool initialized = false;
    if (initialized)
      return;

    auto self = nb::module_::import_("allo.bindings._liballo");
    auto allo_mod = nb::cast<nb::module_>(self.attr("allo"));
    init_allo_transforms(allo_mod);
    auto transform = self.def_submodule("transform");
    init_transform(transform);
    initialized = true;
  });
}
