#include "ir.h"

using InitFunc = void (*)(nb::module_ &);

namespace {
struct SubmoduleDesc {
  std::string_view name;
  InitFunc init;
  const char *doc;
};
} // namespace

static constexpr SubmoduleDesc kSubmodules[] = {
    {"arith", bindArithOps, "arith dialect"},
    {"math", bindMathOps, "math dialect"},
    {"scf", bindSCFOps, "scf dialect"},
    {"cf", bindCFOps, "cf dialect"},
    {"ub", bindUBOps, "ub dialect"},
    {"func", bindFuncOps, "func dialect"},
    {"affine", bindAffineOps, "affine dialect"},
    {"tensor", bindTensorOps, "tensor dialect"},
    {"memref", bindMemRefOps, "memref dialect"},
    {"linalg", bindLinalgOps, "linalg dialect"},
    {"transform", bindTransform, "transform dialect"},
};

static std::once_flag loadIROnce;
static std::once_flag loadSubmoduleOnce[std::size(kSubmodules)];

static nb::module_ ensureIRLoaded(nb::module_ &parent) {
  std::call_once(loadIROnce, [&] {
    auto ir = parent.def_submodule("ir", "core IR");
    bindIR(ir);
  });
  return nb::borrow<nb::module_>(parent.attr("ir"));
}

static nb::object loadSubmodule(nb::module_ &parent, std::string_view target) {
  ensureIRLoaded(parent);

  for (size_t i = 0; i < std::size(kSubmodules); ++i) {
    const auto &d = kSubmodules[i];
    if (d.name != target)
      continue;

    std::call_once(loadSubmoduleOnce[i], [&] {
      auto sm = parent.def_submodule(d.name.data(), d.doc);
      d.init(sm);
    });

    return nb::borrow<nb::object>(parent.attr(d.name.data()));
  }

  throw nb::attribute_error("unknown submodule");
}

NB_MODULE(_liballo, m) {
  m.doc() = "Python bindings to the C++ Allo API";
  llvm::sys::PrintStackTraceOnErrorSignal("_liballo");

  ensureIRLoaded(m);

  m.def("_load_submodule", [](std::string_view name) {
    auto parent = nb::module_::import_("allo.bindings._liballo");
    return loadSubmodule(parent, name);
  });
}
