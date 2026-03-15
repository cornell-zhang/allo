#include "ir.h"
#include "nanobind/stl/unordered_map.h"

using namespace mlir;

namespace {
struct ValueProxy;

enum class ProxyState : uint32_t {
  VALID = 0,
  INVALID = 1,
  STALE = 2,
};

struct FrontendProxy {
  std::string kindStr;
  std::string hierName;
  std::string identifier;
  ProxyState state = ProxyState::VALID;

protected:
  void dumpState(llvm::raw_string_ostream &os) const;
};

struct OperationProxy : public FrontendProxy {
  OperationProxy *parent = nullptr;
  // hold the ownership of children
  std::vector<std::unique_ptr<OperationProxy>> children;
  std::vector<std::unique_ptr<ValueProxy>> values;
  Operation *inst;

  // disable copy and move operations
  OperationProxy() = default;
  OperationProxy(const OperationProxy &) = delete;
  OperationProxy &operator=(const OperationProxy &) = delete;
  OperationProxy(OperationProxy &&) noexcept = default;
  OperationProxy &operator=(OperationProxy &&) noexcept = default;

  void dump(llvm::raw_string_ostream &os) const {
    os.enable_colors(true);
    dumpImpl(os, "", true, true);
  }

private:
  void dumpImpl(llvm::raw_string_ostream &os, StringRef prefix, bool isFirst,
                bool isLast) const;
  void dumpOpType(llvm::raw_string_ostream &os) const;
};

struct ValueProxy : public FrontendProxy {
  OperationProxy *owner = nullptr;
  Value inst;
  unsigned number = 0;

  void dump(llvm::raw_string_ostream &os) const;
};

struct OperationTreeState {
  int loopCounter = 0;   // current level of loop nesting
  int branchCounter = 0; // current level of branch nesting
  int opCounter = 0;
  SmallVector<StringRef, 4> ctrlStack; // stack of control flows
};
} // namespace

void ValueProxy::dump(llvm::raw_string_ostream &os) const {
  os << "[";
  os.changeColor(llvm::raw_string_ostream::RED);
  os << kindStr;
  os.resetColor();
  os << "] ";
  os << identifier << " ";
  dumpState(os);
  os << " alias=" << hierName << "\n";
}

void OperationProxy::dumpOpType(llvm::raw_string_ostream &os) const {
  std::string ss;
  // control flows
  StringRef kindRef = kindStr;
  if (kindRef == "func.func") {
    ss += "function ";
  } else if (kindRef.contains("for") || kindRef.contains("parallel")) {
    ss += "loop ";
  } else if (kindRef.contains("if")) {
    ss += "branch ";
  }
  // effects
  if (kindRef.contains("alloc") || kindRef.contains("empty"))
    ss += "alloc ";
  if (kindRef.contains("dealloc"))
    ss += "free ";
  if (kindRef.contains("load"))
    ss += "read ";
  if (kindRef.contains("store"))
    ss += "write ";

  if (!ss.empty()) {
    ss.pop_back();
    os << "[";
    os.changeColor(llvm::raw_string_ostream::GREEN);
    os << ss;
    os.resetColor();
    os << "]";
  }
}

void FrontendProxy::dumpState(llvm::raw_string_ostream &os) const {
  if (state == ProxyState::VALID) {
    os.changeColor(llvm::raw_string_ostream::GREEN);
    os << "VALID";
  } else if (state == ProxyState::INVALID) {
    os.changeColor(llvm::raw_string_ostream::RED);
    os << "INVALID";
  } else if (state == ProxyState::STALE) {
    os.changeColor(llvm::raw_string_ostream::YELLOW);
    os << "STALE";
  }
  os.resetColor();
}

void OperationProxy::dumpImpl(llvm::raw_string_ostream &os, StringRef prefix,
                              bool isFirst, bool isLast) const {
  os << prefix;
  if (!isFirst)
    os << (isLast ? "└─ " : "├─ ");
  os.changeColor(llvm::raw_string_ostream::MAGENTA);
  os << identifier;
  os.resetColor();
  os << " ";
  dumpState(os);
  os << " alias=";
  os.changeColor((llvm::raw_string_ostream::BLUE));
  os << hierName;
  os.resetColor();
  os << " ";
  dumpOpType(os);
  os << "\n";

  for (auto &value : values) {
    os << "   │  ";
    value->dump(os);
  }

  std::string childPrefix = prefix.str();
  if (!isFirst)
    childPrefix += (isLast ? "   " : "│  ");
  for (size_t i = 0; i < children.size(); ++i) {
    children[i]->dumpImpl(os, childPrefix, false, i == children.size() - 1);
  }
}

static void collectBufferProxies(OperationProxy &opProxy) {
  for (auto v : opProxy.inst->getResults()) {
    if (isa<BaseMemRefType, TensorType>(v.getType())) {
      auto valueProxy = std::make_unique<ValueProxy>();
      valueProxy->owner = &opProxy;
      valueProxy->inst = v;
      valueProxy->kindStr = "buffer";
      valueProxy->number = v.getResultNumber();
      valueProxy->hierName =
          opProxy.hierName + ":res" + std::to_string(v.getResultNumber());
      if (auto idAttr =
              v.getDefiningOp()->getAttrOfType<StringAttr>(allo::OpIdentifier))
        valueProxy->identifier =
            idAttr.str(); // use parent op's identifier for the value if exists
      else
        valueProxy->identifier = valueProxy->hierName;
      opProxy.values.push_back(std::move(valueProxy));
    }
  }
  // special handle for func arguments
  auto funcOp = dyn_cast<FunctionOpInterface>(opProxy.inst);
  if (!funcOp)
    return;
  for (auto arg : funcOp.getArguments()) {
    if (isa<BaseMemRefType, TensorType>(arg.getType())) {
      auto valueProxy = std::make_unique<ValueProxy>();
      valueProxy->owner = &opProxy;
      valueProxy->inst = arg;
      valueProxy->kindStr = "buffer";
      valueProxy->number = arg.getArgNumber();
      valueProxy->hierName =
          opProxy.hierName + ":arg" + std::to_string(arg.getArgNumber());
      if (auto idAttr = funcOp.getArgAttrOfType<StringAttr>(arg.getArgNumber(),
                                                            allo::OpIdentifier))
        valueProxy->identifier = idAttr.str();
      else
        valueProxy->identifier = valueProxy->hierName;
      opProxy.values.push_back(std::move(valueProxy));
    }
  }
}

static std::unique_ptr<OperationProxy>
buildOperationProxy(OperationTreeState &state, Operation *op,
                    OperationProxy *parent) {
  auto proxy = std::make_unique<OperationProxy>();
  proxy->parent = parent;
  proxy->inst = op;
  proxy->kindStr = op->getName().getStringRef().str();

  bool isLoopLike = isa<LoopLikeOpInterface>(op);
  bool isIfLike = isa<affine::AffineIfOp, scf::IfOp>(op);
  bool isFuncLike = isa<FunctionOpInterface>(op);
  bool isCtrlFlow = isLoopLike || isIfLike || isFuncLike;

  // build name for the operation
  if (isLoopLike) {
    proxy->hierName =
        (state.ctrlStack.back() + ".L" + std::to_string(state.loopCounter++))
            .str();
  } else if (isIfLike) {
    proxy->hierName =
        (state.ctrlStack.back() + ".B" + std::to_string(state.branchCounter++))
            .str();
  } else if (isFuncLike) {
    proxy->hierName =
        (state.ctrlStack.back() +
         op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
             .strref())
            .str();
  } else {
    proxy->hierName =
        (state.ctrlStack.back() + ".O" + std::to_string(state.opCounter++))
            .str();
  }

  if (proxy->identifier.empty()) {
    if (auto idAttr = op->getAttrOfType<StringAttr>(allo::OpIdentifier))
      proxy->identifier = idAttr.str();
    else
      proxy->identifier = proxy->hierName;
  }

  // we only care about buffer values for now.
  collectBufferProxies(*proxy);

  if (isCtrlFlow)
    state.ctrlStack.push_back(proxy->hierName);
  // save state
  // RAII will be better though
  int loopCntGuard = state.loopCounter;
  int branchCntGuard = state.branchCounter;
  int opCntGuard = state.opCounter;
  for (unsigned regionIdx = 0; regionIdx < op->getNumRegions(); ++regionIdx) {
    Region &region = op->getRegion(regionIdx);
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (nestedOp.hasTrait<OpTrait::IsTerminator>())
          continue; // skip terminator
        proxy->children.push_back(
            buildOperationProxy(state, &nestedOp, proxy.get()));
      }
    }
  }
  // restore state
  if (isCtrlFlow)
    state.ctrlStack.pop_back();
  state.loopCounter = loopCntGuard;
  state.branchCounter = branchCntGuard;
  state.opCounter = opCntGuard;
  return proxy;
}

static std::unique_ptr<OperationProxy> buildOperationTree(ModuleOp mod) {
  auto proxy = std::make_unique<OperationProxy>();
  proxy->inst = mod;
  proxy->kindStr = mod.getOperationName();
  proxy->identifier = "allo_payload";
  proxy->hierName = "allo_payload";
  OperationTreeState state;
  state.ctrlStack.push_back("");
  for (auto &nested : mod.getOps()) {
    proxy->children.push_back(buildOperationProxy(state, &nested, proxy.get()));
  }
  return proxy;
}

static void completeHierarchyName(OperationProxy &proxy) {
  if (!isa<SymbolOpInterface>(proxy.inst)) {
    proxy.inst->setAttr(
        allo::OpIdentifier,
        StringAttr::get(proxy.inst->getContext(), proxy.hierName));
  }
  for (auto &child : proxy.children) {
    completeHierarchyName(*child);
  }
}

static void finalizeTransform(
    OperationProxy &rootProxy,
    const std::unordered_map<std::string, std::string> &identifierMap = {}) {
  assert(rootProxy.inst && isa<ModuleOp>(rootProxy.inst) &&
         "root proxy must be a module");
  // cleanup the hierarchy name for proxies whose identifier is trivial (same
  // as hierarchy name)
  MLIRContext *ctx = rootProxy.inst->getContext();
  rootProxy.inst->walk([&](Operation *op) {
    if (isa<SymbolOpInterface>(op))
      return; // skip symbol ops since their identifier must be preserved
    if (auto idAttr = op->getAttrOfType<StringAttr>(allo::OpIdentifier)) {
      auto id = idAttr.str();
      if (!identifierMap.count(id))
        op->removeAttr(allo::OpIdentifier);
      else
        op->setAttr(allo::OpIdentifier,
                    StringAttr::get(ctx, identifierMap.at(id)));
    }
  });
}

static std::string identifierAttrName = allo::OpIdentifier.data();

void bindUtils(nb::module_ &m) {

  nb::enum_<ProxyState>(m, "ProxyState")
      .value("VALID", ProxyState::VALID)
      .value("INVALID", ProxyState::INVALID)
      .value("STALE", ProxyState::STALE)
      .export_values();

  nb::class_<FrontendProxy>(m, "FrontendProxy")
      .def_prop_rw(
          "kind_str", [](FrontendProxy &self) { return self.kindStr; },
          [](FrontendProxy &self, std::string_view kind) {
            self.kindStr = kind;
          })
      .def_prop_rw(
          "hierarchy_name", [](FrontendProxy &self) { return self.hierName; },
          [](FrontendProxy &self, std::string_view name) {
            self.hierName = name;
          })
      .def_prop_rw(
          "identifier", [](FrontendProxy &self) { return self.identifier; },
          [](FrontendProxy &self, std::string_view id) {
            self.identifier = id;
          })
      .def_prop_rw(
          "state", [](FrontendProxy &self) { return self.state; },
          [](FrontendProxy &self, ProxyState state) { self.state = state; });

  nb::class_<OperationProxy, FrontendProxy>(m, "OperationProxy")
      .def_prop_rw(
          "parent", [](OperationProxy &self) { return self.parent; },
          [](OperationProxy &self, OperationProxy *parent) {
            self.parent = parent;
          },
          nb::arg("parent").none() = nullptr, nb::rv_policy::reference)
      .def_prop_ro(
          "values",
          [](OperationProxy &self) {
            std::vector<ValueProxy *> out;
            out.reserve(self.values.size());
            for (auto &v : self.values)
              out.push_back(v.get());
            return out;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "children",
          [](OperationProxy &self) {
            std::vector<OperationProxy *> out;
            out.reserve(self.children.size());
            for (auto &c : self.children)
              out.push_back(c.get());
            return out;
          },
          nb::rv_policy::reference_internal)
      .def("__str__",
           [](OperationProxy &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.dump(os);
             return os.str();
           })
      .def(
          "add_child",
          [](OperationProxy &self, unsigned index, std::string_view id,
             std::string_view kind, std::string_view hierName) {
            auto child = std::make_unique<OperationProxy>();
            child->inst = nullptr;
            child->identifier = id;
            child->hierName = hierName;
            child->kindStr = kind;
            child->parent = &self;
            child->state = ProxyState::VALID;
            OperationProxy *childPtr = child.get();
            self.children.insert(self.children.begin() + index,
                                 std::move(child));
            return childPtr;
          },
          nb::rv_policy::reference_internal, nb::arg("index"), nb::arg("id"),
          nb::arg("kind"), nb::arg("hier_name"))
      .def(
          "add_value",
          [](OperationProxy &self, std::string_view id, std::string_view kind,
             std::string_view hierName, unsigned number = 0) {
            auto value = std::make_unique<ValueProxy>();
            value->owner = &self;
            value->identifier = id;
            value->kindStr = kind;
            value->hierName = hierName;
            value->number = number;
            value->state = ProxyState::VALID;
            ValueProxy *valuePtr = value.get();
            self.values.push_back(std::move(value));
            return valuePtr;
          },
          nb::rv_policy::reference_internal, nb::arg("id"), nb::arg("kind"),
          nb::arg("hier_name"), nb::arg("number") = 0)
      .def(
          "splice",
          [](OperationProxy &self, OperationProxy &src, unsigned start = 0) {
            if (start >= src.children.size())
              throw nb::index_error("child index out of range");
            // move children from start to the end to target
            auto it = src.children.begin() + start;
            self.children.insert(self.children.end(),
                                 std::make_move_iterator(it),
                                 std::make_move_iterator(src.children.end()));
            for (auto &child : self.children)
              child->parent = &self;
            // remove the moved children from source
            src.children.erase(it, src.children.end());
          },
          nb::arg("src"), nb::arg("start") = 0);

  nb::class_<ValueProxy, FrontendProxy>(m, "ValueProxy")
      .def_prop_rw(
          "owner", [](ValueProxy &self) { return self.owner; },
          [](ValueProxy &self, OperationProxy *owner) { self.owner = owner; },
          nb::arg("owner").none() = nullptr, nb::rv_policy::reference)
      .def_prop_rw(
          "number", [](ValueProxy &self) { return self.number; },
          [](ValueProxy &self, unsigned number) { self.number = number; });

  m.def("parse_from_string", [](MLIRContext &ctx, const std::string &str) {
    ParserConfig config(&ctx, true, nullptr);
    auto mod = mlir::parseSourceString<ModuleOp>(str, config);
    if (!mod) {
      throw std::runtime_error(
          "Failed to parse MLIR string for operation tree");
    }
    return mod.release();
  });

  m.def("parse_from_file", [](MLIRContext &ctx, const std::string &filename) {
    std::string failMsg;
    auto buffer = openInputFile(filename, &failMsg);
    if (!failMsg.empty()) {
      throw std::runtime_error("Failed to open file: " + failMsg);
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
    ParserConfig config(&ctx, true, nullptr);
    auto mod = mlir::parseSourceFile<ModuleOp>(sourceMgr, config);
    if (!mod) {
      throw std::runtime_error(
          "Failed to parse MLIR file for operation tree: " + filename);
    }
    return mod.release();
  });

  m.def("build_proxy", [](ModuleOp mod) { return buildOperationTree(mod); });

  m.def(
      "finalize_transform",
      [](OperationProxy &proxy,
         const std::unordered_map<std::string, std::string> &identifierMap =
             {}) { finalizeTransform(proxy, identifierMap); },
      nb::arg("proxy"),
      nb::arg("identifier_map") =
          std::unordered_map<std::string, std::string>{});

  m.def(
      "create_operation_proxy",
      [](std::string_view identifier, std::string_view kind,
         std::string_view hierName) {
        auto proxy = std::make_unique<OperationProxy>();
        proxy->identifier = identifier;
        proxy->kindStr = kind;
        proxy->hierName = hierName;
        proxy->state = ProxyState::VALID;
        return proxy;
      },
      nb::arg("identifier"), nb::arg("kind"), nb::arg("hier_name"));

  m.def(
      "complete_hierarchy_name",
      [](OperationProxy &tree) { completeHierarchyName(tree); },
      nb::arg("tree"));

  m.attr("IDENTIFIER_ATTR_NAME") = nb::str(identifierAttrName.c_str());
}
