#include "ir.h"
#include "allo/TransformOps/Utils.h"
#include <set>
using namespace mlir;

OpPrintingFlags getOpPrintingFlags(bool enable_debug = false) {
  auto printingFlags = OpPrintingFlags();
  printingFlags.enableDebugInfo(enable_debug);
  printingFlags.printNameLocAsPrefix(true);
  printingFlags.printGenericOpForm(false);
  return printingFlags;
}

namespace {
struct ProxyValueInfo {
  Value value;
  std::string value_identifier;
  std::string owner_op_identifier;
  std::string owner_op_kind;
  std::string source_kind;
  int64_t source_index = -1;
  std::string type_str;
  bool is_memref = false;
  std::string root_kind = "unknown";
  std::string root_owner_identifier;
  int64_t root_arg_number = -1;
};

struct ProxyNode {
  ProxyNode *parent = nullptr;
  std::vector<std::unique_ptr<ProxyNode>> children;
  std::vector<ProxyValueInfo> values;
  Operation *op = nullptr;
  std::string op_kind;
  std::string hierarchy_name;
  std::string op_identifier;

  ProxyNode() = default;
  ProxyNode(const ProxyNode &) = delete;
  ProxyNode &operator=(const ProxyNode &) = delete;
  ProxyNode(ProxyNode &&) noexcept = default;
  ProxyNode &operator=(ProxyNode &&) noexcept = default;

  void dump(llvm::raw_string_ostream &os, int indent = 0) const {
    std::string indentStr(indent, ' ');
    dumpHelper(os, indentStr, /*isLast=*/true, /*isRoot=*/true);
  }

private:
  static void printPtr(llvm::raw_string_ostream &os, const void *p) {
    os << "0x"
       << llvm::format_hex(reinterpret_cast<std::uintptr_t>(p),
                           /*Width=*/0, /*Upper=*/false);
  }
  void dumpHelper(llvm::raw_string_ostream &os, const std::string &prefix,
                  bool isLast, bool isRoot) const {
    if (hierarchy_name.empty()) {
      return; // skip nodes without hierarchy names to reduce noise
    }
    if (!isRoot) {
      os << prefix << (isLast ? "└─ " : "├─ ");
    } else if (!prefix.empty()) {
      os << prefix;
    }

    // Main line
    os << hierarchy_name
       << "  kind=" << (op_kind.empty() ? "<unknown>" : op_kind)
       << "  children=" << children.size() << "  op=";

    if (op)
      printPtr(os, op);
    else
      os << "<null>";

    os << "  parent=";
    if (parent)
      printPtr(os, parent);
    else
      os << "<null>";
    os << "\n";

    // Recurse
    std::string childPrefix = prefix;
    if (!isRoot)
      childPrefix += (isLast ? "   " : "│  ");

    for (size_t i = 0; i < children.size(); ++i) {
      const bool childLast = (i + 1 == children.size());
      children[i]->dumpHelper(os, childPrefix, childLast, /*isRoot=*/false);
    }
  }
};

struct ProxyTreeState {
  int loopCounter = 0;                 // current level of loop nesting
  int branchCounter = 0;               // current level of branch nesting
  SmallVector<StringRef, 4> ctrlStack; // stack of control flows
};

struct IdentifierCompletionState {
  int visited = 0;
  int assigned = 0;
  int rewritten = 0;
  bool overwrite = false;
  struct Scope {
    std::string parentPath;
    int loopCounter = 0;
    int branchCounter = 0;
    int opCounter = 0;
    int funcCounter = 0;
  };
  SmallVector<Scope, 8> scopeStack;
  std::set<std::string> used;
};
} // namespace

static std::string joinPath(const std::string &parent,
                            const std::string &leaf) {
  if (parent.empty())
    return leaf;
  return parent + "." + leaf;
}

static std::string uniquifyIdentifier(const std::string &base,
                                      std::set<std::string> &used) {
  std::string candidate = base;
  if (!used.count(candidate)) {
    used.insert(candidate);
    return candidate;
  }
  for (int suffix = 1;; ++suffix) {
    candidate = base + "." + std::to_string(suffix);
    if (!used.count(candidate)) {
      used.insert(candidate);
      return candidate;
    }
  }
}

static std::string getOpIdentifier(Operation *op) {
  if (!op)
    return "";
  if (auto idAttr = op->getAttrOfType<StringAttr>(allo::OpIdentifier))
    return idAttr.str();
  if (auto symAttr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return symAttr.str();
  return "";
}

static SmallVector<BlockArgument> collectBlockArguments(Operation *op) {
  SmallVector<BlockArgument> blockArgs;
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks())
      llvm::append_range(blockArgs, block.getArguments());
  }
  return blockArgs;
}

static std::string stringifyType(Type type) {
  std::string out;
  llvm::raw_string_ostream os(out);
  type.print(os);
  return out;
}

static std::string buildValueIdentifier(StringRef ownerIdentifier,
                                        StringRef sourceKind, int64_t index) {
  std::string owner =
      ownerIdentifier.empty() ? "__unknown_owner__" : ownerIdentifier.str();
  return owner + ":" + sourceKind.str() + std::to_string(index);
}

static ProxyValueInfo
buildProxyValueInfo(Value value, StringRef ownerIdentifier, StringRef ownerKind,
                    StringRef sourceKind, int64_t sourceIndex) {
  ProxyValueInfo info;
  info.value = value;
  info.owner_op_identifier = ownerIdentifier.str();
  info.owner_op_kind = ownerKind.str();
  info.source_kind = sourceKind.str();
  info.source_index = sourceIndex;
  info.value_identifier =
      buildValueIdentifier(ownerIdentifier, sourceKind, sourceIndex);
  info.type_str = stringifyType(value.getType());
  info.is_memref = isa<MemRefType>(value.getType());
  if (!info.is_memref)
    return info;

  Value root = allo::resolveMemRefValueRoot(value);
  if (auto arg = dyn_cast<BlockArgument>(root)) {
    info.root_kind = "block_arg";
    info.root_arg_number = static_cast<int64_t>(arg.getArgNumber());
    if (Block *owner = arg.getOwner())
      info.root_owner_identifier = getOpIdentifier(owner->getParentOp());
    return info;
  }

  Operation *defOp = root.getDefiningOp();
  if (!defOp)
    return info;

  if (isa<memref::AllocOp>(defOp))
    info.root_kind = "alloc";
  else if (isa<memref::AllocaOp>(defOp))
    info.root_kind = "alloca";
  else if (isa<memref::GetGlobalOp>(defOp))
    info.root_kind = "global";
  info.root_owner_identifier = getOpIdentifier(defOp);
  return info;
}

static void collectProxyValues(ProxyNode &node) {
  if (!node.op)
    return;

  StringRef ownerIdentifier = node.op_identifier;
  StringRef ownerKind = node.op_kind;
  for (auto [idx, result] : llvm::enumerate(node.op->getResults())) {
    node.values.push_back(buildProxyValueInfo(
        result, ownerIdentifier, ownerKind, "res", static_cast<int64_t>(idx)));
  }

  SmallVector<BlockArgument> blockArgs = collectBlockArguments(node.op);
  for (auto [idx, arg] : llvm::enumerate(blockArgs)) {
    node.values.push_back(buildProxyValueInfo(
        arg, ownerIdentifier, ownerKind, "arg", static_cast<int64_t>(idx)));
  }
}

static void completeIdentifiersRec(IdentifierCompletionState &state,
                                   Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>())
    return;

  state.visited++;

  bool isLoopLike = isa<LoopLikeOpInterface>(op);
  bool isIfLike = isa<affine::AffineIfOp, scf::IfOp>(op);
  bool isFuncLike = isa<FunctionOpInterface>(op);
  auto &scope = state.scopeStack.back();
  const std::string &parentPath = scope.parentPath;

  std::string generated;
  if (isLoopLike) {
    generated = joinPath(parentPath, "L" + std::to_string(scope.loopCounter++));
  } else if (isIfLike) {
    generated =
        joinPath(parentPath, "B" + std::to_string(scope.branchCounter++));
  } else if (isFuncLike) {
    auto symName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    std::string funcName = symName ? symName.str() : "";
    if (funcName.empty())
      funcName = "F" + std::to_string(scope.funcCounter++);
    generated = joinPath(parentPath, funcName);
  } else {
    generated = joinPath(parentPath, "O" + std::to_string(scope.opCounter++));
  }

  auto idAttr = op->getAttrOfType<StringAttr>(allo::OpIdentifier);
  std::string currentId = idAttr ? idAttr.str() : "";
  std::string targetId =
      (!state.overwrite && !currentId.empty()) ? currentId : generated;
  if (targetId.empty())
    targetId = generated;
  std::string uniqueId = uniquifyIdentifier(targetId, state.used);

  if (currentId.empty()) {
    op->setAttr(allo::OpIdentifier,
                StringAttr::get(op->getContext(), uniqueId));
    state.assigned++;
  } else if (currentId != uniqueId || state.overwrite) {
    op->setAttr(allo::OpIdentifier,
                StringAttr::get(op->getContext(), uniqueId));
    if (currentId != uniqueId)
      state.rewritten++;
  }

  if (isLoopLike || isIfLike || isFuncLike) {
    IdentifierCompletionState::Scope childScope;
    childScope.parentPath = uniqueId;
    state.scopeStack.push_back(std::move(childScope));
  }

  for (unsigned regionIdx = 0; regionIdx < op->getNumRegions(); ++regionIdx) {
    Region &region = op->getRegion(regionIdx);
    for (Block &block : region) {
      for (Operation &nestedOp : block)
        completeIdentifiersRec(state, &nestedOp);
    }
  }

  if (isLoopLike || isIfLike || isFuncLike)
    state.scopeStack.pop_back();
}

static void finalizeTransformRec(Operation *op, int &visited, int &removed,
                                 int &keptSymbol) {
  visited++;
  if (op->hasAttr(allo::OpIdentifier)) {
    // OpIdentifier intentionally reuses "sym_name". Keep it for symbol ops
    // because it is the real symbol name, not a temporary transform identifier.
    if (isa<SymbolOpInterface>(op) && !isa<ModuleOp>(op)) {
      keptSymbol++;
    } else {
      op->removeAttr(allo::OpIdentifier);
      removed++;
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block)
        finalizeTransformRec(&nestedOp, visited, removed, keptSymbol);
    }
  }
}

/// For loop/branch nodes without existing identifiers, fallback names use
/// sibling-local counters under the current parent scope.
/// Example loop nest: L0.L0.L0; sibling loops: L0, L1, L2.
static std::unique_ptr<ProxyNode>
buildProxyNode(ProxyTreeState &state, Operation *op, ProxyNode *parent) {
  auto node = std::make_unique<ProxyNode>();
  node->parent = parent;
  node->op = op;
  node->op_kind = op->getName().getStringRef().str();
  if (auto idAttr = op->getAttrOfType<StringAttr>(allo::OpIdentifier))
    node->op_identifier = idAttr.str();

  bool isLoopLike = isa<LoopLikeOpInterface>(op);
  bool isIfLike = isa<affine::AffineIfOp, scf::IfOp>(op);
  bool isFuncLike = isa<FunctionOpInterface>(op);
  if (!node->op_identifier.empty()) {
    node->hierarchy_name = node->op_identifier;
    if (isLoopLike || isIfLike || isFuncLike)
      state.ctrlStack.push_back(node->hierarchy_name);
  } else if (isLoopLike) {
    node->hierarchy_name =
        (state.ctrlStack.back() + ".L" + std::to_string(state.loopCounter++))
            .str();
    state.ctrlStack.push_back(node->hierarchy_name);
  } else if (isIfLike) {
    node->hierarchy_name =
        (state.ctrlStack.back() + ".B" + std::to_string(state.branchCounter++))
            .str();
    state.ctrlStack.push_back(node->hierarchy_name);
  } else if (isFuncLike) {
    node->hierarchy_name =
        (state.ctrlStack.back() +
         op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
             .strref())
            .str();
    state.ctrlStack.push_back(node->hierarchy_name);
  } else if (op->hasAttr(allo::OpIdentifier)) {
    auto idStr = op->getAttrOfType<StringAttr>(allo::OpIdentifier).str();
    node->hierarchy_name = (state.ctrlStack.back() + "." + idStr).str();
  } else {
    node->hierarchy_name = "";
  }
  collectProxyValues(*node);

  int loopCntGuard = state.loopCounter;
  int branchCntGuard = state.branchCounter;

  for (unsigned regionIdx = 0; regionIdx < op->getNumRegions(); ++regionIdx) {
    Region &region = op->getRegion(regionIdx);
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        node->children.push_back(buildProxyNode(state, &nestedOp, node.get()));
      }
    }
  }

  if (isLoopLike || isIfLike || isFuncLike) {
    state.ctrlStack.pop_back();
  }
  state.loopCounter = loopCntGuard;
  state.branchCounter = branchCntGuard;
  return node;
}

static std::unique_ptr<ProxyNode> buildProxyTree(ModuleOp top) {
  auto node = std::make_unique<ProxyNode>();
  node->op = top;
  node->op_kind = ModuleOp::getOperationName();
  node->hierarchy_name = "__allo_module__";
  if (auto idAttr = top->getAttrOfType<StringAttr>(allo::OpIdentifier))
    node->op_identifier = idAttr.str();
  else
    node->op_identifier = "__allo_module__";
  collectProxyValues(*node);

  ProxyTreeState state;
  state.ctrlStack.push_back("");
  for (auto &nestedOp : top.getOps()) {
    node->children.push_back(buildProxyNode(state, &nestedOp, node.get()));
  }
  return node;
}

static void init_context(nb::module_ &m) {
  nb::class_<MLIRContext>(m, "Context")
      .def("__init__",
           [](MLIRContext &self) {
             new (&self) MLIRContext(MLIRContext::Threading::DISABLED);
           })
      .def("printOpOnDiagnostic", &MLIRContext::printOpOnDiagnostic,
           nb::arg("enable"))
      .def("printStackTraceOnDiagnostic",
           &MLIRContext::printStackTraceOnDiagnostic, nb::arg("enable"))
      .def("load_dialects",
           [](MLIRContext &self) {
             DialectRegistry registry;
             allo::registerAllDialects(registry);
             self.appendDialectRegistry(registry);
             self.loadAllAvailableDialects();
           })
      .def("load_transform_dialects",
           [](MLIRContext &self) {
             DialectRegistry registry;
             registry.insert<transform::TransformDialect>();
             allo::registerAllExtensions(registry);
             self.appendDialectRegistry(registry);
             self.loadAllAvailableDialects();
           })
      .def_prop_ro("loaded_dialects", [](MLIRContext &self) {
        std::vector<std::string> dialects;
        for (auto *dialect : self.getLoadedDialects()) {
          dialects.push_back(dialect->getNamespace().str());
        }
        return dialects;
      });
}

static void init_builder(nb::module_ &m) {
  nb::class_<OpBuilder::InsertPoint>(m, "InsertPoint")
      .def(nb::init<>())
      .def_prop_ro(
          "block", [](OpBuilder::InsertPoint &self) { return self.getBlock(); },
          nb::rv_policy::reference);

  nb::class_<OpBuilder>(m, "OpBuilder")
      .def(nb::init<MLIRContext *>())
      .def(nb::init<Operation *>())
      .def(nb::init<Region *>())
      .def_prop_ro("context", &OpBuilder::getContext)
      // insertion point management
      .def(
          "set_insertion_point",
          [](OpBuilder &self, Operation *op) { self.setInsertionPoint(op); },
          nb::arg("op"))
      .def(
          "set_insertion_point_after",
          [](OpBuilder &self, Operation *op) {
            self.setInsertionPointAfter(op);
          },
          nb::arg("op"))
      .def(
          "set_insertion_point_to_start",
          [](OpBuilder &self, Block *block) {
            self.setInsertionPointToStart(block);
          },
          nb::arg("block"))
      .def(
          "set_insertion_point_to_end",
          [](OpBuilder &self, Block *block) {
            self.setInsertionPointToEnd(block);
          },
          nb::arg("block"))
      .def("save_insertion_point",
           [](OpBuilder &self) { return self.saveInsertionPoint(); })
      .def(
          "restore_insertion_point",
          [](OpBuilder &self, OpBuilder::InsertPoint &ip) {
            self.restoreInsertionPoint(ip);
          },
          nb::arg("ip"))
      // affine attributes
      .def(
          "get_affine_dim",
          [](OpBuilder &self, unsigned dim) {
            return self.getAffineDimExpr(dim);
          },
          nb::arg("dim"))
      .def(
          "get_affine_symbol",
          [](OpBuilder &self, unsigned sym) {
            return self.getAffineSymbolExpr(sym);
          },
          nb::arg("sym"))
      .def(
          "get_affine_constant",
          [](OpBuilder &self, int64_t value) {
            return self.getAffineConstantExpr(value);
          },
          nb::arg("value"))
      .def("get_unknown_loc", &OpBuilder::getUnknownLoc)
      .def(
          "create_block",
          [](OpBuilder &self) {
            return self.createBlock(self.getBlock()->getParent());
          },
          nb::rv_policy::reference)
      .def(
          "create_block_in_region",
          [](AlloOpBuilder &self, Region &region,
             const std::vector<Type> &argTypes) {
            Location loc = self.get_loc();
            llvm::SmallVector<Location, 4> locs(argTypes.size(), loc);
            return self.createBlock(&region, {}, argTypes, locs);
          },
          nb::arg("region"), nb::arg("arg_types"))
      .def(
          "get_dict_attr",
          [](OpBuilder &self, nb::dict &dict) {
            llvm::SmallVector<NamedAttribute, 4> attrs;
            for (const auto &[k, v] : dict) {
              auto key = nb::cast<std::string>(k);
              auto value = nb::cast<Attribute>(v);
              attrs.push_back(self.getNamedAttr(key, value));
            }
            return self.getDictionaryAttr(attrs);
          },
          nb::arg("dict"))
      .def("get_str_attr", [](OpBuilder &self, const std::string &value) {
        return self.getStringAttr(value);
      });

  nb::class_<AlloOpBuilder, OpBuilder>(m, "AlloOpBuilder")
      .def(nb::init<MLIRContext *>())
      .def(nb::init<Operation *>())
      .def_prop_rw("loc", &AlloOpBuilder::get_loc, &AlloOpBuilder::set_loc)
      .def("set_unknown_loc", &AlloOpBuilder::set_unknown_loc)
      .def("get_insertion_point_and_loc",
           &AlloOpBuilder::get_insertion_point_and_loc)
      .def("set_insertion_point_and_loc",
           &AlloOpBuilder::set_insertion_point_and_loc, nb::arg("ip"),
           nb::arg("new_loc"));
}

static void init_core_ir(nb::module_ &m) {
  nb::class_<Location>(m, "Location")
      // UnknownLoc init
      .def(
          "__init__",
          [](Location &self, MLIRContext &context) {
            self = Location(UnknownLoc::get(&context));
          },
          nb::arg("context"))
      // FileLineColLoc init
      .def(
          "__init__",
          [](Location &self, const std::string &filename, unsigned line,
             unsigned col, MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, filename);
            self = dyn_cast<Location>(FileLineColLoc::get(attr, line, col));
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context"))
      // NamedLoc init
      .def(
          "__init__",
          [](Location &self, Location &childLoc, const std::string &name,
             MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, name);
            self = dyn_cast<LocationAttr>(NameLoc::get(attr, childLoc));
          },
          nb::arg("child_loc"), nb::arg("name"), nb::arg("context"))
      .def_prop_ro("context", &Location::getContext)
      .def("__str__",
           [](Location &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "set_name",
          [](Location &self, std::string &name) {
            StringAttr attr = StringAttr::get(self.getContext(), name);
            self = dyn_cast<Location>(NameLoc::get(attr, self));
          },
          nb::arg("name"))
      .def_prop_ro("col",
                   [](Location &self) {
                     if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
                       return fileLineColLoc.getColumn();
                     }
                     throw nb::value_error("Location is not a FileLineColLoc");
                   })
      .def_prop_ro("line",
                   [](Location &self) {
                     if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
                       return fileLineColLoc.getLine();
                     }
                     throw nb::value_error("Location is not a FileLineColLoc");
                   })
      .def_prop_ro("filename",
                   [](Location &self) {
                     if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
                       return fileLineColLoc.getFilename().str();
                     }
                     throw nb::value_error("Location is not a FileLineColLoc");
                   })
      .def_prop_ro("name", [](Location &self) {
        if (auto nameLoc = dyn_cast<NameLoc>(self)) {
          return nameLoc.getName().str();
        }
        throw nb::value_error("Location is not a NameLoc");
      });

  nb::class_<Type>(m, "Type")
      .def("__init__",
           [](Type &self) {
             throw nb::type_error(
                 "Type cannot be directly instantiated, to get a Type, use a "
                 "specific Type's get() method");
           })
      .def("__eq__",
           [](Type &self, nb::object &other) {
             Type *otherTy = nb::cast<Type *>(other);
             return (otherTy != nullptr) && self == *otherTy;
           })
      .def("__ne__",
           [](Type &self, nb::object &other) {
             Type *otherTy = nb::cast<Type *>(other);
             return (otherTy == nullptr) || self != *otherTy;
           })
      .def("__str__", [](Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<Value>(m, "Value")
      .def("__str__",
           [](Value &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def(
          "set_attr",
          [](Value &self, const std::string &name, Attribute &attr) {
            if (auto defOp = self.getDefiningOp()) {
              defOp->setAttr(name, attr);
            } else {
              auto arg = cast<BlockArgument>(self);
              unsigned id = arg.getArgNumber();
              std::string argName = name + "_arg" + std::to_string(id);
              Block *owner = arg.getOwner();
              if (owner->isEntryBlock() &&
                  !isa<func::FuncOp>(owner->getParentOp())) {
                owner->getParentOp()->setAttr(name, attr);
              }
            }
          },
          nb::arg("name"), nb::arg("attr"))
      .def_prop_ro("context", &Value::getContext)
      .def_prop_rw(
          "loc", &Value::getLoc,
          [](Value &self, Location loc) { self.setLoc(loc); }, nb::arg("loc"))
      .def_prop_rw(
          "type",
          [](Value &self) {
            auto t = self.getType();
            return PyTypeRegistry::create(t);
          },
          [](Value &self, Type ty) { self.setType(ty); }, nb::arg("type"))
      .def(
          "replace_all_uses_with",
          [](Value &self, Value &val) { self.replaceAllUsesWith(val); },
          nb::arg("val"))
      .def("erase", [](Value &self) {
        if (auto defOp = self.getDefiningOp()) {
          defOp->erase();
        } else {
          auto arg = cast<BlockArgument>(self);
          Block *owner = arg.getOwner();
          owner->eraseArgument(arg.getArgNumber());
        }
      });

  nb::class_<Attribute>(m, "Attribute")
      .def("__str__",
           [](Attribute &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def_prop_ro("context", &Attribute::getContext);

  nb::class_<Region>(m, "Region")
      .def_prop_ro("context", &Region::getContext)
      .def_prop_ro("parent_region", &Region::getParentRegion,
                   nb::rv_policy::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def("empty", &Region::empty)
      .def(
          "front", [](Region &self) { return &self.front(); },
          nb::rv_policy::reference)
      .def(
          "back", [](Region &self) { return &self.back(); },
          nb::rv_policy::reference)
      .def(
          "push_back",
          [](Region &self, Block *block) { self.push_back(block); },
          nb::arg("block"))
      .def(
          "push_front",
          [](Region &self, Block *block) { self.push_front(block); },
          nb::arg("block"))
      .def(
          "emplace_block", [](Region &self) { return &self.emplaceBlock(); },
          nb::rv_policy::reference);

  nb::class_<Block>(m, "Block")
      .def("get_arg_at",
           [](Block &self, unsigned idx) {
             if (idx >= self.getNumArguments()) {
               throw nb::index_error("block argument index out of range");
             }
             return self.getArgument(idx);
           })
      .def(
          "add_arg",
          [](Block &self, Type type) {
            Location loc = UnknownLoc::get(type.getContext());
            return self.addArgument(type, loc);
          },
          nb::arg("type"))
      .def(
          "add_arg_at_loc",
          [](Block &self, Type type, Location loc) {
            return self.addArgument(type, loc);
          },
          nb::arg("type"), nb::arg("loc"))
      .def_prop_ro("num_args", &Block::getNumArguments)
      .def("move_before",
           [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore, nb::arg("block"))
      .def_prop_ro("parent_region", &Block::getParent, nb::rv_policy::reference)
      .def(
          "merge_before",
          [](Block &self, Block &dst) {
            // ref: RewriterBase::mergeBlocks()
            if (self.getNumArguments() != 0)
              throw std::runtime_error("This block has arguments, don't merge");
            dst.getOperations().splice(dst.begin(), self.getOperations());
            self.dropAllUses();
            self.erase();
          },
          nb::arg("dst"))
      .def(
          "replace_use_in_block_with",
          [](Block &self, Value &oldVal, Value &newVal) {
            oldVal.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              Block *currentBlock = user->getBlock();
              while (currentBlock) {
                if (currentBlock == &self)
                  return true;
                // Move up one level
                currentBlock =
                    currentBlock->getParent()->getParentOp()->getBlock();
              }
              return false;
            });
          },
          nb::arg("old_val"), nb::arg("new_val"))
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("has_terminator",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::IsTerminator>();
           })
      .def("remove_terminator",
           [](Block &self) {
             if (self.empty() ||
                 !self.back().hasTrait<OpTrait::IsTerminator>()) {
               return;
             }
             self.back().erase();
           })
      .def("has_return",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::ReturnLike>();
           })
      .def("erase", [](Block &self) { self.erase(); });

  // Base Operation class
  nb::class_<Operation>(m, "Operation")
      .def_prop_ro("context", &Operation::getContext)
      .def_prop_ro("loc", &Operation::getLoc)
      .def_prop_ro("name", [](Operation &self) {
        return self.getName().getStringRef().str();
      });

  nb::class_<OpState>(m, "OpState")
      .def_prop_ro("context", &OpState::getContext)
      .def_prop_ro("loc", &OpState::getLoc)
      .def("set_attr", [](OpState &self, std::string &name,
                          Attribute &attr) { self->setAttr(name, attr); })
      .def_prop_ro("num_operands",
                   [](OpState &self) { return self->getNumOperands(); })
      .def(
          "get_operand_at",
          [](OpState &self, unsigned idx) {
            if (idx >= self->getNumOperands()) {
              throw nb::index_error("Op operand index out of range");
            }
            return self->getOperand(idx);
          },
          nb::arg("idx"))
      .def_prop_ro("num_results",
                   [](OpState &self) { return self->getNumResults(); })
      .def("get_result_at",
           [](OpState &self, unsigned idx) {
             if (idx >= self->getNumResults())
               throw nb::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def_prop_ro("num_regions",
                   [](OpState &self) { return self->getNumRegions(); })
      .def(
          "get_region_at",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw nb::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          nb::rv_policy::reference)
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self->print(os, printingFlags);
             return os.str();
           })
      .def("append_operand",
           [](OpState &self, Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify",
           [](OpState &self) -> bool {
             return succeeded(verify(self.getOperation()));
           })
      .def_prop_ro(
          "operation", [](OpState &self) { return self.getOperation(); },
          nb::rv_policy::reference)
      .def_prop_ro(
          "block",
          [](OpState &self) { return self.getOperation()->getBlock(); },
          nb::rv_policy::reference);

  nb::class_<ModuleOp, OpState>(m, "ModuleOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder) {
            return ModuleOp::create(builder.get_loc());
          },
          nb::arg("builder"))
      .def_prop_ro(
          "body", [](ModuleOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("push_back",
           [](ModuleOp &self, Operation *op) { self.getBody()->push_back(op); })
      .def(
          "lookup_function",
          [](ModuleOp &self, const std::string &name) {
            return self.lookupSymbol<func::FuncOp>(name);
          },
          nb::arg("name"));

  nb::class_<ProxyNode>(m, "ProxyNode")
      .def_prop_ro("values",
                   [](ProxyNode &self) {
                     nb::list out;
                     for (auto &value : self.values)
                       out.append(value);
                     return out;
                   })
      .def_prop_ro(
          "parent", [](ProxyNode &self) { return self.parent; },
          nb::rv_policy::reference)
      .def_prop_ro(
          "op", [](ProxyNode &self) { return self.op; },
          nb::rv_policy::reference)
      .def_prop_ro("children",
                   [](ProxyNode &self) {
                     nb::list out;
                     for (auto &c : self.children)
                       out.append(c.get());
                     return out;
                   })
      .def_prop_ro("op_kind", [](ProxyNode &self) { return self.op_kind; })
      .def_prop_ro("hierarchy_name",
                   [](ProxyNode &self) { return self.hierarchy_name; })
      .def_prop_ro("op_identifier",
                   [](ProxyNode &self) { return self.op_identifier; })
      .def("__str__", [](ProxyNode &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.dump(os);
        return os.str();
      });

  nb::class_<ProxyValueInfo>(m, "ProxyValueInfo")
      .def_prop_ro("value", [](ProxyValueInfo &self) { return self.value; })
      .def_prop_ro("value_identifier",
                   [](ProxyValueInfo &self) { return self.value_identifier; })
      .def_prop_ro(
          "owner_op_identifier",
          [](ProxyValueInfo &self) { return self.owner_op_identifier; })
      .def_prop_ro("owner_op_kind",
                   [](ProxyValueInfo &self) { return self.owner_op_kind; })
      .def_prop_ro("source_kind",
                   [](ProxyValueInfo &self) { return self.source_kind; })
      .def_prop_ro("source_index",
                   [](ProxyValueInfo &self) { return self.source_index; })
      .def_prop_ro("type_str",
                   [](ProxyValueInfo &self) { return self.type_str; })
      .def_prop_ro("is_memref",
                   [](ProxyValueInfo &self) { return self.is_memref; })
      .def_prop_ro("root_kind",
                   [](ProxyValueInfo &self) { return self.root_kind; })
      .def_prop_ro(
          "root_owner_identifier",
          [](ProxyValueInfo &self) { return self.root_owner_identifier; })
      .def_prop_ro("root_arg_number",
                   [](ProxyValueInfo &self) { return self.root_arg_number; });

  m.attr("OP_IDENTIFIER_ATTR_NAME") = nb::str(allo::OpIdentifier.data());

  m.def("parse_proxy_tree",
        [](ModuleOp &module) { return buildProxyTree(module); });

  m.def(
      "complete_op_identifiers",
      [](ModuleOp &module, bool overwrite) {
        IdentifierCompletionState state;
        state.overwrite = overwrite;
        IdentifierCompletionState::Scope rootScope;
        rootScope.parentPath = "";
        state.scopeStack.push_back(std::move(rootScope));
        state.used.insert("__allo_module__");
        module->setAttr(
            allo::OpIdentifier,
            StringAttr::get(module->getContext(), "__allo_module__"));
        for (auto &nestedOp : module.getOps())
          completeIdentifiersRec(state, &nestedOp);
        nb::dict out;
        out["visited"] = state.visited;
        out["assigned"] = state.assigned;
        out["rewritten"] = state.rewritten;
        return out;
      },
      nb::arg("module"), nb::arg("overwrite") = false);

  m.def(
      "finalize_transform",
      [](ModuleOp &module) {
        int visited = 0;
        int removed = 0;
        int keptSymbol = 0;
        finalizeTransformRec(module.getOperation(), visited, removed,
                             keptSymbol);
        nb::dict out;
        out["visited"] = visited;
        out["removed"] = removed;
        out["kept_symbol"] = keptSymbol;
        return out;
      },
      nb::arg("module"));

  m.def(
      "parse_from_string",
      [](MLIRContext &ctx, const std::string &str) {
        ParserConfig config(&ctx, true, nullptr);
        auto mod = mlir::parseSourceString<ModuleOp>(str, config);
        if (!mod) {
          throw std::runtime_error(
              "Failed to parse MLIR string for proxy tree");
        }
        return mod.release();
      },
      nb::arg("context"), nb::arg("s"));

  m.def(
      "parse_from_file",
      [](MLIRContext &ctx, const std::string &filename) {
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
              "Failed to parse MLIR file for proxy tree: " + filename);
        }
        return mod.release();
      },
      nb::arg("context"), nb::arg("filename"));

  m.def("verify",
        [](Operation &op) -> bool { return succeeded(mlir::verify(&op)); });
}

static void init_types(nb::module_ &m) {
  nb::class_<FunctionType, Type>(m, "FunctionType")
      .def_static(
          "get",
          [](const std::vector<Type> &argTypes,
             const std::vector<Type> &retTypes, MLIRContext &context) {
            return FunctionType::get(&context, argTypes, retTypes);
          },
          nb::arg("arg_types"), nb::arg("ret_types"), nb::arg("context"))
      .def_prop_ro("arg_types",
                   [](FunctionType &self) {
                     std::vector<Type> argTypes;
                     for (Type ty : self.getInputs()) {
                       argTypes.push_back(ty);
                     }
                     return argTypes;
                   })
      .def_prop_ro("res_types",
                   [](FunctionType &self) {
                     std::vector<Type> retTypes;
                     for (Type ty : self.getResults()) {
                       retTypes.push_back(ty);
                     }
                     return retTypes;
                   })
      .def_prop_ro("num_args", &FunctionType::getNumInputs)
      .def_prop_ro("num_results", &FunctionType::getNumResults);
  PyTypeRegistry::registerType<FunctionType>();

  nb::class_<NoneType, Type>(m, "NoneType")
      .def_static(
          "get", [](MLIRContext &context) { return NoneType::get(&context); },
          nb::arg("context"));
  PyTypeRegistry::registerType<NoneType>();

  nb::class_<IntegerType, Type>(m, "IntegerType")
      .def_static(
          "get",
          [](unsigned width, MLIRContext &context) {
            return IntegerType::get(&context, width);
          },
          nb::arg("width"), nb::arg("context"))
      .def_static("get_width", [](Type &ty) {
        if (auto intType = dyn_cast<IntegerType>(ty)) {
          return intType.getWidth();
        }
        throw nb::type_error("Type is not an IntegerType");
      });
  PyTypeRegistry::registerType<IntegerType>();

  nb::class_<IndexType, Type>(m, "IndexType")
      .def_static(
          "get", [](MLIRContext &context) { return IndexType::get(&context); },
          nb::arg("context"))
      .def_static(
          "isa", [](Type &ty) { return isa<IndexType>(ty); }, nb::arg("type"));
  PyTypeRegistry::registerType<IndexType>();

  // Float Types
  (void)nb::class_<FloatType, Type>(m, "FloatType");
  PyTypeRegistry::registerType<FloatType>();

  nb::class_<Float16Type, Type>(m, "F16Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float16Type::get(&context); },
          nb::arg("context"));
  PyTypeRegistry::registerType<Float16Type>();

  nb::class_<Float32Type, Type>(m, "F32Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float32Type::get(&context); },
          nb::arg("context"));
  PyTypeRegistry::registerType<Float32Type>();

  nb::class_<Float64Type, Type>(m, "F64Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float64Type::get(&context); },
          nb::arg("context"));
  PyTypeRegistry::registerType<Float64Type>();

  nb::class_<BFloat16Type, Type>(m, "BF16Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return BFloat16Type::get(&context); },
          nb::arg("context"));
  PyTypeRegistry::registerType<BFloat16Type>();

  // RankedTensorType
  nb::class_<RankedTensorType, Type>(m, "RankedTensorType")
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType,
             std::optional<Attribute> encoding) {
            return RankedTensorType::get(shape, elementType,
                                         encoding.value_or(Attribute()));
          },
          nb::arg("shape"), nb::arg("element_type"),
          nb::arg("encoding").none() = nb::none())
      .def_prop_ro("encoding", &RankedTensorType::getEncoding)
      .def_prop_ro("element_type",
                   [](RankedTensorType &self) { return self.getElementType(); })
      .def_prop_ro("shape",
                   [](RankedTensorType &self) {
                     auto shape = self.getShape();
                     std::vector<int64_t> ret;
                     for (auto dim : shape) {
                       ret.push_back(dim);
                     }
                     return ret;
                   })
      .def_prop_ro("rank", &RankedTensorType::getRank)
      .def(
          "get_dim_size_at",
          [](RankedTensorType &self, unsigned index) {
            if (index >= self.getRank()) {
              throw nb::index_error("tensor type dimension index out of range");
            }
            return self.getDimSize(index);
          },
          nb::arg("index"))
      .def(
          "set_element_type",
          [](RankedTensorType &self, Type newElementType) {
            return RankedTensorType::get(self.getShape(), newElementType,
                                         self.getEncoding());
          },
          nb::arg("type"));
  PyTypeRegistry::registerType<RankedTensorType>();

  nb::class_<MemRefType, Type>(m, "MemRefType")
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType, AffineMap map,
             std::optional<Attribute> memorySpace) {
            return MemRefType::get(shape, elementType, map,
                                   memorySpace.value_or(IntegerAttr()));
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("affine_maps"),
          nb::arg("memory_space").none() = nb::none())
      .def_prop_ro("element_type",
                   [](MemRefType &self) { return self.getElementType(); })
      .def_prop_ro("shape",
                   [](MemRefType &self) {
                     auto shape = self.getShape();
                     std::vector<int64_t> ret;
                     for (auto dim : shape) {
                       ret.push_back(dim);
                     }
                     return ret;
                   })
      .def_prop_ro("rank", &MemRefType::getRank);
  PyTypeRegistry::registerType<MemRefType>();
}

static void init_values(nb::module_ &m) {
  nb::class_<BlockArgument, Value>(m, "BlockArgument")
      .def_prop_ro("arg_number", &BlockArgument::getArgNumber)
      .def_prop_ro("owner", &BlockArgument::getOwner, nb::rv_policy::reference);

  nb::class_<OpResult, Value>(m, "OpResult")
      .def_prop_ro("owner", &OpResult::getOwner, nb::rv_policy::reference)
      .def_prop_ro("res_no", &OpResult::getResultNumber);
}

static void init_attributes(nb::module_ &m) {
  nb::class_<IntegerAttr, Attribute>(m, "IntegerAttr")
      .def_static(
          "get",
          [](Type ty, int64_t value) { return IntegerAttr::get(ty, value); },
          nb::arg("ty"), nb::arg("value"))
      .def("get_signless", &IntegerAttr::getInt)
      .def("get_signed", &IntegerAttr::getSInt)
      .def("get_unsigned", &IntegerAttr::getUInt);
  PyAttributeRegistry::registerAttr<IntegerAttr>();

  nb::class_<FloatAttr, Attribute>(m, "FloatAttr")
      .def_static(
          "get",
          [](Type ty, double value) { return FloatAttr::get(ty, value); },
          nb::arg("ty"), nb::arg("value"));

  PyAttributeRegistry::registerAttr<FloatAttr>();

  nb::class_<UnitAttr, Attribute>(m, "UnitAttr")
      .def_static(
          "get", [](MLIRContext &context) { return UnitAttr::get(&context); },
          nb::arg("context"));
  PyAttributeRegistry::registerAttr<UnitAttr>();

  nb::class_<StringAttr, Attribute>(m, "StringAttr")
      .def_static(
          "get",
          [](const std::string &value, MLIRContext &context) {
            return StringAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"))
      .def_prop_ro("value",
                   [](StringAttr &self) { return self.getValue().str(); });
  PyAttributeRegistry::registerAttr<StringAttr>();

  nb::class_<BoolAttr, Attribute>(m, "BoolAttr")
      .def_static(
          "get",
          [](bool value, MLIRContext &context) {
            return BoolAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"));
  PyAttributeRegistry::registerAttr<BoolAttr>();

  nb::class_<DenseI32ArrayAttr, Attribute>(m, "DenseI32ArrayAttr")
      .def_static(
          "get",
          [](MLIRContext &context,
             const std::vector<int32_t> &values) -> DenseI32ArrayAttr {
            return DenseI32ArrayAttr::get(&context, values);
          },
          nb::arg("context"), nb::arg("values"))
      .def("size", [](DenseI32ArrayAttr &self) { return self.getSize(); });
  PyAttributeRegistry::registerAttr<DenseI32ArrayAttr>();

  nb::class_<DenseI64ArrayAttr, Attribute>(m, "DenseI64ArrayAttr")
      .def_static(
          "get",
          [](MLIRContext &context,
             const std::vector<int64_t> &values) -> DenseI64ArrayAttr {
            return DenseI64ArrayAttr::get(&context, values);
          },
          nb::arg("context"), nb::arg("values"))
      .def("size", [](DenseI64ArrayAttr &self) { return self.getSize(); });
  PyAttributeRegistry::registerAttr<DenseI64ArrayAttr>();

  nb::class_<DictionaryAttr, Attribute>(m, "DictionaryAttr")
      .def_static(
          "get",
          [](MLIRContext &context, nb::dict &dict) {
            llvm::SmallVector<NamedAttribute, 4> attrs;
            for (const auto &[k, v] : dict) {
              std::string key = nb::cast<std::string>(k);
              Attribute value = nb::cast<Attribute>(v);
              attrs.push_back(
                  NamedAttribute(StringAttr::get(&context, key), value));
            }
            return DictionaryAttr::get(&context, attrs);
          },
          nb::arg("context"), nb::arg("d"));
  PyAttributeRegistry::registerAttr<DictionaryAttr>();
}

static void init_affine_objects(nb::module_ &m) {

  nb::class_<IntegerSet>(m, "IntegerSet")
      .def_static(
          "get",
          [](unsigned numDims, unsigned numSymbols,
             const std::vector<AffineExpr> &constraints,
             const std::vector<int> &eqFlags) {
            // convert eqFlags to SmallVector
            llvm::SmallVector<bool, 4> eqFlagsSmall;
            for (int flag : eqFlags) {
              eqFlagsSmall.push_back(flag != 0);
            }
            return IntegerSet::get(numDims, numSymbols, constraints,
                                   eqFlagsSmall);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"), nb::arg("constraints"),
          nb::arg("context"))
      .def_prop_ro("num_dims", &IntegerSet::getNumDims)
      .def_prop_ro("num_symbols", &IntegerSet::getNumSymbols)
      .def_prop_ro("num_constraints", &IntegerSet::getNumConstraints)
      .def("__str__", [](IntegerSet &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<AffineExpr>(m, "AffineExpr")
      .def("__str__",
           [](AffineExpr &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      // operator overloading
      .def("__add__",
           [](AffineExpr &self, AffineExpr &other) { return self + other; })
      .def("__sub__",
           [](AffineExpr &self, AffineExpr &other) { return self - other; })
      .def("__mul__",
           [](AffineExpr &self, AffineExpr &other) { return self * other; })
      .def("__floordiv__",
           [](AffineExpr &self, AffineExpr &other) {
             return self.floorDiv(other);
           })
      .def("__truediv__", [](AffineExpr &self,
                             AffineExpr &other) { return self.ceilDiv(other); })
      .def("__mod__",
           [](AffineExpr &self, AffineExpr &other) { return self % other; });

  nb::class_<AffineMap>(m, "AffineMap")
      .def_static(
          "get",
          [](const std::vector<unsigned> &dimSizes,
             const std::vector<unsigned> &symbolSizes,
             const std::vector<AffineExpr> &results, MLIRContext &context) {
            return AffineMap::get(dimSizes.size(), symbolSizes.size(), results,
                                  &context);
          },
          nb::arg("dim_sizes"), nb::arg("symbol_sizes"), nb::arg("results"),
          nb::arg("context"))
      .def_static(
          "get_identity",
          [](unsigned dimCount, MLIRContext &context) {
            return AffineMap::getMultiDimIdentityMap(dimCount, &context);
          },
          nb::arg("dim_count"), nb::arg("context"))
      .def_prop_ro("num_dims", &AffineMap::getNumDims)
      .def_prop_ro("num_symbols", &AffineMap::getNumSymbols)
      .def_prop_ro("num_results", &AffineMap::getNumResults)
      .def("__str__", [](AffineMap &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<AffineMapAttr, Attribute>(m, "AffineMapAttr")
      .def_static(
          "get",
          [](AffineMap map, MLIRContext &context) {
            return AffineMapAttr::get(map);
          },
          nb::arg("map"), nb::arg("context"))
      .def_prop_ro("map", &AffineMapAttr::getValue);

  PyAttributeRegistry::registerAttr<AffineMapAttr>();

  nb::class_<affine::AffineBound>(m, "AffineBound")
      .def_prop_ro("map", &affine::AffineBound::getMap)
      .def_prop_ro("operands", [](affine::AffineBound &self) {
        std::vector<Value> vals;
        for (Value val : self.getOperands()) {
          vals.push_back(val);
        }
        return vals;
      });
}

void init_ir(nb::module_ &m) {
  init_context(m);
  init_builder(m);
  init_core_ir(m);
  init_types(m);
  init_values(m);
  init_attributes(m);
  init_affine_objects(m);
}
