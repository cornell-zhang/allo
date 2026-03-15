#include "ir.h"

using namespace mlir;

namespace {

static bool isInstanceOf(nb::handle obj, nb::handle cls) {
  int ret = PyObject_IsInstance(obj.ptr(), cls.ptr());
  if (ret < 0)
    throw nb::python_error();
  return ret == 1;
}

static nb::object dynCastOperation(Operation *op, nb::handle cls) {
  if (op == nullptr)
    return nb::none();
  nb::object wrapped = PyOpRegistry::create(op);
  if (isInstanceOf(wrapped, cls))
    return wrapped;
  return nb::none();
}

static nb::object castOperation(Operation *op, nb::handle cls) {
  nb::object wrapped = dynCastOperation(op, cls);
  if (!wrapped.is_none())
    return wrapped;
  throw nb::type_error("Operation cannot be cast to the requested wrapper "
                       "type");
}

} // namespace

static void bindContext(nb::module_ &m) {
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
      .def("get_loaded_dialects", [](MLIRContext &self) {
        std::vector<std::string> dialects;
        for (auto *dialect : self.getLoadedDialects()) {
          dialects.push_back(dialect->getNamespace().str());
        }
        return dialects;
      });
}

static void bindBuilder(nb::module_ &m) {
  nb::class_<OpBuilder::InsertPoint>(m, "InsertPoint")
      .def(nb::init<>())
      .def(
          "get_block",
          [](OpBuilder::InsertPoint &self) { return self.getBlock(); },
          nb::rv_policy::reference);

  nb::class_<OpBuilder>(m, "OpBuilder")
      .def(nb::init<MLIRContext *>(), nb::arg("context"))
      .def(nb::init<Operation *>(), nb::arg("operation"))
      .def(nb::init<Region *>(), nb::arg("region"))
      .def("get_context", &OpBuilder::getContext)
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
          "create_free_block",
          [](OpBuilder &self, Region &region,
             const std::vector<Type> &argTypes = {}) {
            return self.createBlock(self.getBlock()->getParent());
          },
          nb::rv_policy::reference, nb::arg("region"),
          nb::arg("arg_types") = std::vector<Type>())
      .def(
          "create_block_in_region",
          [](OpBuilder &self, Location loc, Region &region,
             const std::vector<Type> &argTypes) {
            llvm::SmallVector<Location, 4> locs(argTypes.size(), loc);
            return self.createBlock(&region, {}, argTypes, locs);
          },
          nb::rv_policy::reference, nb::arg("loc"), nb::arg("region"),
          nb::arg("arg_types"))
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
      .def(
          "get_string_attr",
          [](OpBuilder &self, std::string_view value) {
            return self.getStringAttr(value);
          },
          nb::arg("value"));

  nb::class_<AlloOpBuilder, OpBuilder>(m, "AlloOpBuilder")
      .def(nb::init<MLIRContext *>(), nb::arg("context"))
      .def(nb::init<Operation *>(), nb::arg("operation"))
      .def("get_loc", &AlloOpBuilder::getLocation)
      .def("set_loc", &AlloOpBuilder::setLocation, nb::arg("new_loc"))
      .def("set_unknown_loc", &AlloOpBuilder::setUnknownLoc)
      .def("get_insertion_point_and_loc",
           &AlloOpBuilder::getInsertionPointAndLoc)
      .def("set_insertion_point_and_loc",
           &AlloOpBuilder::setInsertionPointAndLoc, nb::arg("ip"),
           nb::arg("new_loc"));
}

static void bindCoreIR(nb::module_ &m) {
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
          [](Location &self, std::string_view filename, unsigned line,
             unsigned col, MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, filename);
            self = dyn_cast<Location>(FileLineColLoc::get(attr, line, col));
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context"))
      // NamedLoc init
      .def(
          "__init__",
          [](Location &self, Location &childLoc, std::string_view name,
             MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, name);
            self = dyn_cast<LocationAttr>(NameLoc::get(attr, childLoc));
          },
          nb::arg("child_loc"), nb::arg("name"), nb::arg("context"))
      .def("get_context", &Location::getContext)
      .def("__str__",
           [](Location &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "set_name",
          [](Location &self, std::string_view name) {
            StringAttr attr = StringAttr::get(self.getContext(), name);
            self = dyn_cast<Location>(NameLoc::get(attr, self));
          },
          nb::arg("name"))
      .def("get_col",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getColumn();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_line",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getLine();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_filename",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getFilename().str();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_name", [](Location &self) {
        if (auto nameLoc = dyn_cast<NameLoc>(self)) {
          return nameLoc.getName().str();
        }
        throw nb::value_error("Location is not a NameLoc");
      });

  nb::class_<Type>(m, "Type")
      .def_static("cast",
                  [](Type &other) { return PyTypeRegistry::create(other); })
      .def("__init__",
           [](Type &self) {
             throw nb::type_error(
                 "Type cannot be directly instantiated, to get a Type, use a "
                 "specific Type's get() method");
           })
      .def(
          "__eq__",
          [](Type &self, nb::object &other) {
            Type *otherTy = nb::cast<Type *>(other);
            return (otherTy != nullptr) && self == *otherTy;
          },
          nb::arg("other"))
      .def(
          "__ne__",
          [](Type &self, nb::object &other) {
            Type *otherTy = nb::cast<Type *>(other);
            return (otherTy == nullptr) || self != *otherTy;
          },
          nb::arg("other"))
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
          [](Value &self, std::string_view name, Attribute &attr) {
            if (Operation *defOp = self.getDefiningOp()) {
              defOp->setAttr(name, attr);
            } else {
              auto arg = cast<BlockArgument>(self);
              Block *owner = arg.getOwner();
              if (owner->isEntryBlock() &&
                  !isa<func::FuncOp>(owner->getParentOp())) {
                owner->getParentOp()->setAttr(name, attr);
              }
            }
          },
          nb::arg("name"), nb::arg("attr"))
      .def("get_context", &Value::getContext)
      .def("get_loc", &Value::getLoc)
      .def(
          "set_loc", [](Value &self, Location loc) { self.setLoc(loc); },
          nb::arg("loc"))
      .def("get_type",
           [](Value &self) {
             auto t = self.getType();
             return PyTypeRegistry::create(t);
           })
      .def(
          "set_type", [](Value &self, Type ty) { self.setType(ty); },
          nb::arg("type"))
      .def(
          "replace_all_uses_with",
          [](Value &self, Value &val) { self.replaceAllUsesWith(val); },
          nb::arg("val"));

  nb::class_<Attribute>(m, "Attribute")
      .def_static(
          "cast",
          [](Attribute &other) { return PyAttributeRegistry::create(other); })
      .def("__str__",
           [](Attribute &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("get_context", &Attribute::getContext);

  nb::class_<Region>(m, "Region")
      .def("get_context", &Region::getContext)
      .def("get_parent_region", &Region::getParentRegion,
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
      .def(
          "get_arg_at",
          [](Block &self, unsigned idx) {
            if (idx >= self.getNumArguments()) {
              throw nb::index_error("block argument index out of range");
            }
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def("get_args",
           [](Block &self) {
             std::vector<Value> args;
             for (auto arg : self.getArguments())
               args.push_back(arg);
             return args;
           })
      .def("get_arg_types",
           [](Block &self) {
             std::vector<Type> argTypes;
             for (auto arg : self.getArguments())
               argTypes.push_back(arg.getType());
             return argTypes;
           })
      .def("get_num_args", &Block::getNumArguments)
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
      .def("get_parent_region", &Block::getParent, nb::rv_policy::reference)
      .def("get_parent_op", &Block::getParentOp, nb::rv_policy::reference)
      .def("insert_before", &Block::insertBefore, nb::arg("block"))
      .def(
          "move_before", [](Block &self, Block &dst) { self.moveBefore(&dst); },
          nb::arg("dst"))
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "get_terminator", [](Block &self) { return self.getTerminator(); },
          nb::rv_policy::reference)
      .def("erase", [](Block &self) { self.erase(); });

  // Base Operation class
  nb::class_<Operation>(m, "Operation")
      .def("get_context", &Operation::getContext)
      .def("get_loc", &Operation::getLoc)
      .def("get_name",
           [](Operation &self) { return self.getName().getStringRef().str(); });

  nb::class_<OpState>(m, "OpState")
      .def("__init__",
           [](OpState &self) {
             throw nb::type_error("OpState cannot be directly instantiated, "
                                  "use an Op's getState() "
                                  "method to get an OpState");
           })
      .def("get_context", &OpState::getContext)
      .def("get_loc", &OpState::getLoc)
      .def(
          "set_attr",
          [](OpState &self, std::string_view name, Attribute &attr) {
            self->setAttr(name, attr);
          },
          nb::arg("name"), nb::arg("attr"))
      .def(
          "get_attribute",
          [](OpState &self, std::string_view attrName) {
            return PyAttributeRegistry::create(self->getAttr(attrName));
          },
          nb::arg("attr_name"))
      .def("get_num_operands",
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
      .def("get_num_results",
           [](OpState &self) { return self->getNumResults(); })
      .def(
          "get_result_at",
          [](OpState &self, unsigned idx) {
            if (idx >= self->getNumResults())
              throw nb::index_error("Op result index out of range");
            return self->getResult(idx);
          },
          nb::arg("idx"))
      .def("get_num_regions",
           [](OpState &self) { return self->getNumRegions(); })
      .def(
          "get_region_at",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw nb::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          nb::rv_policy::reference, nb::arg("idx"))
      .def(
          "get_block",
          [](OpState &self) { return self.getOperation()->getBlock(); },
          nb::rv_policy::reference)
      .def(
          "get_operation", [](OpState &self) { return self.getOperation(); },
          nb::rv_policy::reference)
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self->print(os, printingFlags);
             return os.str();
           })
      .def("dump_with_loc",
           [](OpState &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags(true);
             self.print(os, printingFlags);
             return os.str();
           })
      .def("verify", [](OpState &self) -> bool {
        return succeeded(verify(self.getOperation()));
      });

  auto moduleOp = bindOp<ModuleOp>(m, "ModuleOp");
  bindConstructor(moduleOp,
                  [](AlloOpBuilder &builder) {
                    return ModuleOp::create(builder.getLocation());
                  })
      .def(
          "get_body", [](ModuleOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def(
          "push_back",
          [](ModuleOp &self, Operation *op) { self.getBody()->push_back(op); },
          nb::arg("op"));
}

static void bindTypes(nb::module_ &m) {
  nb::class_<FunctionType, Type>(m, "FunctionType")
      .def_static(
          "get",
          [](const std::vector<Type> &argTypes,
             const std::vector<Type> &retTypes, MLIRContext &context) {
            return FunctionType::get(&context, argTypes, retTypes);
          },
          nb::arg("arg_types"), nb::arg("ret_types"), nb::arg("context"))
      .def("get_arg_types",
           [](FunctionType &self) {
             std::vector<Type> argTypes;
             for (Type ty : self.getInputs()) {
               argTypes.push_back(ty);
             }
             return argTypes;
           })
      .def("get_res_types",
           [](FunctionType &self) {
             std::vector<Type> retTypes;
             for (Type ty : self.getResults()) {
               retTypes.push_back(ty);
             }
             return retTypes;
           })
      .def("get_num_args", &FunctionType::getNumInputs)
      .def("get_num_results", &FunctionType::getNumResults);
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
      .def_static(
          "get_width",
          [](Type &ty) {
            if (auto intType = dyn_cast<IntegerType>(ty)) {
              return intType.getWidth();
            }
            throw nb::type_error("Type is not an IntegerType");
          },
          nb::arg("ty"));
  PyTypeRegistry::registerType<IntegerType>();

  nb::class_<IndexType, Type>(m, "IndexType")
      .def_static(
          "get", [](MLIRContext &context) { return IndexType::get(&context); },
          nb::arg("context"));
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

  nb::class_<UnrankedTensorType, Type>(m, "UnrankedTensorType")
      .def_static(
          "get",
          [](Type elementType) { return UnrankedTensorType::get(elementType); },
          nb::arg("element_type"))
      .def("get_element_type",
           [](UnrankedTensorType &self) { return self.getElementType(); });
  PyTypeRegistry::registerType<UnrankedTensorType>();

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
      .def("get_encoding", &RankedTensorType::getEncoding)
      .def("get_element_type",
           [](RankedTensorType &self) { return self.getElementType(); })
      .def("get_shape",
           [](RankedTensorType &self) {
             auto shape = self.getShape();
             std::vector<int64_t> ret;
             for (auto dim : shape) {
               ret.push_back(dim);
             }
             return ret;
           })
      .def("get_rank", &RankedTensorType::getRank)
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

  nb::class_<UnrankedMemRefType, Type>(m, "UnrankedMemRefType")
      .def_static(
          "get",
          [](Type elementType, std::optional<Attribute> memorySpace) {
            return UnrankedMemRefType::get(elementType,
                                           memorySpace.value_or(Attribute{}));
          },
          nb::arg("element_type"), nb::arg("memory_space").none() = nb::none())
      .def("get_element_type",
           [](UnrankedMemRefType &self) { return self.getElementType(); });
  PyTypeRegistry::registerType<UnrankedMemRefType>();

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
      .def("get_element_type",
           [](MemRefType &self) { return self.getElementType(); })
      .def("get_shape",
           [](MemRefType &self) {
             auto shape = self.getShape();
             std::vector<int64_t> ret;
             for (auto dim : shape) {
               ret.push_back(dim);
             }
             return ret;
           })
      .def("get_rank", &MemRefType::getRank);
  PyTypeRegistry::registerType<MemRefType>();
}

static void bindValues(nb::module_ &m) {
  nb::class_<BlockArgument, Value>(m, "BlockArgument")
      .def("get_arg_number", &BlockArgument::getArgNumber)
      .def("get_owner", &BlockArgument::getOwner, nb::rv_policy::reference);

  nb::class_<OpResult, Value>(m, "OpResult")
      .def("get_owner", &OpResult::getOwner, nb::rv_policy::reference)
      .def("get_res_no", &OpResult::getResultNumber);
}

static void bindAttributes(nb::module_ &m) {
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
          [](std::string_view value, MLIRContext &context) {
            return StringAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"))
      .def("get_value", [](StringAttr &self) { return self.getValue().str(); });
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

  nb::class_<FlatSymbolRefAttr, Attribute>(m, "FlatSymbolRefAttr")
      .def_static(
          "get",
          [](std::string_view value, MLIRContext &context) {
            return FlatSymbolRefAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"))
      .def("get_value",
           [](FlatSymbolRefAttr &self) { return self.getValue().str(); });
  PyAttributeRegistry::registerAttr<FlatSymbolRefAttr>();

  nb::class_<StridedLayoutAttr, Attribute>(m, "StridedLayoutAttr")
      .def_static(
          "get",
          [](MLIRContext &context, int64_t offset,
             const std::vector<int64_t> &strides) {
            return StridedLayoutAttr::get(&context, offset, strides);
          },
          nb::arg("context"), nb::arg("offset"), nb::arg("strides"))
      .def("get_strides",
           [](StridedLayoutAttr &self) {
             std::vector<int64_t> strides;
             for (auto stride : self.getStrides()) {
               strides.push_back(stride);
             }
             return strides;
           })
      .def("get_offset", &StridedLayoutAttr::getOffset);
  PyAttributeRegistry::registerAttr<StridedLayoutAttr>();

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

  nb::class_<TypeAttr, Attribute>(m, "TypeAttr")
      .def_static(
          "get", [](Type ty) { return TypeAttr::get(ty); }, nb::arg("type"))
      .def("get_type", &TypeAttr::getValue);
  PyAttributeRegistry::registerAttr<TypeAttr>();
}

static void bindAffineObjects(nb::module_ &m) {
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
      .def("get_num_dims", &IntegerSet::getNumDims)
      .def("get_num_symbols", &IntegerSet::getNumSymbols)
      .def("get_num_constraints", &IntegerSet::getNumConstraints)
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
      .def(
          "__add__",
          [](AffineExpr &self, AffineExpr &other) { return self + other; },
          nb::arg("other"))
      .def(
          "__sub__",
          [](AffineExpr &self, AffineExpr &other) { return self - other; },
          nb::arg("other"))
      .def(
          "__mul__",
          [](AffineExpr &self, AffineExpr &other) { return self * other; },
          nb::arg("other"))
      .def(
          "__floordiv__",
          [](AffineExpr &self, AffineExpr &other) {
            return self.floorDiv(other);
          },
          nb::arg("other"))
      .def(
          "__truediv__",
          [](AffineExpr &self, AffineExpr &other) {
            return self.ceilDiv(other);
          },
          nb::arg("other"))
      .def(
          "__mod__",
          [](AffineExpr &self, AffineExpr &other) { return self % other; },
          nb::arg("other"));

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
      .def("get_num_dims", &AffineMap::getNumDims)
      .def("get_num_symbols", &AffineMap::getNumSymbols)
      .def("get_num_results", &AffineMap::getNumResults)
      .def("get_sub_map", &AffineMap::getSubMap)
      .def("__str__",
           [](AffineMap &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("simplify", [](AffineMap &self) { self = simplifyAffineMap(self); });

  nb::class_<affine::AffineBound>(m, "AffineBound")
      .def("get_map", &affine::AffineBound::getMap)
      .def("get_operands", [](affine::AffineBound &self) {
        std::vector<Value> vals;
        for (Value val : self.getOperands()) {
          vals.push_back(val);
        }
        return vals;
      });

  // some utils
  m.def("fully_compose_affine_map",
        [](AffineMap &map, const std::vector<Value> &operands) {
          AffineMap composedMap = map;
          SmallVector<Value, 4> ops(operands.begin(), operands.end());
          affine::fullyComposeAffineMapAndOperands(&composedMap, &ops, true);
          affine::canonicalizeMapAndOperands(&composedMap, &ops);
          return std::make_pair(composedMap, ops);
        });
}

void bindIR(nb::module_ &m) {
  m.def(
      "isa",
      [](Operation *op, nb::handle cls) {
        return !dynCastOperation(op, cls).is_none();
      },
      nb::arg("op"), nb::arg("cls"));
  m.def(
      "isa",
      [](OpState &op, nb::handle cls) {
        return !dynCastOperation(op.getOperation(), cls).is_none();
      },
      nb::arg("op"), nb::arg("cls"));
  m.def(
      "cast",
      [](Operation *op, nb::handle cls) { return castOperation(op, cls); },
      nb::arg("op"), nb::arg("cls"));
  m.def(
      "cast",
      [](OpState &op, nb::handle cls) {
        return castOperation(op.getOperation(), cls);
      },
      nb::arg("op"), nb::arg("cls"));
  m.def(
      "dyn_cast",
      [](Operation *op, nb::handle cls) { return dynCastOperation(op, cls); },
      nb::arg("op"), nb::arg("cls"));
  m.def(
      "dyn_cast",
      [](OpState &op, nb::handle cls) {
        return dynCastOperation(op.getOperation(), cls);
      },
      nb::arg("op"), nb::arg("cls"));

  bindContext(m);
  bindBuilder(m);
  bindCoreIR(m);
  bindTypes(m);
  bindValues(m);
  bindAttributes(m);
  bindAffineObjects(m);
}
