#include "ir.h"

using namespace mlir;

void bindFuncOps(nb::module_ &m) {
  bindOp<func::FuncOp>(m, "FuncOp")
      .def(
          "__init__",
          [](func::FuncOp &self, AlloOpBuilder &builder, std::string_view name,
             FunctionType &type) {
            self = func::FuncOp::create(builder, builder.getLocation(), name,
                                        type);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("type"))
      .def(
          "get_arg_at",
          [](func::FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def("get_num_args", &func::FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); },
          nb::rv_policy::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, unsigned argNo, std::string_view name,
             Attribute &attr) {
            if (argNo >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to Value &"val"
            self.setArgAttr(argNo, name, attr);
          },
          nb::arg("arg_no"), nb::arg("name"), nb::arg("attr"))
      .def("get_func_type", &func::FuncOp::getFunctionType)
      .def("set_type", &func::FuncOp::setType, nb::arg("type"))
      .def("get_func_name",
           [](func::FuncOp &self) { return self.getName().str(); });

  auto returnOp = bindOp<func::ReturnOp>(m, "ReturnOp");
  bindConstructor(
      returnOp,
      [](AlloOpBuilder &builder, const std::vector<Value> &operands) {
        return func::ReturnOp::create(builder, builder.getLocation(), operands);
      },
      nb::arg("operands"));

  auto callOp = bindOp<func::CallOp>(m, "CallOp");
  bindConstructor(
      callOp,
      [](AlloOpBuilder &builder, func::FuncOp &func,
         const std::vector<Value> &args) {
        return func::CallOp::create(builder, builder.getLocation(), func, args);
      },
      nb::arg("func"), nb::arg("args"));
}

void bindAffineOps(nb::module_ &m) {
  // affine ops
  auto affineFor = bindOp<affine::AffineForOp>(m, "AffineForOp");
  bindConstructor(
      affineFor,
      [](AlloOpBuilder &builder, const std::vector<Value> &lb, AffineMap lbMap,
         const std::vector<Value> &ub, AffineMap ubMap, int64_t step) {
        return affine::AffineForOp::create(builder, builder.getLocation(), lb,
                                           lbMap, ub, ubMap, step);
      },
      nb::arg("lb_operands"), nb::arg("lb_map"), nb::arg("ub_operands"),
      nb::arg("ub_map"), nb::arg("step") = 1)
      .def(
          "__init__",
          [](affine::AffineForOp &self, AlloOpBuilder &builder, int64_t lb,
             int64_t ub, int64_t step = 1) {
            self = affine::AffineForOp::create(builder, builder.getLocation(),
                                               lb, ub, step);
          },
          nb::arg("builder"), nb::arg("lb"), nb::arg("ub"), nb::arg("step") = 1)
      .def("get_induction_var", &affine::AffineForOp::getInductionVar)
      .def(
          "get_body", [](affine::AffineForOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("get_upper_bound", &affine::AffineForOp::getUpperBound)
      .def("get_lower_bound", &affine::AffineForOp::getLowerBound)
      .def("get_constant_upper_bound",
           &affine::AffineForOp::getConstantUpperBound)
      .def("get_constant_lower_bound",
           &affine::AffineForOp::getConstantLowerBound)
      .def("get_has_constant_lower_bound",
           &affine::AffineForOp::hasConstantLowerBound)
      .def("get_has_constant_upper_bound",
           &affine::AffineForOp::hasConstantUpperBound)
      .def("get_step", &affine::AffineForOp::getStepAsInt)
      .def("has_constant_bounds", &affine::AffineForOp::hasConstantBounds);

  auto affineIf = bindOp<affine::AffineIfOp>(m, "AffineIfOp");
  bindConstructor(
      affineIf,
      [](AlloOpBuilder &builder, IntegerSet set,
         const std::vector<Value> &operands, bool withElse) {
        return affine::AffineIfOp::create(builder, builder.getLocation(), set,
                                          operands, withElse);
      },
      nb::arg("set"), nb::arg("operands"), nb::arg("with_else") = false)
      .def("get_integer_set", &affine::AffineIfOp::getIntegerSet)
      .def("get_then_block", &affine::AffineIfOp::getThenBlock,
           nb::rv_policy::reference)
      .def("get_else_block", &affine::AffineIfOp::getElseBlock,
           nb::rv_policy::reference);

  auto affineLoad = bindOp<affine::AffineLoadOp>(m, "AffineLoadOp");
  bindConstructor(
      affineLoad,
      [](AlloOpBuilder &builder, Value &memref, AffineMap &map,
         const std::vector<Value> &operands) {
        return affine::AffineLoadOp::create(builder, builder.getLocation(),
                                            memref, map, operands);
      },
      nb::arg("memref"), nb::arg("map"), nb::arg("operands"));
  affineLoad.def("get_value", &affine::AffineLoadOp::getValue)
      .def("get_memref", &affine::AffineLoadOp::getMemRef)
      .def("get_map", &affine::AffineLoadOp::getAffineMap)
      .def("get_operands", [](affine::AffineLoadOp &self) {
        auto operands = self.getMapOperands();
        return std::vector<Value>(operands.begin(), operands.end());
      });

  auto affineStore = bindOp<affine::AffineStoreOp>(m, "AffineStoreOp");
  bindConstructor(
      affineStore,
      [](AlloOpBuilder &builder, Value &value, Value &memref, AffineMap &map,
         const std::vector<Value> &operands) {
        return affine::AffineStoreOp::create(builder, builder.getLocation(),
                                             value, memref, map, operands);
      },
      nb::arg("value"), nb::arg("memref"), nb::arg("map"), nb::arg("operands"));
  affineStore
      .def("get_value",
           [](affine::AffineStoreOp &self) { return self.getValueToStore(); })
      .def("get_memref", &affine::AffineStoreOp::getMemRef)
      .def("get_map", &affine::AffineStoreOp::getAffineMap)
      .def("get_operands", [](affine::AffineStoreOp &self) {
        auto operands = self.getMapOperands();
        return std::vector<Value>(operands.begin(), operands.end());
      });

  auto affineApply = bindOp<affine::AffineApplyOp>(m, "AffineApplyOp");
  bindConstructor(
      affineApply,
      [](AlloOpBuilder &builder, AffineMap &map,
         const std::vector<Value> &operands) {
        return affine::AffineApplyOp::create(builder, builder.getLocation(),
                                             map, operands);
      },
      nb::arg("map"), nb::arg("operands"));
  affineApply.def("get_map", &affine::AffineApplyOp::getAffineMap)
      .def("get_operands", [](affine::AffineApplyOp &self) {
        auto operands = self.getMapOperands();
        return std::vector<Value>(operands.begin(), operands.end());
      });
}

void bindSCFOps(nb::module_ &m) {
  // scf ops
  auto forOp = bindOp<scf::ForOp>(m, "ForOp");
  bindConstructor(
      forOp,
      [](AlloOpBuilder &builder, Value &lb, Value &ub, Value &step,
         const std::vector<Value> &initArgs) {
        return scf::ForOp::create(builder, builder.getLocation(), lb, ub, step,
                                  initArgs);
      },
      nb::arg("lb"), nb::arg("ub"), nb::arg("step"),
      nb::arg("init_args") = std::vector<Value>())
      .def("get_induction_var", &scf::ForOp::getInductionVar)
      .def(
          "get_body", [](scf::ForOp &self) { return self.getBody(0); },
          nb::rv_policy::reference);

  auto ifOp = bindOp<scf::IfOp>(m, "IfOp");
  bindConstructor(
      ifOp,
      [](AlloOpBuilder &builder, const std::vector<Type> &resultTypes,
         Value &cond, bool withElse) {
        return scf::IfOp::create(builder, builder.getLocation(), resultTypes,
                                 cond, withElse);
      },
      nb::arg("res_types"), nb::arg("cond"), nb::arg("with_else") = false)
      .def("get_then_block", &scf::IfOp::thenBlock, nb::rv_policy::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, nb::rv_policy::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);

  auto yieldOp = bindOp<scf::YieldOp>(m, "YieldOp");
  bindConstructor(
      yieldOp,
      [](AlloOpBuilder &builder, const std::vector<Value> &results) {
        return scf::YieldOp::create(builder, builder.getLocation(), results);
      },
      nb::arg("results"));

  auto whileOp = bindOp<scf::WhileOp>(m, "WhileOp");
  bindConstructor(
      whileOp,
      [](AlloOpBuilder &builder, const std::vector<Type> &resultTypes,
         const std::vector<Value> &operands) {
        return scf::WhileOp::create(builder, builder.getLocation(), resultTypes,
                                    operands);
      },
      nb::arg("result_types"), nb::arg("operands"))
      .def("get_before", &scf::WhileOp::getBefore, nb::rv_policy::reference)
      .def("get_after", &scf::WhileOp::getAfter, nb::rv_policy::reference);

  auto conditionOp = bindOp<scf::ConditionOp>(m, "ConditionOp");
  bindConstructor(
      conditionOp,
      [](AlloOpBuilder &builder, Value &cond, const std::vector<Value> &args) {
        return scf::ConditionOp::create(builder, builder.getLocation(), cond,
                                        args);
      },
      nb::arg("cond"), nb::arg("args"));

  auto parallelOp = bindOp<scf::ParallelOp>(m, "ParallelOp");
  bindConstructor(
      parallelOp,
      [](AlloOpBuilder &builder, const std::vector<Value> &lbs,
         const std::vector<Value> &ubs, const std::vector<Value> &steps,
         const std::vector<Value> &initArgs) {
        return scf::ParallelOp::create(builder, builder.getLocation(), lbs, ubs,
                                       steps, initArgs);
      },
      nb::arg("lbs"), nb::arg("ubs"), nb::arg("steps"),
      nb::arg("init_args") = std::vector<Value>())
      .def(
          "get_body", [](scf::ParallelOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("get_induction_vars", [](scf::ParallelOp &self) {
        auto ivs = self.getInductionVars();
        return std::vector<Value>(ivs.begin(), ivs.end());
      });
}

void bindCFOps(nb::module_ &m) {
  auto condBranchOp = bindOp<cf::CondBranchOp>(m, "CondBranchOp");
  bindConstructor(
      condBranchOp,
      [](AlloOpBuilder &builder, Value &cond, Block *trueDst, Block *falseDst) {
        return cf::CondBranchOp::create(builder, builder.getLocation(), cond,
                                        trueDst, falseDst);
      },
      nb::arg("cond"), nb::arg("true_dest"), nb::arg("false_dest"));

  auto branchOp = bindOp<cf::BranchOp>(m, "BranchOp");
  bindConstructor(
      branchOp,
      [](AlloOpBuilder &builder, Block *dest, const std::vector<Value> &args) {
        return cf::BranchOp::create(builder, builder.getLocation(), dest, args);
      },
      nb::arg("dest"), nb::arg("args"));
}

void bindArithOps(nb::module_ &m) {
  // constant ops
  (void)bindOp<arith::ConstantOp>(m, "ConstantOp");

  auto constantInt =
      bindOp<arith::ConstantIntOp, arith::ConstantOp>(m, "ConstantIntOp");
  bindConstructor(
      constantInt,
      [](AlloOpBuilder &builder, IntegerType &type, int64_t value) {
        return arith::ConstantIntOp::create(builder, builder.getLocation(),
                                            type, value);
      },
      nb::arg("type"), nb::arg("value"));

  auto constantFloat =
      bindOp<arith::ConstantFloatOp, arith::ConstantOp>(m, "ConstantFloatOp");
  bindConstructor(
      constantFloat,
      [](AlloOpBuilder &builder, Float32Type &type, float value) {
        return arith::ConstantFloatOp::create(builder, builder.getLocation(),
                                              type, APFloat(value));
      },
      nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             Float64Type &type, double value) {
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             Float16Type &type, float value) {
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             BFloat16Type &type, float value) {
            // bf16 does not satisfy IEEE754, so we need to convert manually
            const llvm::fltSemantics &sem = type.getFloatSemantics();
            llvm::APFloat val(value);
            bool lost;
            val.convert(sem, llvm::APFloat::rmNearestTiesToEven, &lost);
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, val);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"));

  auto constantIndex =
      bindOp<arith::ConstantIndexOp, arith::ConstantOp>(m, "ConstantIndexOp");
  bindConstructor(
      constantIndex,
      [](AlloOpBuilder &builder, int64_t value) {
        return arith::ConstantIndexOp::create(builder, builder.getLocation(),
                                              value);
      },
      nb::arg("value"));

  // casts / conversions
  (void)bindSourceToTypeOp<arith::SIToFPOp>(m, "SIToFPOp");
  (void)bindSourceToTypeOp<arith::UIToFPOp>(m, "UIToFPOp");
  (void)bindSourceToTypeOp<arith::FPToSIOp>(m, "FPToSIOp");
  (void)bindSourceToTypeOp<arith::FPToUIOp>(m, "FPToUIOp");
  (void)bindSourceToTypeOp<arith::ExtFOp>(m, "ExtFOp");
  (void)bindSourceToTypeOp<arith::TruncFOp>(m, "TruncFOp");
  (void)bindTypeToSourceOp<arith::IndexCastOp>(m, "IndexCastOp");

  // integer extension / truncation / bitcast
  (void)bindSourceToTypeOp<arith::ExtSIOp>(m, "ExtSIOp");
  (void)bindSourceToTypeOp<arith::ExtUIOp>(m, "ExtUIOp");
  (void)bindSourceToTypeOp<arith::BitcastOp>(m, "BitcastOp");
  (void)bindSourceToTypeOp<arith::TruncIOp>(m, "TruncIOp");

  // floating ops
  (void)bindBinaryValueOp<arith::AddFOp>(m, "AddFOp");
  (void)bindBinaryValueOp<arith::SubFOp>(m, "SubFOp");
  (void)bindBinaryValueOp<arith::MulFOp>(m, "MulFOp");
  (void)bindBinaryValueOp<arith::DivFOp>(m, "DivFOp");
  (void)bindBinaryValueOp<arith::RemFOp>(m, "RemFOp");
  (void)bindUnaryValueOp<arith::NegFOp>(m, "NegFOp", "input");

  // integer arithmetic
  (void)bindBinaryValueOp<arith::AddIOp>(m, "AddIOp");
  (void)bindBinaryValueOp<arith::SubIOp>(m, "SubIOp");
  (void)bindBinaryValueOp<arith::MulIOp>(m, "MulIOp");
  (void)bindBinaryValueOp<arith::DivSIOp>(m, "DivSIOp");
  (void)bindBinaryValueOp<arith::DivUIOp>(m, "DivUIOp");
  (void)bindBinaryValueOp<arith::CeilDivSIOp>(m, "CeilDivSIOp");
  (void)bindBinaryValueOp<arith::CeilDivUIOp>(m, "CeilDivUIOp");
  (void)bindBinaryValueOp<arith::FloorDivSIOp>(m, "FloorDivSIOp");
  (void)bindBinaryValueOp<arith::RemSIOp>(m, "RemSIOp");
  (void)bindBinaryValueOp<arith::RemUIOp>(m, "RemUIOp");

  // fused / special ops
  auto fmaOp = bindOp<math::FmaOp>(m, "FmaOp");
  bindConstructor(
      fmaOp,
      [](AlloOpBuilder &builder, Value &a, Value &b, Value &c) {
        return math::FmaOp::create(builder, builder.getLocation(), a, b, c);
      },
      nb::arg("a"), nb::arg("b"), nb::arg("c"));

  // shifts
  (void)bindBinaryValueOp<arith::ShLIOp>(m, "ShLIOp");
  (void)bindBinaryValueOp<arith::ShRUIOp>(m, "ShRUIOp");
  (void)bindBinaryValueOp<arith::ShRSIOp>(m, "ShRSIOp");

  // mins / maxs
  (void)bindBinaryValueOp<arith::MinSIOp>(m, "MinSIOp");
  (void)bindBinaryValueOp<arith::MinUIOp>(m, "MinUIOp");
  (void)bindBinaryValueOp<arith::MinimumFOp>(m, "MinimumFOp");
  (void)bindBinaryValueOp<arith::MinNumFOp>(m, "MinNumFOp");
  (void)bindBinaryValueOp<arith::MaxSIOp>(m, "MaxSIOp");
  (void)bindBinaryValueOp<arith::MaxUIOp>(m, "MaxUIOp");
  (void)bindBinaryValueOp<arith::MaximumFOp>(m, "MaximumFOp");
  (void)bindBinaryValueOp<arith::MaxNumFOp>(m, "MaxNumFOp");

  // comparisons (int)
  auto cmpIOp = bindOp<arith::CmpIOp>(m, "CmpIOp");
  bindConstructor(
      cmpIOp,
      [](AlloOpBuilder &builder, std::size_t pred, Value &lhs, Value &rhs) {
        return arith::CmpIOp::create(builder, builder.getLocation(),
                                     static_cast<arith::CmpIPredicate>(pred),
                                     lhs, rhs);
      },
      nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (float)
  auto cmpFOp = bindOp<arith::CmpFOp>(m, "CmpFOp");
  bindConstructor(
      cmpFOp,
      [](AlloOpBuilder &builder, std::size_t pred, Value &lhs, Value &rhs) {
        return arith::CmpFOp::create(builder, builder.getLocation(),
                                     static_cast<arith::CmpFPredicate>(pred),
                                     lhs, rhs);
      },
      nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // logical
  (void)bindBinaryValueOp<arith::AndIOp>(m, "AndIOp");
  (void)bindBinaryValueOp<arith::XOrIOp>(m, "XOrIOp");
  (void)bindBinaryValueOp<arith::OrIOp>(m, "OrIOp");

  auto selectOp = bindOp<arith::SelectOp>(m, "SelectOp");
  bindConstructor(
      selectOp,
      [](AlloOpBuilder &builder, Value &condition, Value &trueValue,
         Value &falseValue) {
        return arith::SelectOp::create(builder, builder.getLocation(),
                                       condition, trueValue, falseValue);
      },
      nb::arg("condition"), nb::arg("true_value"), nb::arg("false_value"));
}

void bindMathOps(nb::module_ &m) {
  (void)bindUnaryValueOp<math::FloorOp>(m, "FloorOp");
  (void)bindUnaryValueOp<math::CeilOp>(m, "CeilOp");
  (void)bindUnaryValueOp<math::ExpOp>(m, "ExpOp");
  (void)bindUnaryValueOp<math::Exp2Op>(m, "Exp2Op");
  (void)bindUnaryValueOp<math::CosOp>(m, "CosOp");
  (void)bindUnaryValueOp<math::SinOp>(m, "SinOp");
  (void)bindUnaryValueOp<math::LogOp>(m, "LogOp");
  (void)bindUnaryValueOp<math::Log2Op>(m, "Log2Op");
  (void)bindUnaryValueOp<math::ErfOp>(m, "ErfOp");
  (void)bindUnaryValueOp<math::SqrtOp>(m, "SqrtOp");
  (void)bindUnaryValueOp<math::RsqrtOp>(m, "RsqrtOp");
  (void)bindUnaryValueOp<math::AbsFOp>(m, "AbsFOp");
  (void)bindUnaryValueOp<math::AbsIOp>(m, "AbsIOp");
  (void)bindBinaryValueOp<math::PowFOp>(m, "PowFOp", "base", "exponent");
  (void)bindUnaryValueOp<math::TanOp>(m, "TanOp");
}

void bindTensorOps(nb::module_ &m) {
  (void)bindValueRangeOp<tensor::ExtractOp>(m, "ExtractOp", "tensor",
                                            "indices");
  (void)bindTwoValueRangeOp<tensor::InsertOp>(m, "InsertOp", "value", "tensor",
                                              "indices");

  auto splatOp = bindOp<tensor::SplatOp>(m, "SplatOp");
  bindConstructor(
      splatOp,
      [](AlloOpBuilder &builder, Value &value,
         const std::vector<int64_t> &shape) {
        return tensor::SplatOp::create(builder, builder.getLocation(), value,
                                       shape);
      },
      nb::arg("value"), nb::arg("shape"));

  (void)bindSourceToTypeOp<tensor::CastOp>(m, "CastOp", "input", "dst_type");

  auto emptyOp = bindOp<tensor::EmptyOp>(m, "EmptyOp");
  bindConstructor(
      emptyOp,
      [](AlloOpBuilder &builder, const std::vector<int64_t> &shape,
         Type &elementType) {
        return tensor::EmptyOp::create(builder, builder.getLocation(), shape,
                                       elementType);
      },
      nb::arg("shape"), nb::arg("element_type"))
      .def(
          "__init__",
          [](tensor::EmptyOp &self, AlloOpBuilder &builder, Type &type) {
            if (auto tensor = dyn_cast<RankedTensorType>(type)) {
              self = tensor::EmptyOp::create(
                  builder, builder.getLocation(), tensor.getShape(),
                  tensor.getElementType(), tensor.getEncoding());
              return;
            }
            if (auto memref = dyn_cast<MemRefType>(type)) {
              self = tensor::EmptyOp::create(
                  builder, builder.getLocation(), memref.getShape(),
                  memref.getElementType(), memref.getMemorySpace());
              return;
            }
            throw nb::type_error("Unsupported type for tensor.EmptyOp");
          },
          nb::arg("builder"), nb::arg("type"));

  auto extractSlice = bindOp<tensor::ExtractSliceOp>(m, "ExtractSliceOp");
  bindConstructor(
      extractSlice,
      [](AlloOpBuilder &builder, Value &source,
         const std::vector<Value> &offsets, const std::vector<Value> &sizes,
         const std::vector<Value> &strides) {
        return tensor::ExtractSliceOp::create(builder, builder.getLocation(),
                                              source, offsets, sizes, strides);
      },
      nb::arg("source"), nb::arg("offsets"), nb::arg("sizes"),
      nb::arg("strides"))
      .def(
          "__init__",
          [](tensor::ExtractSliceOp &self, AlloOpBuilder &builder,
             Type &resType, Value &source, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes, const std::vector<Value> &strides,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = tensor::ExtractSliceOp::create(
                builder, builder.getLocation(), resType, source, offsets, sizes,
                strides, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("source"),
          nb::arg("offsets"), nb::arg("sizes"), nb::arg("strides"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  auto insertSlice = bindOp<tensor::InsertSliceOp>(m, "InsertSliceOp");
  bindConstructor(
      insertSlice,
      [](AlloOpBuilder &builder, Value &source, Value &dest,
         const std::vector<Value> &offsets, const std::vector<Value> &sizes,
         const std::vector<Value> &strides) {
        return tensor::InsertSliceOp::create(builder, builder.getLocation(),
                                             source, dest, offsets, sizes,
                                             strides);
      },
      nb::arg("source"), nb::arg("dest"), nb::arg("offsets"), nb::arg("sizes"),
      nb::arg("strides"))
      .def(
          "__init__",
          [](tensor::InsertSliceOp &self, AlloOpBuilder &builder, Value &source,
             Value &dest, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes, const std::vector<Value> &strides,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = tensor::InsertSliceOp::create(
                builder, builder.getLocation(), source, dest, offsets, sizes,
                strides, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("dest"),
          nb::arg("offsets"), nb::arg("sizes"), nb::arg("strides"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  auto gatherOp = bindOp<tensor::GatherOp>(m, "GatherOp");
  bindConstructor(
      gatherOp,
      [](AlloOpBuilder &builder, Type &resType, Value &source, Value &indices,
         const std::vector<int64_t> &dims, bool unique) {
        return tensor::GatherOp::create(builder, builder.getLocation(), resType,
                                        source, indices, dims, unique);
      },
      nb::arg("res_type"), nb::arg("source"), nb::arg("indices"),
      nb::arg("dims"), nb::arg("unique") = false);

  auto scatterOp = bindOp<tensor::ScatterOp>(m, "ScatterOp");
  bindConstructor(
      scatterOp,
      [](AlloOpBuilder &builder, Type &resType, Value &source, Value &dest,
         Value &indices, const std::vector<int64_t> &dims, bool unique) {
        return tensor::ScatterOp::create(builder, builder.getLocation(),
                                         resType, source, dest, indices, dims,
                                         unique);
      },
      nb::arg("res_type"), nb::arg("source"), nb::arg("dest"),
      nb::arg("indices"), nb::arg("dims"), nb::arg("unique") = false);
}

void bindMemRefOps(nb::module_ &m) {
  (void)bindValueRangeOp<memref::LoadOp>(m, "LoadOp", "memref", "indices");
  (void)bindTwoValueRangeOp<memref::StoreOp>(m, "StoreOp", "value", "memref",
                                             "indices");

  auto allocOp = bindOp<memref::AllocOp>(m, "AllocOp");
  bindConstructor(
      allocOp,
      [](AlloOpBuilder &builder, MemRefType &type) {
        return memref::AllocOp::create(builder, builder.getLocation(), type);
      },
      nb::arg("type"));

  auto subViewOp = bindOp<memref::SubViewOp>(m, "SubViewOp");
  bindConstructor(
      subViewOp,
      [](AlloOpBuilder &builder, Value &source,
         const std::vector<Value> &offsets, const std::vector<Value> &sizes,
         const std::vector<Value> &strides) {
        return memref::SubViewOp::create(builder, builder.getLocation(), source,
                                         offsets, sizes, strides);
      },
      nb::arg("source"), nb::arg("offsets"), nb::arg("sizes"),
      nb::arg("strides"))
      .def(
          "__init__",
          [](memref::SubViewOp &self, AlloOpBuilder &builder, Value &source,
             const std::vector<int64_t> &offsets,
             const std::vector<int64_t> &sizes,
             const std::vector<int64_t> &strides) {
            self = memref::SubViewOp::create(builder, builder.getLocation(),
                                             source, offsets, sizes, strides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("offsets"),
          nb::arg("sizes"), nb::arg("strides"))
      .def(
          "__init__",
          [](memref::SubViewOp &self, AlloOpBuilder &builder, Type &type,
             Value &source, Value &offset, Value &size, Value &stride,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = memref::SubViewOp::create(
                builder, builder.getLocation(), type, source, offset, size,
                stride, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("source"),
          nb::arg("offset"), nb::arg("size"), nb::arg("stride"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  (void)bindBinaryValueOp<memref::CopyOp>(m, "CopyOp", "source", "dest");

  auto globalOp = bindOp<memref::GlobalOp>(m, "GlobalOp");
  bindConstructor(
      globalOp,
      [](AlloOpBuilder &builder, std::string_view name,
         std::string_view visibility, MemRefType &type, Attribute &initValue,
         bool isConstant, int64_t alignment) {
        auto visAttr = builder.getStringAttr(visibility);
        auto alignAttr =
            builder.getIntegerAttr(builder.getI64Type(), alignment);
        return memref::GlobalOp::create(builder, builder.getLocation(), name,
                                        visAttr, type, initValue, isConstant,
                                        alignAttr);
      },
      nb::arg("name"), nb::arg("visibility"), nb::arg("res_type"),
      nb::arg("init_value"), nb::arg("is_constant"), nb::arg("alignment"));

  auto getGlobalOp = bindOp<memref::GetGlobalOp>(m, "GetGlobalOp");
  bindConstructor(
      getGlobalOp,
      [](AlloOpBuilder &builder, Type &resType, std::string_view name) {
        return memref::GetGlobalOp::create(builder, builder.getLocation(),
                                           resType, name);
      },
      nb::arg("res_type"), nb::arg("name"));

  auto transposeOp = bindOp<memref::TransposeOp>(m, "TransposeOp");
  bindConstructor(
      transposeOp,
      [](AlloOpBuilder &builder, Value &input, AffineMap &permutation) {
        auto permAttr = AffineMapAttr::get(permutation);
        return memref::TransposeOp::create(builder, builder.getLocation(),
                                           input, permAttr);
      },
      nb::arg("input"), nb::arg("permutation"));

  auto reshapeOp = bindOp<memref::ReshapeOp>(m, "ReshapeOp");
  bindConstructor(
      reshapeOp,
      [](AlloOpBuilder &builder, Type &resType, Value &input, Value &shape) {
        return memref::ReshapeOp::create(builder, builder.getLocation(),
                                         resType, input, shape);
      },
      nb::arg("res_type"), nb::arg("input"), nb::arg("shape"));
}

void bindLinalgOps(nb::module_ &m) {
  (void)bindBinaryInputsInitOp<linalg::MatmulOp>(m, "MatmulOp", "lhs", "rhs",
                                                 "result");
  (void)bindUnaryInitOp<linalg::FillOp>(m, "FillOp", "value", "output");

  auto broadcastOp = bindOp<linalg::BroadcastOp>(m, "BroadcastOp");
  bindConstructor(
      broadcastOp,
      [](AlloOpBuilder &builder, Value &input, Value &init,
         const std::vector<int64_t> &dims) {
        return linalg::BroadcastOp::create(builder, builder.getLocation(),
                                           input, init, dims);
      },
      nb::arg("input"), nb::arg("init"), nb::arg("dims"));

  (void)bindBinaryInputsInitOp<linalg::AddOp>(m, "AddOp");
  (void)bindBinaryInputsInitOp<linalg::SubOp>(m, "SubOp");
  (void)bindBinaryInputsInitOp<linalg::MulOp>(m, "MulOp");
  (void)bindBinaryInputsInitOp<linalg::DivOp>(m, "DivOp");
  (void)bindBinaryInputsInitOp<linalg::DivUnsignedOp>(m, "DivUnsignedOp");
  (void)bindBinaryInputsInitOp<linalg::PowFOp>(m, "PowFOp", "base", "exponent",
                                               "init");
  (void)bindUnaryInitOp<linalg::FloorOp>(m, "FloorOp");
  (void)bindUnaryInitOp<linalg::ExpOp>(m, "ExpOp");
  (void)bindUnaryInitOp<linalg::LogOp>(m, "LogOp");
  (void)bindUnaryInitOp<linalg::SqrtOp>(m, "SqrtOp");
  (void)bindUnaryInitOp<linalg::ReciprocalOp>(m, "ReciprocalOp");
  (void)bindUnaryInitOp<linalg::RsqrtOp>(m, "RsqrtOp");
  (void)bindUnaryInitOp<linalg::SquareOp>(m, "SquareOp");
  (void)bindBinaryInputsInitOp<linalg::DotOp>(m, "DotOp");

  nb::enum_<utils::IteratorType>(m, "IteratorType")
      .value("PAR", utils::IteratorType::parallel)
      .value("RED", utils::IteratorType::reduction)
      .export_values();

  auto genericOp = bindOp<linalg::GenericOp>(m, "GenericOp");
  bindConstructor(
      genericOp,
      [](AlloOpBuilder &builder, const std::vector<Type> &resTypes,
         const std::vector<Value> &inputs, const std::vector<Value> &outputs,
         const std::vector<AffineMap> &indexingMaps,
         const std::vector<utils::IteratorType> &iteratorTypes) {
        return linalg::GenericOp::create(builder, builder.getLocation(),
                                         resTypes, inputs, outputs,
                                         indexingMaps, iteratorTypes);
      },
      nb::arg("result_types"), nb::arg("inputs"), nb::arg("outputs"),
      nb::arg("indexing_maps"), nb::arg("iterator_types"))
      .def("get_body", [](linalg::GenericOp &self) { return self.getBody(); })
      .def(
          "add_entry_block",
          [](linalg::GenericOp &self) {
            SmallVector<Type, 4> blockArgTypes;
            SmallVector<Location, 4> blockArgLocs;
            for (ValueRange container : {self.getInputs(), self.getOutputs()}) {
              for (Value v : container) {
                Type t = v.getType();
                blockArgTypes.push_back(isa<MemRefType, RankedTensorType>(t)
                                            ? getElementTypeOrSelf(t)
                                            : t);
                blockArgLocs.push_back(v.getLoc());
              }
            }
            Block *block = &self->getRegion(0).emplaceBlock();
            block->addArguments(blockArgTypes, blockArgLocs);
            return block;
          },
          nb::rv_policy::reference);

  auto yieldOp = bindOp<linalg::YieldOp>(m, "YieldOp");
  bindConstructor(
      yieldOp,
      [](AlloOpBuilder &builder, const std::vector<Value> &values) {
        return linalg::YieldOp::create(builder, builder.getLocation(), values);
      },
      nb::arg("values"));
}

void bindUBOps(nb::module_ &m) { (void)m; }
