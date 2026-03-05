#include "ir.h"

using namespace mlir;

void init_ub_ops(nb::module_ &m) {
  nb::class_<ub::PoisonOp, OpState>(m, "PoisonOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Type &type) -> Value {
            return ub::PoisonOp::create(builder, builder.get_loc(), type);
          },
          nb::arg("builder"), nb::arg("type"));
}

void init_func_ops(nb::module_ &m) {
  nb::class_<func::FuncOp, OpState>(m, "FuncOp")
      .def_static("create",
                  [](AlloOpBuilder &builder, const std::string &name,
                     FunctionType &type) {
                    return func::FuncOp::create(builder, builder.get_loc(),
                                                name, type);
                  })
      .def(
          "get_arg_at",
          [](func::FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def_prop_ro("num_args", &func::FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); },
          nb::rv_policy::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, unsigned arg_no, const std::string &name,
             Attribute &attr) {
            if (arg_no >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to Value &"val"
            self.setArgAttr(arg_no, name, attr);
          },
          nb::arg("arg_no"), nb::arg("name"), nb::arg("attr"))
      .def_prop_ro("func_type", &func::FuncOp::getFunctionType)
      .def("set_type", &func::FuncOp::setType, nb::arg("type"))
      .def_prop_ro("func_name",
                   [](func::FuncOp &self) { return self.getName().str(); });

  nb::class_<func::ReturnOp, OpState>(m, "ReturnOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &operands) {
            return func::ReturnOp::create(builder, builder.get_loc(), operands);
          },
          nb::arg("builder"), nb::arg("operands"));

  nb::class_<func::CallOp, OpState>(m, "CallOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, func::FuncOp &func,
             const std::vector<Value> &args) {
            return func::CallOp::create(builder, builder.get_loc(), func, args);
          },
          nb::arg("builder"), nb::arg("func"), nb::arg("args"));
}

void init_affine_ops(nb::module_ &m) {
  // affine ops
  nb::class_<affine::AffineForOp, OpState>(m, "AffineForOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &lb,
             AffineMap lbMap, const std::vector<Value> &ub, AffineMap ubMap,
             int64_t step = 1) {
            return affine::AffineForOp::create(builder, builder.get_loc(), lb,
                                               lbMap, ub, ubMap, step);
          },
          nb::arg("builder"), nb::arg("lb_operands"), nb::arg("lb_map"),
          nb::arg("ub_operands"), nb::arg("ub_map"), nb::arg("step") = 1)
      .def_static(
          "create",
          [](AlloOpBuilder &builder, int64_t lb, int64_t ub, int64_t step = 1) {
            return affine::AffineForOp::create(builder, builder.get_loc(), lb,
                                               ub, step);
          })
      .def_prop_ro("induction_var", &affine::AffineForOp::getInductionVar)
      .def_prop_ro(
          "body", [](affine::AffineForOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def_prop_ro("upper_bound", &affine::AffineForOp::getUpperBound)
      .def_prop_ro("lower_bound", &affine::AffineForOp::getLowerBound)
      .def_prop_ro("constant_upper_bound",
                   &affine::AffineForOp::getConstantUpperBound)
      .def_prop_ro("constant_lower_bound",
                   &affine::AffineForOp::getConstantLowerBound)
      .def_prop_ro("has_constant_lower_bound",
                   &affine::AffineForOp::hasConstantLowerBound)
      .def_prop_ro("has_constant_upper_bound",
                   &affine::AffineForOp::hasConstantUpperBound)
      .def_prop_ro("step", &affine::AffineForOp::getStepAsInt)
      .def("has_constant_bounds", &affine::AffineForOp::hasConstantBounds);

  nb::class_<affine::AffineIfOp, OpState>(m, "AffineIfOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, IntegerSet set,
             const std::vector<Value> &operands, bool withElse = false) {
            return affine::AffineIfOp::create(builder, builder.get_loc(), set,
                                              operands, withElse);
          },
          nb::arg("builder"), nb::arg("set"), nb::arg("operands"),
          nb::arg("with_else") = false)
      .def_prop_ro("integer_set", &affine::AffineIfOp::getIntegerSet)
      .def_prop_ro("then_block", &affine::AffineIfOp::getThenBlock,
                   nb::rv_policy::reference)
      .def_prop_ro("else_block", &affine::AffineIfOp::getElseBlock,
                   nb::rv_policy::reference);
}

void init_scf_ops(nb::module_ &m) {
  // scf ops
  nb::class_<scf::ForOp, OpState>(m, "ForOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lb, Value &ub, Value &step,
             const std::vector<Value> &initArgs = {}) {
            return scf::ForOp::create(builder, builder.get_loc(), lb, ub, step,
                                      initArgs);
          },
          nb::arg("builder"), nb::arg("lb"), nb::arg("ub"), nb::arg("step"),
          nb::arg("init_args") = std::vector<Value>())
      .def_prop_ro("induction_var", &scf::ForOp::getInductionVar)
      .def_prop_ro(
          "body", [](scf::ForOp &self) { return self.getBody(0); },
          nb::rv_policy::reference);

  nb::class_<scf::IfOp, OpState>(m, "IfOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Type> &resultTypes,
             Value &cond, bool withElse = false) {
            return scf::IfOp::create(builder, builder.get_loc(), resultTypes,
                                     cond, withElse);
          },
          nb::arg("builder"), nb::arg("res_types"), nb::arg("cond"),
          nb::arg("with_else") = false)
      .def_prop_ro("then_block", &scf::IfOp::thenBlock,
                   nb::rv_policy::reference)
      .def_prop_ro("else_block", &scf::IfOp::elseBlock,
                   nb::rv_policy::reference)
      .def_prop_ro("then_yield", &scf::IfOp::thenYield)
      .def_prop_ro("else_yield", &scf::IfOp::elseYield);

  nb::class_<scf::YieldOp, OpState>(m, "YieldOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &results) {
            return scf::YieldOp::create(builder, builder.get_loc(), results);
          },
          nb::arg("builder"), nb::arg("results"));

  nb::class_<scf::WhileOp, OpState>(m, "WhileOp")
      .def_static("create",
                  [](AlloOpBuilder &builder,
                     const std::vector<Type> &resultTypes,
                     const std::vector<Value> &operands) {
                    return scf::WhileOp::create(builder, builder.get_loc(),
                                                resultTypes, operands);
                  })
      .def_prop_ro("before", &scf::WhileOp::getBefore, nb::rv_policy::reference)
      .def_prop_ro("after", &scf::WhileOp::getAfter, nb::rv_policy::reference);

  nb::class_<scf::ConditionOp, OpState>(m, "ConditionOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &cond,
             const std::vector<Value> &args) {
            return scf::ConditionOp::create(builder, builder.get_loc(), cond,
                                            args);
          },
          nb::arg("builder"), nb::arg("cond"), nb::arg("args"));

  nb::class_<scf::ParallelOp, OpState>(m, "ParallelOp")
      .def_static("create",
                  [](AlloOpBuilder &builder, const std::vector<Value> &lbs,
                     const std::vector<Value> &ubs,
                     const std::vector<Value> &steps,
                     const std::vector<Value> &initArgs = {}) {
                    return scf::ParallelOp::create(builder, builder.get_loc(),
                                                   lbs, ubs, steps, initArgs);
                  })
      .def_prop_ro(
          "body", [](scf::ParallelOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def_prop_ro("induction_vars", [](scf::ParallelOp &self) {
        auto ivs = self.getInductionVars();
        return std::vector<Value>(ivs.begin(), ivs.end());
      });

  nb::class_<scf::ReduceOp, OpState>(m, "ReduceOp")
      .def_static("create", [](AlloOpBuilder &builder,
                               const std::vector<Value> &operands) {
        return scf::ReduceOp::create(builder, builder.get_loc(), operands);
      });
}

void init_cf_ops(nb::module_ &m) {
  nb::class_<cf::CondBranchOp, OpState>(m, "CondBranchOp")
      .def_static("create", [](AlloOpBuilder &builder, Value &cond,
                               Block *trueDest, Block *falseDest) {
        return cf::CondBranchOp::create(builder, builder.get_loc(), cond,
                                        trueDest, falseDest);
      });

  nb::class_<cf::BranchOp, OpState>(m, "BranchOp")
      .def_static("create", [](AlloOpBuilder &builder, Block *dest,
                               const std::vector<Value> &args) {
        return cf::BranchOp::create(builder, builder.get_loc(), dest, args);
      });
}

void init_arith_ops(nb::module_ &m) {
  // constant ops
  (void)nb::class_<arith::ConstantOp, OpState>(m, "ConstantOp");

  nb::class_<arith::ConstantIntOp, arith::ConstantOp>(m, "ConstantIntOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, IntegerType &type,
             int64_t value) -> Value {
            return arith::ConstantIntOp::create(builder, builder.get_loc(),
                                                type, value);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"));

  nb::class_<arith::ConstantFloatOp, arith::ConstantOp>(m, "ConstantFloatOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Float32Type &type, float value) -> Value {
            return arith::ConstantFloatOp::create(builder, builder.get_loc(),
                                                  type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Float64Type &type, double value) -> Value {
            return arith::ConstantFloatOp::create(builder, builder.get_loc(),
                                                  type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Float16Type &type, float value) -> Value {
            return arith::ConstantFloatOp::create(builder, builder.get_loc(),
                                                  type, APFloat(value));
          })
      .def_static(
          "create",
          [](AlloOpBuilder &builder, BFloat16Type &type, float value) -> Value {
            // bf16 does not satisfy IEEE754, so we need to convert manually
            const llvm::fltSemantics &sem = type.getFloatSemantics();
            llvm::APFloat val(value);
            bool lost;
            val.convert(sem, llvm::APFloat::rmNearestTiesToEven, &lost);
            return arith::ConstantFloatOp::create(builder, builder.get_loc(),
                                                  type, val);
          });

  nb::class_<arith::ConstantIndexOp, arith::ConstantOp>(m, "ConstantIndexOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, int64_t value) -> Value {
            return arith::ConstantIndexOp::create(builder, builder.get_loc(),
                                                  value);
          },
          nb::arg("builder"), nb::arg("value"));

  // casts / conversions
  nb::class_<arith::SIToFPOp, OpState>(m, "SIToFPOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::SIToFPOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::UIToFPOp, OpState>(m, "UIToFPOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::UIToFPOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::FPToSIOp, OpState>(m, "FPToSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::FPToSIOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::FPToUIOp, OpState>(m, "FPToUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::FPToUIOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::ExtFOp, OpState>(m, "ExtFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::ExtFOp::create(builder, builder.get_loc(), dstType,
                                         src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::TruncFOp, OpState>(m, "TruncFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::TruncFOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::IndexCastOp, OpState>(m, "IndexCastOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Type &dstType, Value &src) -> Value {
            return arith::IndexCastOp::create(builder, builder.get_loc(),
                                              dstType, src);
          },
          nb::arg("builder"), nb::arg("dst_type"), nb::arg("src"));

  // integer extension / truncation / bitcast
  nb::class_<arith::ExtSIOp, OpState>(m, "ExtSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::ExtSIOp::create(builder, builder.get_loc(), dstType,
                                          src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::ExtUIOp, OpState>(m, "ExtUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::ExtUIOp::create(builder, builder.get_loc(), dstType,
                                          src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::BitcastOp, OpState>(m, "BitcastOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::BitcastOp::create(builder, builder.get_loc(), dstType,
                                            src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::TruncIOp, OpState>(m, "TruncIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &src, Type &dstType) -> Value {
            return arith::TruncIOp::create(builder, builder.get_loc(), dstType,
                                           src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  // floating ops
  nb::class_<arith::AddFOp, OpState>(m, "AddFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::AddFOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubFOp, OpState>(m, "SubFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::SubFOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulFOp, OpState>(m, "MulFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MulFOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivFOp, OpState>(m, "DivFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::DivFOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemFOp, OpState>(m, "RemFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::RemFOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::NegFOp, OpState>(m, "NegFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input) -> Value {
            return arith::NegFOp::create(builder, builder.get_loc(), input);
          },
          nb::arg("builder"), nb::arg("input"));

  // integer arithmetic
  nb::class_<arith::AddIOp, OpState>(m, "AddIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::AddIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubIOp, OpState>(m, "SubIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::SubIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulIOp, OpState>(m, "MulIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MulIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivSIOp, OpState>(m, "DivSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::DivSIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivUIOp, OpState>(m, "DivUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::DivUIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivSIOp, OpState>(m, "CeilDivSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::CeilDivSIOp::create(builder, builder.get_loc(), lhs,
                                              rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivUIOp, OpState>(m, "CeilDivUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::CeilDivUIOp::create(builder, builder.get_loc(), lhs,
                                              rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::FloorDivSIOp, OpState>(m, "FloorDivSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::FloorDivSIOp::create(builder, builder.get_loc(), lhs,
                                               rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemSIOp, OpState>(m, "RemSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::RemSIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemUIOp, OpState>(m, "RemUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::RemUIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // fused / special ops
  nb::class_<math::FmaOp, OpState>(m, "FmaOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &a, Value &b, Value &c) -> Value {
            return math::FmaOp::create(builder, builder.get_loc(), a, b, c);
          },
          nb::arg("builder"), nb::arg("a"), nb::arg("b"), nb::arg("c"));

  // shifts
  nb::class_<arith::ShLIOp, OpState>(m, "ShLIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::ShLIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRUIOp, OpState>(m, "ShRUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::ShRUIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRSIOp, OpState>(m, "ShRSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::ShRSIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // mins / maxs
  nb::class_<arith::MinSIOp, OpState>(m, "MinSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MinSIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinUIOp, OpState>(m, "MinUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MinUIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinimumFOp, OpState>(m, "MinimumFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MinimumFOp::create(builder, builder.get_loc(), lhs,
                                             rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinNumFOp, OpState>(m, "MinNumFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MinNumFOp::create(builder, builder.get_loc(), lhs,
                                            rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxSIOp, OpState>(m, "MaxSIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MaxSIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxUIOp, OpState>(m, "MaxUIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MaxUIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaximumFOp, OpState>(m, "MaximumFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MaximumFOp::create(builder, builder.get_loc(), lhs,
                                             rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxNumFOp, OpState>(m, "MaxNumFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::MaxNumFOp::create(builder, builder.get_loc(), lhs,
                                            rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (int)
  nb::class_<arith::CmpIOp, OpState>(m, "CmpIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, std::size_t pred, Value &lhs,
             Value &rhs) -> Value {
            return arith::CmpIOp::create(
                builder, builder.get_loc(),
                static_cast<arith::CmpIPredicate>(pred), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (float)
  nb::class_<arith::CmpFOp, OpState>(m, "CmpFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, std::size_t pred, Value &lhs,
             Value &rhs) -> Value {
            return arith::CmpFOp::create(
                builder, builder.get_loc(),
                static_cast<arith::CmpFPredicate>(pred), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // logical
  nb::class_<arith::AndIOp, OpState>(m, "AndIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::AndIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::XOrIOp, OpState>(m, "XOrIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::XOrIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::OrIOp, OpState>(m, "OrIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs) -> Value {
            return arith::OrIOp::create(builder, builder.get_loc(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SelectOp, OpState>(m, "SelectOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &condition, Value &trueValue,
             Value &falseValue) -> Value {
            return arith::SelectOp::create(builder, builder.get_loc(),
                                           condition, trueValue, falseValue);
          },
          nb::arg("builder"), nb::arg("condition"), nb::arg("true_value"),
          nb::arg("false_value"));
}

void init_math_ops(nb::module_ &m) {

  nb::class_<math::FloorOp, OpState>(m, "FloorOp")

      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::FloorOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::CeilOp, OpState>(m, "CeilOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::CeilOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::ExpOp, OpState>(m, "ExpOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::ExpOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::Exp2Op, OpState>(m, "Exp2Op")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::Exp2Op::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::CosOp, OpState>(m, "CosOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::CosOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::SinOp, OpState>(m, "SinOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::SinOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::LogOp, OpState>(m, "LogOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::LogOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::Log2Op, OpState>(m, "Log2Op")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::Log2Op::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::ErfOp, OpState>(m, "ErfOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::ErfOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::SqrtOp, OpState>(m, "SqrtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::SqrtOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::RsqrtOp, OpState>(m, "RsqrtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::RsqrtOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::AbsFOp, OpState>(m, "AbsFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::AbsFOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::AbsIOp, OpState>(m, "AbsIOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::AbsIOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::PowFOp, OpState>(m, "PowFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &base, Value &exponent) -> Value {
            return math::PowFOp::create(builder, builder.get_loc(), base,
                                        exponent);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"));

  nb::class_<math::TanOp, OpState>(m, "TanOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &val) -> Value {
            return math::TanOp::create(builder, builder.get_loc(), val);
          },
          nb::arg("builder"), nb::arg("val"));
}

void init_tensor_ops(nb::module_ &m) {
  nb::class_<tensor::ExtractOp, OpState>(m, "ExtractOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &tensor,
             const std::vector<Value> &indices) -> Value {
            return tensor::ExtractOp::create(builder, builder.get_loc(), tensor,
                                             indices);
          },
          nb::arg("builder"), nb::arg("tensor"), nb::arg("indices"));

  nb::class_<tensor::InsertOp, OpState>(m, "InsertOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &value, Value &tensor,
             const std::vector<Value> &indices) -> Value {
            return Value{tensor::InsertOp::create(builder, builder.get_loc(),
                                                  value, tensor, indices)};
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("tensor"),
          nb::arg("indices"));

  nb::class_<tensor::SplatOp, OpState>(m, "SplatOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &value,
             const std::vector<int64_t> &shape) -> Value {
            return Value{tensor::SplatOp::create(builder, builder.get_loc(),
                                                 value, shape)};
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("shape"));

  nb::class_<tensor::CastOp, OpState>(m, "CastOp")

      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Type &dstType) -> Value {
            return Value{tensor::CastOp::create(builder, builder.get_loc(),
                                                dstType,

                                                input)};
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("dst_type"));

  nb::class_<tensor::EmptyOp, OpState>(m, "EmptyOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<int64_t> &shape,
             Type &elementType) -> Value {
            return Value{tensor::EmptyOp::create(builder, builder.get_loc(),
                                                 shape,

                                                 elementType)};
          },
          nb::arg("builder"), nb::arg("shape"), nb::arg("element_type"))

      .def_static("create", [](AlloOpBuilder &builder, Type &type) -> Value {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          return Value{tensor::EmptyOp::create(
              builder, builder.get_loc(), tensor.getShape(),
              tensor.getElementType(), tensor.getEncoding())};
        }
        if (auto memref = dyn_cast<MemRefType>(type)) {
          return Value{tensor::EmptyOp::create(
              builder, builder.get_loc(), memref.getShape(),
              memref.getElementType(), memref.getMemorySpace())};
        }
        throw nb::type_error("Unsupported type for tensor.EmptyOp");
      });
}

void init_memref_ops(nb::module_ &m) {
  nb::class_<memref::LoadOp, OpState>(m, "LoadOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &memref,
             const std::vector<Value> &indices) {
            return memref::LoadOp::create(builder, builder.get_loc(), memref,
                                          indices);
          },
          nb::arg("builder"), nb::arg("memref"), nb::arg("indices"));
  nb::class_<memref::StoreOp, OpState>(m, "StoreOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &value, Value &memref,
             const std::vector<Value> &indices) {
            return memref::StoreOp::create(builder, builder.get_loc(), value,
                                           memref, indices);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("memref"),
          nb::arg("indices"));

  nb::class_<memref::AllocOp, OpState>(m, "AllocOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, MemRefType &type) -> Value {
            return memref::AllocOp::create(builder, builder.get_loc(), type);
          },
          nb::arg("builder"), nb::arg("type"));
}

void init_linalg_ops(nb::module_ &m) {

  nb::class_<linalg::MatmulOp, OpState>(m, "MatmulOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &result) {
            return linalg::MatmulOp::create(builder, builder.get_loc(),

                                            {lhs, rhs}, result);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("result"));

  nb::class_<linalg::FillOp, OpState>(m, "FillOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &value, Value &output) {
            return linalg::FillOp::create(builder, builder.get_loc(), value,
                                          output);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("output"));

  nb::class_<linalg::BroadcastOp, OpState>(m, "BroadcastOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init,
             const std::vector<int64_t> &dims) {
            return linalg::BroadcastOp::create(builder, builder.get_loc(),
                                               input, init, dims);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"),
          nb::arg("dims"));

  nb::class_<linalg::AddOp, OpState>(m, "AddOp")

      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::AddOp::create(builder, builder.get_loc(), {lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::SubOp, OpState>(m, "SubOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::SubOp::create(builder, builder.get_loc(), {lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::MulOp, OpState>(m, "MulOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::MulOp::create(builder, builder.get_loc(), {lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::DivOp, OpState>(m, "DivOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::DivOp::create(builder, builder.get_loc(), {lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::DivUnsignedOp, OpState>(m, "DivUnsignedOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::DivUnsignedOp::create(builder, builder.get_loc(),
                                                 {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::PowFOp, OpState>(m, "PowFOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &base, Value &exponent,
             Value &init) {
            return linalg::PowFOp::create(builder, builder.get_loc(),
                                          {base, exponent}, init);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"),
          nb::arg("init"));

  nb::class_<linalg::FloorOp, OpState>(m, "FloorOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::FloorOp::create(builder, builder.get_loc(), input,
                                           init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::ExpOp, OpState>(m, "ExpOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::ExpOp::create(builder, builder.get_loc(), input,
                                         init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::LogOp, OpState>(m, "LogOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::LogOp::create(builder, builder.get_loc(), input,
                                         init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::SqrtOp, OpState>(m, "SqrtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::SqrtOp::create(builder, builder.get_loc(), input,
                                          init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::ReciprocalOp, OpState>(m, "ReciprocalOp")

      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::ReciprocalOp::create(builder, builder.get_loc(),
                                                input, init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::RsqrtOp, OpState>(m, "RsqrtOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::RsqrtOp::create(builder, builder.get_loc(), input,
                                           init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::SquareOp, OpState>(m, "SquareOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &input, Value &init) {
            return linalg::SquareOp::create(builder, builder.get_loc(), input,
                                            init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::DotOp, OpState>(m, "DotOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, Value &lhs, Value &rhs, Value &init) {
            return linalg::DotOp::create(builder, builder.get_loc(), {lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::enum_<utils::IteratorType>(m, "IteratorType")
      .value("PAR", utils::IteratorType::parallel)
      .value("RED", utils::IteratorType::reduction)
      .export_values();

  nb::class_<linalg::GenericOp, OpState>(m, "GenericOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Type> &resultTypes,
             const std::vector<Value> &inputs,
             const std::vector<Value> &outputs,
             const std::vector<AffineMap> &indexingMaps,
             const std::vector<utils::IteratorType> &iteratorTypes) {
            return linalg::GenericOp::create(builder, builder.get_loc(),
                                             resultTypes, inputs, outputs,
                                             indexingMaps, iteratorTypes);
          })
      .def_prop_ro("body",
                   [](linalg::GenericOp &self) { return self.getBody(); })
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

  nb::class_<linalg::YieldOp, OpState>(m, "YieldOp")
      .def_static(
          "create",
          [](AlloOpBuilder &builder, const std::vector<Value> &values) {
            return linalg::YieldOp::create(builder, builder.get_loc(), values);
          },
          nb::arg("builder"), nb::arg("values"));
}
