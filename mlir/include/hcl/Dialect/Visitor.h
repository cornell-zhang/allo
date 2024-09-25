/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: ScaleHLS
 * https://github.com/hanchenye/scalehls
 */

#ifndef HCL_DIALECT_HLSCPP_VISITOR_H
#define HCL_DIALECT_HLSCPP_VISITOR_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

/// This class is a visitor for SSACFG operation nodes.
template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class HLSCppVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // SCF statements.
            scf::ForOp, scf::IfOp, scf::ParallelOp, scf::ReduceOp,
            scf::ReduceReturnOp, scf::YieldOp,
            // Affine statements.
            affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp,
            affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp,
            affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineYieldOp,
            affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
            affine::AffineDmaStartOp, affine::AffineDmaWaitOp,
            // Memref-related statements.
            memref::AllocOp, memref::AllocaOp, memref::LoadOp, memref::StoreOp,
            memref::GetGlobalOp, hcl::GetGlobalFixedOp, memref::GlobalOp,
            memref::DeallocOp, memref::DmaStartOp, memref::DmaWaitOp,
            memref::ViewOp, memref::SubViewOp, memref::ReinterpretCastOp,
            memref::AtomicRMWOp,
            // Tensor-related statements.
            tensor::ExtractOp, tensor::InsertOp, tensor::SplatOp, memref::DimOp,
            memref::RankOp,
            // Unary expressions.
            math::AbsFOp, math::AbsIOp, math::CeilOp, math::CosOp, math::SinOp,
            math::TanhOp, math::SqrtOp, math::RsqrtOp, math::ExpOp,
            math::Exp2Op, math::PowFOp, math::LogOp, math::Log2Op,
            math::Log10Op, arith::NegFOp,
            // Float binary expressions.
            arith::CmpFOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
            arith::DivFOp, arith::RemFOp,
            // Integer binary expressions.
            arith::CmpIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
            arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
            arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp,
            // Logical expressions.
            arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, hcl::GetIntBitOp, hcl::SetIntBitOp,
            hcl::GetIntSliceOp, hcl::SetIntSliceOp, hcl::BitReverseOp,
            // Special operations.
            func::CallOp, func::ReturnOp, arith::SelectOp, arith::ConstantOp,
            arith::TruncIOp, arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp,
            arith::ExtFOp, arith::IndexCastOp, arith::UIToFPOp, arith::SIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp, arith::BitcastOp,
            hcl::FixedToFloatOp, hcl::FloatToFixedOp, hcl::IntToFixedOp,
            hcl::FixedToIntOp, hcl::FixedToFixedOp, UnrealizedConversionCastOp,
            // HCL operations.
            hcl::CreateLoopHandleOp, hcl::CreateOpHandleOp, hcl::AddFixedOp,
            hcl::SubFixedOp, hcl::MulFixedOp, hcl::DivFixedOp, hcl::CmpFixedOp,
            hcl::MinFixedOp, hcl::MaxFixedOp, hcl::PrintOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitOp(opNode, args...);
            })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation.");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // SCF statements.
  HANDLE(scf::ForOp);
  HANDLE(scf::IfOp);
  HANDLE(scf::ParallelOp);
  HANDLE(scf::ReduceOp);
  HANDLE(scf::ReduceReturnOp);
  HANDLE(scf::YieldOp);

  // Affine statements.
  HANDLE(affine::AffineForOp);
  HANDLE(affine::AffineIfOp);
  HANDLE(affine::AffineParallelOp);
  HANDLE(affine::AffineApplyOp);
  HANDLE(affine::AffineMaxOp);
  HANDLE(affine::AffineMinOp);
  HANDLE(affine::AffineLoadOp);
  HANDLE(affine::AffineStoreOp);
  HANDLE(affine::AffineYieldOp);
  HANDLE(affine::AffineVectorLoadOp);
  HANDLE(affine::AffineVectorStoreOp);
  HANDLE(affine::AffineDmaStartOp);
  HANDLE(affine::AffineDmaWaitOp);

  // Memref-related statements.
  HANDLE(memref::AllocOp);
  HANDLE(memref::AllocaOp);
  HANDLE(memref::LoadOp);
  HANDLE(memref::StoreOp);
  HANDLE(memref::GetGlobalOp);
  HANDLE(hcl::GetGlobalFixedOp);
  HANDLE(memref::GlobalOp);
  HANDLE(memref::DeallocOp);
  HANDLE(memref::DmaStartOp);
  HANDLE(memref::DmaWaitOp);
  HANDLE(memref::AtomicRMWOp);
  HANDLE(memref::ViewOp);
  HANDLE(memref::SubViewOp);
  HANDLE(memref::ReinterpretCastOp);

  // Tensor-related statements.
  HANDLE(tensor::ExtractOp);
  HANDLE(tensor::InsertOp);
  HANDLE(tensor::SplatOp);
  HANDLE(memref::DimOp);
  HANDLE(memref::RankOp);

  // Unary expressions.
  HANDLE(math::AbsFOp);
  HANDLE(math::AbsIOp);
  HANDLE(math::CeilOp);
  HANDLE(math::CosOp);
  HANDLE(math::SinOp);
  HANDLE(math::TanhOp);
  HANDLE(math::SqrtOp);
  HANDLE(math::RsqrtOp);
  HANDLE(math::ExpOp);
  HANDLE(math::Exp2Op);
  HANDLE(math::PowFOp);
  HANDLE(math::LogOp);
  HANDLE(math::Log2Op);
  HANDLE(math::Log10Op);
  HANDLE(arith::NegFOp);

  // Float binary expressions.
  HANDLE(arith::CmpFOp);
  HANDLE(arith::AddFOp);
  HANDLE(arith::SubFOp);
  HANDLE(arith::MulFOp);
  HANDLE(arith::DivFOp);
  HANDLE(arith::RemFOp);

  // Integer binary expressions.
  HANDLE(arith::CmpIOp);
  HANDLE(arith::AddIOp);
  HANDLE(arith::SubIOp);
  HANDLE(arith::MulIOp);
  HANDLE(arith::DivSIOp);
  HANDLE(arith::RemSIOp);
  HANDLE(arith::DivUIOp);
  HANDLE(arith::RemUIOp);
  HANDLE(arith::MaxSIOp);
  HANDLE(arith::MinSIOp);
  HANDLE(arith::MaxUIOp);
  HANDLE(arith::MinUIOp);

  // Bit operations.
  HANDLE(arith::XOrIOp);
  HANDLE(arith::AndIOp);
  HANDLE(arith::OrIOp);
  HANDLE(arith::ShLIOp);
  HANDLE(arith::ShRSIOp);
  HANDLE(arith::ShRUIOp);
  HANDLE(hcl::GetIntBitOp);
  HANDLE(hcl::SetIntBitOp);
  HANDLE(hcl::GetIntSliceOp);
  HANDLE(hcl::SetIntSliceOp);
  HANDLE(hcl::BitReverseOp);
  HANDLE(hcl::FixedToFloatOp);
  HANDLE(hcl::FloatToFixedOp);
  HANDLE(hcl::IntToFixedOp);
  HANDLE(hcl::FixedToIntOp);
  HANDLE(hcl::FixedToFixedOp);
  // Logical operations
  HANDLE(hcl::LogicalAndOp);
  HANDLE(hcl::LogicalOrOp);

  // Special operations.
  HANDLE(func::CallOp);
  HANDLE(func::ReturnOp);
  HANDLE(arith::SelectOp);
  HANDLE(arith::ConstantOp);
  HANDLE(arith::TruncIOp);
  HANDLE(arith::TruncFOp);
  HANDLE(arith::ExtUIOp);
  HANDLE(arith::ExtSIOp);
  HANDLE(arith::ExtFOp);
  HANDLE(arith::IndexCastOp);
  HANDLE(arith::UIToFPOp);
  HANDLE(arith::SIToFPOp);
  HANDLE(arith::FPToUIOp);
  HANDLE(arith::FPToSIOp);
  HANDLE(arith::BitcastOp);
  HANDLE(UnrealizedConversionCastOp);

  // HCL operations
  HANDLE(hcl::CreateLoopHandleOp);
  HANDLE(hcl::CreateOpHandleOp);
  HANDLE(hcl::PrintOp);

  // Fixed point operations
  HANDLE(hcl::AddFixedOp);
  HANDLE(hcl::SubFixedOp);
  HANDLE(hcl::MulFixedOp);
  HANDLE(hcl::DivFixedOp);
  HANDLE(hcl::CmpFixedOp);
  HANDLE(hcl::MinFixedOp);
  HANDLE(hcl::MaxFixedOp);

#undef HANDLE
};
} // namespace hcl
} // namespace mlir

#endif // HCL_DIALECT_HLSCPP_VISITOR_H