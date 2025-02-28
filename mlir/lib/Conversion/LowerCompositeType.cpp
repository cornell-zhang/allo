/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// LowerCompositeType Pass
// This file defines the lowering of composite types such as structs.
// This pass is separated from AlloToLLVM because it could be used
// in other backends as well, such as HLS backend.
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {

void deadStructConstructElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> structConstructOps;
  func.walk([&](Operation *op) {
    if (auto structConstructOp = dyn_cast<StructConstructOp>(op)) {
      structConstructOps.push_back(structConstructOp);
    }
  });
  std::reverse(structConstructOps.begin(), structConstructOps.end());
  for (auto op : structConstructOps) {
    auto structValue = op->getResult(0);
    if (structValue.use_empty()) {
      op->erase();
    }
  }
}

void deadMemRefAllocElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> memRefAllocOps;
  func.walk([&](Operation *op) {
    if (auto memRefAllocOp = dyn_cast<memref::AllocOp>(op)) {
      memRefAllocOps.push_back(memRefAllocOp);
    }
  });
  std::reverse(memRefAllocOps.begin(), memRefAllocOps.end());
  for (auto op : memRefAllocOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void deadAffineLoadElimination(func::FuncOp &func) {
  SmallVector<Operation *, 8> affineLoadOps;
  func.walk([&](Operation *op) {
    if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      affineLoadOps.push_back(affineLoadOp);
    }
  });
  std::reverse(affineLoadOps.begin(), affineLoadOps.end());
  for (auto op : affineLoadOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

void lowerStructType(func::FuncOp &func, ModuleOp &mod) {
  bool structAsArg = false;
  std::map<mlir::detail::ValueImpl *, SmallVector<Value, 8>>
      structMemRef2fieldMemRefs;
  // First, process structs variables from function arguments
  Region::BlockArgListType funcArgs = func.getArguments();
  uint origFuncArgNum = funcArgs.size();
  FunctionType functionType = func.getFunctionType();
  SmallVector<Type, 20> decomposedArgTypes =
      llvm::to_vector<20>(func.getArgumentTypes()); // The argument type list
                                                    // after decomposing structs
  uint argInsertPos = funcArgs.size();
  std::map<uint, uint> structArgNo2memberArgStartNo;
  std::map<uint, uint> structArgNo2memberArgNum;
  for (int i = 0; i < origFuncArgNum; i++) {
    Region::BlockArgListType funcArgs = func.getArguments();
    Region &functionBody = func.getFunctionBody(); // Insert argument use this
    BlockArgument arg = funcArgs[i];
    // Look for arguments with struct type
    Type argType = arg.getType();
    if (const MemRefType memrefArgType = argType.dyn_cast<MemRefType>()) {
      Type elementType = memrefArgType.getElementType();
      if (const StructType structArgType = elementType.dyn_cast<StructType>()) {
        // Confirmed that `arg` is a `memref<Struct>` argument
        // Append members to function arguments at back
        uint structArgNo = arg.getArgNumber();
        uint memberArgStartPos = argInsertPos;
        structArgNo2memberArgStartNo.insert(
            std::make_pair(structArgNo, memberArgStartPos));
        ArrayRef<Type> memberTypes = structArgType.getElementTypes();
        for (const Type memberType : memberTypes) { // Memref member
          if (const MemRefType memrefMemberType =
                  memberType.dyn_cast<MemRefType>()) {
            functionBody.addArgument(memberType, func.getLoc());
            decomposedArgTypes.push_back(memrefMemberType);
          } else { // Create memref pointer for other member type
            MemRefType memberPtr =
                MemRefType::get(ArrayRef<int64_t>(), memberType);
            functionBody.addArgument(memberPtr, func.getLoc());
            decomposedArgTypes.push_back(memberPtr);
          }
          argInsertPos++;
        }
        // All members are appended
        structArgNo2memberArgNum.insert(
            std::make_pair(structArgNo, argInsertPos - memberArgStartPos));
        // The original struct argument is not removed yet here
      }
    }
  }
  // Further actions when struct arguments exist
  if (decomposedArgTypes.size() > origFuncArgNum) {
    structAsArg = true;
    // Collect the map between original struct arguments and newly added member
    // arguments
    for (auto pair : structArgNo2memberArgStartNo) {
      SmallVector<Value, 8> memberArgs;
      const BlockArgument &structArg = func.getArgument(pair.first);
      uint memberArgsNum = structArgNo2memberArgNum[pair.first];
      for (uint i = 0; i < memberArgsNum; i++) {
        const BlockArgument &memberArg = func.getArgument(i + pair.second);
        memberArgs.push_back(memberArg);
      }
      structMemRef2fieldMemRefs.insert(
          std::make_pair(structArg.cast<Value>().getImpl(), memberArgs));
    }
  }

  SmallVector<Operation *, 10> structGetOps;
  func.walk([&](Operation *op) {
    if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
      structGetOps.push_back(structGetOp);
    }
  });

  for (auto op : structGetOps) {
    // Collect info from structGetOp
    auto structGetOp = dyn_cast<StructGetOp>(op);
    Value struct_value = structGetOp->getOperand(0);
    Value struct_field = structGetOp->getResult(0);
    Location loc = op->getLoc();
    auto index = structGetOp.getIndex();

    // This flag indicates whether we can erase
    // struct construct and relevant ops.
    bool erase_struct_construct = false;

    // The defOp can be either a StructConstructOp or
    // a load from a memref.
    // Load: we are operating on a memref of struct
    // Construct: we are operating on a struct value
    Operation *defOp = struct_value.getDefiningOp();
    if (auto affine_load = dyn_cast<affine::AffineLoadOp>(defOp)) {
      // Case 1: defOp is loadOp from memref
      // Note: the idea to lower struct from memref is to
      // first create a memref for each struct field, and then
      // add store operation to those memrefs right after the
      // struct construction. With that, we can replace
      // struct get with loading from field memrefs.

      // Step1: create memref for each field
      Value struct_memref = affine_load.getMemref();
      // Try to find field_memrefs associated with this struct_memref
      SmallVector<Value, 4> field_memrefs;
      auto it = structMemRef2fieldMemRefs.find(struct_memref.getImpl());
      if (it == structMemRef2fieldMemRefs.end()) {
        // Create a memref for each field
        OpBuilder builder(struct_memref.getDefiningOp());
        StructType struct_type = struct_value.getType().cast<StructType>();

        for (Type field_type : struct_type.getElementTypes()) {
          MemRefType newMemRefType = struct_memref.getType()
                                         .cast<MemRefType>()
                                         .clone(field_type)
                                         .cast<MemRefType>();
          Value field_memref =
              builder.create<memref::AllocOp>(loc, newMemRefType);
          field_memrefs.push_back(field_memref);
        }
        structMemRef2fieldMemRefs.insert(
            std::make_pair(struct_memref.getImpl(), field_memrefs));
        erase_struct_construct = true;
      } else {
        field_memrefs.append(it->second);
        erase_struct_construct = false;
      }

      // Step2: add store to each field memref
      for (auto &use : struct_memref.getUses()) {
        if (auto storeOp = dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
          // Find a storeOp to the struct memref, we add
          // store to each field memref here.
          OpBuilder builder(storeOp);
          for (const auto &field_memref_en : llvm::enumerate(field_memrefs)) {
            auto field_memref = field_memref_en.value();
            auto field_index = field_memref_en.index();
            // Find the struct_construct op
            auto struct_construct_op = dyn_cast<StructConstructOp>(
                storeOp.getOperand(0).getDefiningOp());
            builder.create<affine::AffineStoreOp>(
                loc, struct_construct_op.getOperand(field_index), field_memref,
                storeOp.getAffineMap(), storeOp.getIndices());
          }
          // erase the storeOp that stores to the struct memref
          if (erase_struct_construct) {
            storeOp.erase();
          }
          break;
        }
      }

      // Step3: replace structGetOp with load from field memrefs
      // Only for non-arguments
      if (field_memrefs[index].getDefiningOp() != nullptr) {
        OpBuilder load_builder(op);
        Value loaded_field = load_builder.create<affine::AffineLoadOp>(
            loc, field_memrefs[index], affine_load.getAffineMap(),
            affine_load.getIndices());
        struct_field.replaceAllUsesWith(loaded_field);
      } else { // For arguments, just replace with themselves
        struct_field.replaceAllUsesWith(field_memrefs[index]);
      }
      op->erase(); // erase structGetOp
      auto allStructGets = affine_load->getUsers();
      // Remove the load after no struct_get using it is left
      if (allStructGets.empty())
        affine_load->erase();
    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(defOp)) {
      // Case 2: defOp is a struct construction op
      Value replacement = defOp->getOperand(index);
      struct_field.replaceAllUsesWith(replacement);
      op->erase(); // erase structGetOp
    } else {
      llvm_unreachable("unexpected defOp for structGetOp");
    }
  }

  // Final process of function with struct arguments
  if (structAsArg) {
    // Finally remove the struct arguments
    uint erasedStructArgNum = 0;
    for (auto pair : structArgNo2memberArgNum) {
      Region &functionBody = func.getFunctionBody(); // Insert argument use this
      uint structArgIndex = pair.first - erasedStructArgNum;
      functionBody.eraseArgument(structArgIndex);
      decomposedArgTypes.erase(&decomposedArgTypes[structArgIndex]);
      erasedStructArgNum++; // Shift left the indices after removing one
                            // argument
    }
    // Set the updated function signature
    FunctionType newFuncType = FunctionType::get(
        func.getContext(), decomposedArgTypes, func.getResultTypes());
    func.setType(newFuncType);

    // Update callers of this function
    // Reference:
    // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/MemRef/Transforms/NormalizeMemRefs.cpp
    llvm::SmallDenseSet<func::FuncOp, 8> funcOpsToUpdate;
    std::optional<SymbolTable::UseRange> symbolUses = func.getSymbolUses(mod);
    for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
      Operation *userOp = symbolUse.getUser();
      OpBuilder builder(userOp);
      auto callOp = dyn_cast<func::CallOp>(userOp);
      if (!callOp)
        continue;

      Operation::operand_range orig_params = callOp.getArgOperands();
      // Find parameters of memref<struct> type
      for (int i = 0; i < origFuncArgNum; i++) {
        MutableOperandRange parameters = callOp.getArgOperandsMutable();
        auto &arg = parameters[i];
        auto value = arg.get();
        if (!llvm::ValueIsPresent<Value>::isPresent(value)) {
          continue;
        }
        Type type = value.getType();
        if (MemRefType memrefType = dyn_cast<MemRefType>(type)) {
          if (StructType structType =
                  dyn_cast<StructType>(memrefType.getElementType())) {
            // Struct param found
            // Find the last store to the memref<struct> before the call
            // to locate the StructConstruct op
            Value::user_range memrefUses = value.getUsers();
            affine::AffineStoreOp *storeOp = nullptr;
            for (Operation *memrefUse : memrefUses) {
              if (auto storeOpTest = dyn_cast<affine::AffineStoreOp>(memrefUse))
                storeOp = &storeOpTest;
            }
            assert(storeOp != nullptr);
            const Value &structStored = storeOp->getValueToStore();
            // Find the struct members from the construct op
            StructConstructOp structConOp =
                dyn_cast<StructConstructOp>(structStored.getDefiningOp());
            assert(structConOp != nullptr);
            Operation::operand_range origMembers = structConOp->getOperands();
            // Pass the original members of the struct
            for (const Value &member : origMembers) {
              parameters.append(member);
            }
          }
        }
      }
      uint removedParamNum = 0;
      for (auto pair : structArgNo2memberArgNum) {
        MutableOperandRange parameters = callOp.getArgOperandsMutable();
        uint removeIndex = pair.first - removedParamNum;
        parameters.erase(removeIndex);
        removedParamNum++;
      }
    }
  }

  // Run DCE after all struct get is folded
  deadAffineLoadElimination(func);
  deadStructConstructElimination(func);
  deadMemRefAllocElimination(func);
}

Value buildStructFromInt(OpBuilder &builder, Location loc, Value int_value,
                         StructType struct_type, int lo) {
  SmallVector<Value, 4> struct_elements;
  for (Type field_type : struct_type.cast<StructType>().getElementTypes()) {
    if (field_type.isa<IntegerType>()) {
      int field_bitwidth = field_type.getIntOrFloatBitWidth();
      int hi = lo + (field_bitwidth - 1);
      Value hi_idx = builder.create<mlir::arith::ConstantIndexOp>(loc, hi);
      Value lo_idx = builder.create<mlir::arith::ConstantIndexOp>(loc, lo);
      lo += field_bitwidth;
      Value field_value = builder.create<mlir::allo::GetIntSliceOp>(
          loc, field_type, int_value, hi_idx, lo_idx);
      struct_elements.push_back(field_value);
    } else if (field_type.isa<StructType>()) {
      struct_elements.push_back(buildStructFromInt(
          builder, loc, int_value, field_type.cast<StructType>(), lo));
    } else {
      llvm_unreachable("unexpected type");
    }
  }
  return builder.create<StructConstructOp>(loc, struct_type, struct_elements);
}

void lowerIntToStructOp(func::FuncOp &func) {
  SmallVector<Operation *, 10> intToStructOps;
  func.walk([&](Operation *op) {
    if (auto intToStructOp = dyn_cast<IntToStructOp>(op)) {
      intToStructOps.push_back(intToStructOp);
    }
  });

  for (auto op : intToStructOps) {
    auto intToStructOp = dyn_cast<IntToStructOp>(op);
    Value struct_value = intToStructOp->getResult(0);
    Value int_value = intToStructOp->getOperand(0);
    Location loc = op->getLoc();
    // Step1: create get_bit op for each field
    StructType struct_type = struct_value.getType().cast<StructType>();
    OpBuilder builder(op);
    int lo = 0;
    // Step2: create struct construct op
    auto struct_construct =
        buildStructFromInt(builder, loc, int_value, struct_type, lo);
    // Step3: replace intToStructOp with struct construct
    struct_value.replaceAllUsesWith(struct_construct);
  }

  // Erase intToStructOps
  for (auto op : intToStructOps) {
    op->erase();
  }
}

bool isLegal(func::FuncOp &func) {
  bool legal = true;
  func.walk([&](Operation *op) {
    if (auto structGetOp = dyn_cast<StructGetOp>(op)) {
      legal = false;
      llvm::errs()
          << "Error: [Pass][LowerCompositeType] structGetOp is not legal: "
          << *op << "\n";
      WalkResult::interrupt();
    } else if (auto structConstructOp = dyn_cast<StructConstructOp>(op)) {
      legal = false;
      llvm::errs() << "Error: [Pass][LowerCompositeType] structConstructOp is "
                      "not legal: "
                   << *op << "\n";
      WalkResult::interrupt();
    }
  });
  return legal;
}

/// Pass entry point
bool applyLowerCompositeType(ModuleOp &mod) {
  // First check if there are any struct operations to lower
  bool hasStructOps = false;
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<StructGetOp, StructConstructOp, IntToStructOp>(op)) {
        hasStructOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasStructOps)
      break;
  }

  // If no struct operations, return success without doing anything
  if (!hasStructOps) {
    return true;
  }

  // Only apply transformations if we found struct operations
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerIntToStructOp(func);
  }
  // Only run DCE if we actually did some transformations
  applyMemRefDCE(mod);

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerStructType(func, mod);
  }
  // Run final DCE pass
  applyMemRefDCE(mod);

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    if (!isLegal(func)) {
      return false;
    }
  }
  return true;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerCompositeTypeTransformation
    : public LowerCompositeTypeBase<AlloLowerCompositeTypeTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerCompositeType(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass() {
  return std::make_unique<AlloLowerCompositeTypeTransformation>();
}
} // namespace allo
} // namespace mlir