#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>

#define DEBUG_TYPE "llvm-to-affine-access"

namespace mlir {
#define GEN_PASS_DEF_LLVMTOMEMREFPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

struct AffineAccess {
  Value base;
  SmallVector<Value> inputs;
  AffineMap map;
};

/// See llvm/Support/Alignment.h
static AffineExpr alignTo(AffineExpr expr, uint64_t a) {
  return (expr + a - 1).floorDiv(a) * a;
}

static std::optional<AffineExpr> getGepAffineExpr(const DataLayout &dataLayout,
                                                  LLVM::GEPOp gep) {
  // TODO what happens if we get a negative index
  auto context = gep.getContext();

  Type currentType = gep.getElemType();
  AffineExpr expr;
  auto constIndex = dyn_cast<IntegerAttr>(gep.getIndices()[0]);
  if (constIndex)
    expr = getAffineConstantExpr(constIndex.getInt(), context);
  else
    // We still have no way of knowing whether this will be a symbol or dim so
    // make it a dim for now.
    expr = getAffineDimExpr(0, context);
  AffineExpr offset = expr * dataLayout.getTypeSize(currentType);

  for (auto &&[i, index] :
       llvm::drop_begin(llvm::enumerate(gep.getIndices()))) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    AffineExpr expr;
    if (constIndex)
      expr = getAffineConstantExpr(constIndex.getInt(), context);
    else
      expr = getAffineDimExpr(i, context);
    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVM::LLVMArrayType arrayType) {
              offset = offset + expr * dataLayout.getTypeSize(
                                           arrayType.getElementType());
              currentType = arrayType.getElementType();
              return false;
            })
            .Case([&](LLVM::LLVMStructType structType) {
              ArrayRef<Type> body = structType.getBody();
              int64_t indexInt;
              auto constIndex = dyn_cast<IntegerAttr>(index);
              if (constIndex)
                indexInt = constIndex.getInt();
              else
                return true;

              for (uint32_t i : llvm::seq(indexInt)) {
                if (!structType.isPacked())
                  offset =
                      alignTo(offset, dataLayout.getTypeABIAlignment(body[i]));
                offset = offset + dataLayout.getTypeSize(body[i]);
              }

              // Align for the current type as well.
              if (!structType.isPacked())
                offset = alignTo(
                    offset, dataLayout.getTypeABIAlignment(body[indexInt]));
              currentType = body[indexInt];
              return false;
            })
            .Default([&](Type type) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Unsupported type for offset computations" << type
                         << "\n");
              return true;
            });

    if (shouldCancel)
      return std::nullopt;
  }

  return offset;
}

static bool isValidBlockArgument(Value v) {
  if (auto ba = dyn_cast<BlockArgument>(v))
    return isa<affine::AffineForOp, affine::AffineParallelOp, func::FuncOp>(
        ba.getOwner()->getParentOp());
  return false;
}

static FailureOr<AffineMap> buildAffineMap(MLIRContext *ctx,
                                           const DataLayout &dataLayout,
                                           SmallVector<Value> indices) {
  auto map = AffineMap::get(ctx);
  for (Value v : indices) {
    if (isValidBlockArgument(v))
      return failure();
  }
  return failure();
}

static FailureOr<AffineAccess> buildAffineAccess(const DataLayout &dataLayout,
                                                 Value addr) {
  LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << addr << "\n");
  if (auto ba = dyn_cast<BlockArgument>(addr))
    if (isa<affine::AffineForOp, affine::AffineParallelOp, func::FuncOp>(
            ba.getOwner()->getParentOp()))
      return AffineAccess{.base = addr,
                          .inputs = {},
                          .map =
                              AffineMap::getConstantMap(0, addr.getContext())};
  if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(addr.getDefiningOp())) {
    LLVM_DEBUG(llvm::dbgs() << "gep " << gep << "\n");
    auto base = gep.getBase();
    buildAffineAccess(dataLayout, base);
    auto maybeExpr = getGepAffineExpr(dataLayout, gep);
    LLVM_DEBUG({
      if (maybeExpr)
        llvm::dbgs() << "expr " << *maybeExpr << "\n";
      else
        llvm::dbgs() << "none\n";
    });
  }
  return failure();
}

struct LLVMToAffineAccess
    : public impl::LLVMToAffineAccessPassBase<LLVMToMemref> {
  using Base::Base;
  void runOnOperation() {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    op->walk([&](LLVM::StoreOp store) {
      Value addr = store.getAddr();
      buildAffineAccess(dataLayoutAnalysis.getAtOrAbove(store), addr);
    });
    op->walk([&](LLVM::LoadOp load) {
      Value addr = load.getAddr();
      buildAffineAccess(dataLayoutAnalysis.getAtOrAbove(load), addr);
    });
  }
};

std::unique_ptr<Pass> mlir::createLLVMToAffineAccessPass() {
  return std::make_unique<LLVMToAffineAccess>();
}
