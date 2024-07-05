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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
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
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>

#define DEBUG_TYPE "llvm-to-affine-access"

namespace mlir {
#define GEN_PASS_DEF_LLVMTOAFFINEACCESSPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

using PtrVal = TypedValue<LLVM::LLVMPointerType>;
using MemrefVal = TypedValue<MemRefType>;

struct AffineAccess {
  MemrefVal base;
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

  LLVM_DEBUG(llvm::dbgs() << "offset " << offset << "\n");

  return offset;
}

// TODO collect scopes before instantiating them and only instantiate the
// innermost ones
static BlockArgument scopeAddr(PtrVal addr) {
  IRRewriter rewriter(addr.getContext());
  if (auto ba = dyn_cast<BlockArgument>(addr)) {
    Block *block = ba.getOwner();
    SmallVector<Location> locs = llvm::map_to_vector(
        block->getArguments(), [](BlockArgument a) { return a.getLoc(); });
    Block *newBlock =
        rewriter.createBlock(block, block->getArgumentTypes(), locs);
    auto scope = rewriter.create<affine::AffineScopeOp>(
        ba.getLoc(), block->getTerminator()->getOperandTypes(), ValueRange(ba));
    Block *innerBlock = rewriter.createBlock(
        &scope.getRegion(), {}, TypeRange(ba.getType()), {ba.getLoc()});
    rewriter.replaceAllUsesWith(block, newBlock);
    rewriter.inlineBlockBefore(block, innerBlock, innerBlock->begin(),
                               innerBlock->getArguments());
    rewriter.setInsertionPointToEnd(innerBlock);
    Operation *terminator = innerBlock->getTerminator();
    auto yieldOp = rewriter.create<affine::AffineYieldOp>(
        terminator->getLoc(), terminator->getOperands());
    IRMapping mapping;
    mapping.map(ValueRange(terminator->getOperands()),
                ValueRange(scope->getOpResults()));
    rewriter.setInsertionPointAfter(scope);
    rewriter.clone(*terminator, mapping);
    rewriter.eraseOp(terminator);
    return innerBlock->getArgument(0);
  }
  return nullptr;
}

static Value convertToMemref(PtrVal addr) {
  OpBuilder builder(addr.getContext());
  if (auto ba = dyn_cast<BlockArgument>(addr))
    builder.setInsertionPointToStart(ba.getOwner());
  else
    builder.setInsertionPointAfter(addr.getDefiningOp());
  return builder.create<memref::AtAddrOp>(addr.getLoc(), addr);
}

static FailureOr<AffineAccess> buildAffineAccess(const DataLayout &dataLayout,
                                                 PtrVal addr) {
  if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(addr.getDefiningOp())) {
    LLVM_DEBUG(llvm::dbgs() << "gep " << gep << "\n");
    auto base = cast<PtrVal>(gep.getBase());
    auto expr = getGepAffineExpr(dataLayout, gep);
    if (!expr)
      return failure();

    auto aa = buildAffineAccess(dataLayout, base);
    if (failed(aa))
      return failure();

    AffineMap map = AffineMap::get(*expr);

    AffineAccess newAA;
    newAA.inputs = aa->inputs;
    newAA.inputs.insert(newAA.inputs.begin(), gep->operand_begin(),
                        gep->operand_end());
    newAA.base = aa->base;
    newAA.map = aa->map.compose(map);

    LLVM_DEBUG(llvm::dbgs() << "composed " << newAA.map << "\n");
  }

  AffineAccess aa;
  aa.base = convertToMemref(addr);
  aa.inputs = {};
  aa.map = AffineMap::getConstantMap(0, addr.getContext());

  return aa;
}

struct LLVMToAffineAccessPass
    : public impl::LLVMToAffineAccessPassBase<LLVMToAffineAccessPass> {
  using Base::Base;
  void runOnOperation() {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    // TODO getting the layout analysis for every op seems bad :)
    op->walk([&](LLVM::StoreOp store) {
      PtrVal addr = store.getAddr();
      LLVM_DEBUG(llvm::dbgs()
                 << "Building affine access for " << store << "\n");
      buildAffineAccess(dataLayoutAnalysis.getAtOrAbove(store), addr);
    });
    op->walk([&](LLVM::LoadOp load) {
      PtrVal addr = load.getAddr();
      LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << load << "\n");
      buildAffineAccess(dataLayoutAnalysis.getAtOrAbove(load), addr);
    });
  }
};

std::unique_ptr<Pass> mlir::createLLVMToAffineAccessPass() {
  return std::make_unique<LLVMToAffineAccessPass>();
}
