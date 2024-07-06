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
#include "mlir/IR/BuiltinTypes.h"
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
using MemRefVal = MemrefValue;

struct AffineAccess {
  MemRefVal base;
  AffineExpr expr;
};

struct DimOrSym {
  Value val;
  bool isDim;
};

struct AffineAccessBuilder;
static FailureOr<AffineAccess>
buildAffineAccess(const DataLayout &dataLayout, PtrVal addr,
                  AffineAccessBuilder &valueToPos);

static Value convertToIndex(Value v) {
  OpBuilder builder(v.getContext());
  if (v.getType() == builder.getIndexType())
    return v;
  if (auto ba = dyn_cast<BlockArgument>(v))
    builder.setInsertionPointToStart(ba.getOwner());
  else
    builder.setInsertionPointAfter(v.getDefiningOp());
  return builder
      .create<arith::IndexCastOp>(v.getLoc(), builder.getIndexType(), v)
      .getResult();
}

static MemRefVal convertToMemref(PtrVal addr) {
  OpBuilder builder(addr.getContext());
  if (auto ba = dyn_cast<BlockArgument>(addr))
    builder.setInsertionPointToStart(ba.getOwner());
  else
    builder.setInsertionPointAfter(addr.getDefiningOp());
  return cast<MemRefVal>(
      builder.create<memref::AtAddrOp>(addr.getLoc(), addr).getResult());
}

struct AffineAccessBuilder {
  DenseMap<Value, unsigned> valueToPos;
  SmallVector<Value> symbolOperands;
  SmallVector<Value> dimOperands;
  unsigned numDims = 0;
  unsigned numSymbols = 0;

  AffineMap map;
  SmallVector<Value> operands;
  MemRefVal base;

  LogicalResult build(const DataLayout &dataLayout, PtrVal addr) {
    auto aa = buildAffineAccess(dataLayout, addr, *this);
    if (failed(aa))
      return failure();
    base = aa->base;
    map = AffineMap::get(numDims, numSymbols, aa->expr);
    operands =
        llvm::map_to_vector(llvm::concat<Value>(dimOperands, symbolOperands),
                            [&](Value v) { return convertToIndex(v); });
    affine::canonicalizeMapAndOperands(&map, &operands);
    LLVM_DEBUG(llvm::dbgs() << "Built map: " << map << "\n");
    return success();
  }

  unsigned get(Value v, unsigned &num, SmallVectorImpl<Value> &operands) {
    // TODO follow this through casts, exts, etc
    auto it = valueToPos.find(v);
    if (it != valueToPos.end())
      return it->getSecond();
    unsigned newPos = valueToPos.size();
    valueToPos.insert({v, newPos});
    operands.push_back(v);
    num++;
    return newPos;
  }

  template <typename... Ts>
  inline FailureOr<AffineExpr> buildPassthrough(Operation *op) {
    if (isa<Ts...>(op)) {
      assert(op->getNumOperands() == 1);
      return buildExpr(op->getOperand(0));
    }
    return failure();
  }

  template <typename... Ts>
  inline FailureOr<AffineExpr>
  buildBinOpExpr(Operation *op,
                 AffineExpr (AffineExpr::*handler)(AffineExpr) const) {
    if (isa<Ts...>(op)) {
      assert(op->getNumOperands() == 2);
      auto lhs = buildExpr(op->getOperand(0));
      auto rhs = buildExpr(op->getOperand(1));
      if (failed(lhs) || failed(rhs))
        return failure();
      return ((*lhs).*handler)(*rhs);
    }
    return failure();
  }

  FailureOr<AffineExpr> buildExpr(Value v) {
    auto context = v.getContext();
    Operation *op = v.getDefiningOp();
    if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
      return getAffineConstantExpr(cst.value(), context);
    } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
      return getAffineConstantExpr(cst.value(), context);
    } else if (affine::isValidDim(v)) {
      return getAffineDimExpr(get(v, numDims, dimOperands), v.getContext());
    } else if (affine::isValidSymbol(v)) {
      return getAffineSymbolExpr(get(v, numSymbols, symbolOperands),
                                 v.getContext());
    }

    if (op) {
      // clang-format off
#define RIS(X) do { auto res = X; if (succeeded(res)) return *res; } while (0)
      RIS((buildBinOpExpr<LLVM::AddOp, arith::AddIOp>(
               op, &AffineExpr::operator+)));
      RIS((buildBinOpExpr<LLVM::SubOp, arith::SubIOp>(
               op, &AffineExpr::operator-)));
      RIS((buildBinOpExpr<LLVM::URemOp, arith::RemSIOp, LLVM::SRemOp, arith::RemUIOp>(
               op, &AffineExpr::operator%)));
      // TODO need to check that we dont end up with dim * dim or other invalid
      // expression
      RIS((buildBinOpExpr<LLVM::MulOp, arith::MulIOp>(
               op, &AffineExpr::operator*)));
      RIS((buildBinOpExpr<LLVM::UDivOp, LLVM::SDivOp, arith::DivUIOp, arith::DivSIOp>(
               op, &AffineExpr::floorDiv)));
      RIS((buildPassthrough<
           LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp,
           arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
           arith::IndexCastOp, arith::IndexCastUIOp>(op)));
#undef RIS
      // clang-format on
    } else {
      auto ba = dyn_cast<BlockArgument>(v);
      assert(ba);
      // It is a block argument invalid for either dim or sym - we will scope it
      // later
      // TODO I think we may grab an affine op reduction block arg - we should
      // handle these separately
      return getAffineSymbolExpr(get(v, numSymbols, symbolOperands), context);
    }
    return failure();
  }

  FailureOr<AffineExpr> getExpr(llvm::PointerUnion<IntegerAttr, Value> index,
                                MLIRContext *context) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    if (constIndex) {
      return getAffineConstantExpr(constIndex.getInt(), context);
    } else {
      return buildExpr(cast<Value>(index));
    }
  }
};

/// See llvm/Support/Alignment.h
static AffineExpr alignTo(AffineExpr expr, uint64_t a) {
  return (expr + a - 1).floorDiv(a) * a;
}

static std::optional<AffineExpr>
getGepAffineExpr(const DataLayout &dataLayout, LLVM::GEPOp gep,
                 AffineAccessBuilder &valueToPos) {
  // TODO what happens if we get a negative index
  auto context = gep.getContext();

  Type currentType = gep.getElemType();
  auto expr = valueToPos.getExpr(gep.getIndices()[0], context);
  if (failed(expr))
    return std::nullopt;
  AffineExpr offset = (*expr) * dataLayout.getTypeSize(currentType);

  for (auto &&[_i, _index] :
       llvm::drop_begin(llvm::enumerate(gep.getIndices()))) {
    auto index = _index;
    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVM::LLVMArrayType arrayType) {
              auto expr = valueToPos.getExpr(index, context);
              if (failed(expr))
                return true;
              offset = offset + (*expr) * dataLayout.getTypeSize(
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
    rewriter.create<affine::AffineYieldOp>(terminator->getLoc(),
                                           terminator->getOperands());
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

static FailureOr<AffineAccess>
buildAffineAccess(const DataLayout &dataLayout, PtrVal addr,
                  AffineAccessBuilder &valueToPos) {
  if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(addr.getDefiningOp())) {
    LLVM_DEBUG(llvm::dbgs() << "gep " << gep << "\n");
    auto base = cast<PtrVal>(gep.getBase());

    auto gepExpr = getGepAffineExpr(dataLayout, gep, valueToPos);
    if (!gepExpr)
      return failure();

    auto aa = buildAffineAccess(dataLayout, base, valueToPos);
    if (failed(aa))
      return failure();

    AffineAccess newAA;
    newAA.base = aa->base;
    newAA.expr = aa->expr + *gepExpr;
    LLVM_DEBUG(llvm::dbgs() << "added " << newAA.expr << "\n");
    return newAA;
  }

  AffineAccess aa;
  aa.base = convertToMemref(addr);
  aa.expr = getAffineConstantExpr(0, addr.getContext());
  LLVM_DEBUG(llvm::dbgs() << "base " << aa.expr << "\n");
  return aa;
}

struct LLVMToAffineAccessPass
    : public impl::LLVMToAffineAccessPassBase<LLVMToAffineAccessPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    // TODO getting the layout analysis for every op seems bad :)
    op->walk([&](LLVM::StoreOp store) {
      PtrVal addr = store.getAddr();
      LLVM_DEBUG(llvm::dbgs()
                 << "Building affine access for " << store << "\n");
      AffineAccessBuilder aab;
      auto dl = dataLayoutAnalysis.getAtOrAbove(store);
      auto res = aab.build(dl, addr);
      if (failed(res))
        return;

      Type ty = store.getValue().getType();
      IRRewriter builder(store);
      auto vty =
          VectorType::get({(int64_t)dl.getTypeSize(ty)}, builder.getI8Type());
      auto bitcast = builder.create<LLVM::BitcastOp>(store.getLoc(), vty,
                                                     store.getValue());
      builder.replaceOpWithNewOp<affine::AffineVectorStoreOp>(
          store, bitcast, aab.base, aab.map, aab.operands);
    });
    op->walk([&](LLVM::LoadOp load) {
      PtrVal addr = load.getAddr();
      LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << load << "\n");
      AffineAccessBuilder aab;
      auto dl = dataLayoutAnalysis.getAtOrAbove(load);
      auto res = aab.build(dl, addr);
      if (failed(res))
        return;

      IRRewriter builder(load);
      auto vty = VectorType::get({(int64_t)dl.getTypeSize(load.getType())},
                                 builder.getI8Type());
      auto vecLoad = builder.create<affine::AffineVectorLoadOp>(
          load.getLoc(), vty, aab.base, aab.map, aab.operands);
      builder.replaceOpWithNewOp<LLVM::BitcastOp>(load, load.getType(),
                                                  vecLoad);
    });
  }
};

std::unique_ptr<Pass> mlir::createLLVMToAffineAccessPass() {
  return std::make_unique<LLVMToAffineAccessPass>();
}
