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
#include "llvm/ADT/SmallPtrSet.h"
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
  PtrVal base;
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

template <typename From, typename To, auto F>
struct ConverterBase {
  DenseMap<From, To> map;
  To operator()(From p) {
    auto it = map.find(p);
    if (it != map.end())
      return it->getSecond();
    auto converted = F(p);
    map.insert({p, converted});
    return converted;
  }
  SmallVector<To> operator()(ValueRange range) {
    return llvm::map_to_vector(range, [&](From v) { return (*this)(v); });
  }
};

using MemrefConverter = ConverterBase<PtrVal, MemRefVal, convertToMemref>;
using IndexConverter = ConverterBase<Value, Value, convertToIndex>;

BlockArgument getScopeRemap(affine::AffineScopeOp scope, Value v) {
  for (unsigned i = 0; i < scope->getNumOperands(); i++)
    if (scope->getOperand(i) == v)
      return scope.getRegion().begin()->getArgument(i);
  return nullptr;
}

// TODO To preserve correctness, we need to keep track of values for which
// converting indexing to the index type preserves the semantics, i.e. no
// overflows or underflows or trucation etc and insert a runtime guard against
// that
struct AffineAccessBuilder {
  Operation *accessOp;
  bool legalizeSymbols;
  AffineAccessBuilder(Operation *accessOp, bool legalizeSymbols = false)
      : accessOp(accessOp), legalizeSymbols(legalizeSymbols) {}

  DenseMap<Value, unsigned> valueToPos;
  SmallVector<Value> symbolOperands;
  SmallVector<Value> dimOperands;
  SmallPtrSet<Value, 4> illegalSymbols;

  AffineMap map;
  SmallVector<Value> operands;
  PtrVal base = nullptr;

  LogicalResult build(const DataLayout &dataLayout, PtrVal addr) {
    auto aa = buildAffineAccess(dataLayout, addr, *this);
    if (failed(aa))
      return failure();
    base = aa->base;
    map = AffineMap::get(dimOperands.size(), symbolOperands.size(), aa->expr);
    auto concat = llvm::concat<Value>(dimOperands, symbolOperands);
    operands = SmallVector<Value>(concat.begin(), concat.end());
    affine::canonicalizeMapAndOperands(&map, &operands);
    LLVM_DEBUG(llvm::dbgs() << "Built map: " << map << "\n");
    return success();
  }

  SmallVector<Value> symbolsForScope;
  bool legalized = false;

  bool isLegal() { return illegalSymbols.size() == 0 || legalized; }

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    assert(region->getBlocks().size() == 1);
    SmallVector<AffineExpr> newExprs;
    if (!region->isAncestor(accessOp->getParentRegion()))
      return;
    // An illegal symbol will be legalized either by defining in at the top
    // level in a region, or by remapping it in the scope
    for (auto sym : illegalSymbols) {
      assert(region->isAncestor(sym.getParentRegion()));
      if (sym.getParentRegion()->isProperAncestor(region)) {
        symbols.insert(sym);
      } else if (auto ba = dyn_cast<BlockArgument>(sym)) {
        if (ba.getOwner()->getParent() == region) {
          symbols.insert(sym);
        }
      }
    }
    if (!region->isProperAncestor(accessOp->getParentRegion()))
      return;
    // We redefine dims to be symbols in this scope
    for (auto dim : dimOperands) {
      if (dim.getParentRegion()->isProperAncestor(region)) {
        symbols.insert(dim);
        symbolsForScope.push_back(dim);
      }
    }
  }

  void rescope(affine::AffineScopeOp scope) {
    SmallVector<AffineExpr> newExprs;
    for (auto &expr : map.getResults()) {
      auto newExpr = expr;
      for (auto sym : symbolsForScope) {
        unsigned dimPos = getPosition(sym, dimOperands);
        assert(dimOperands[dimPos] == sym);
        BlockArgument newSym = getScopeRemap(scope, sym);
        assert(newSym);
        unsigned newSymPos = getPosition(newSym, symbolOperands);
        AffineExpr dimExpr = getAffineDimExpr(dimPos, accessOp->getContext());
        AffineExpr newSymExpr =
            getAffineDimExpr(newSymPos, accessOp->getContext());
        newExpr = newExpr.replace(dimExpr, newSymExpr);
      }
      for (auto sym : illegalSymbols) {
        if (sym.getParentRegion() == &scope.getRegion())
          continue;
        BlockArgument newSym = getScopeRemap(scope, sym);
        assert(newSym);
        auto it = llvm::find(symbolOperands, sym);
        assert(it != symbolOperands.end());
        *it = newSym;
      }
      newExprs.push_back(newExpr);
    }
    map = AffineMap::get(dimOperands.size(), symbolOperands.size(), newExprs,
                         accessOp->getContext());
    auto concat = llvm::concat<Value>(dimOperands, symbolOperands);
    operands = SmallVector<Value>(concat.begin(), concat.end());
    affine::canonicalizeMapAndOperands(&map, &operands);
    legalized = true;
  }

  unsigned getPosition(Value v, SmallVectorImpl<Value> &operands) {
    // TODO follow this through casts, exts, etc
    auto it = valueToPos.find(v);
    if (it != valueToPos.end())
      return it->getSecond();
    unsigned newPos = operands.size();
    valueToPos.insert({v, newPos});

    operands.push_back(v);
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
    } else if (affine::isValidSymbol(v)) {
      return getAffineSymbolExpr(getPosition(v, symbolOperands),
                                 v.getContext());
    } else if (affine::isValidDim(v)) {
      return getAffineDimExpr(getPosition(v, dimOperands), v.getContext());
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
    }

    // TODO We may find an affine op reduction block arg - we may be able to
    // handle them

    if (legalizeSymbols) {
      illegalSymbols.insert(v);
      return getAffineSymbolExpr(getPosition(v, symbolOperands), context);
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

  auto indicesRange = gep.getIndices();
  auto indices = SmallVector<LLVM::GEPIndicesAdaptor<ValueRange>::value_type>(
      indicesRange.begin(), indicesRange.end());
  assert(indices.size() > 0);
  Type currentType = gep.getElemType();
  auto expr = valueToPos.getExpr(indices[0], context);
  if (failed(expr))
    return std::nullopt;
  AffineExpr offset = (*expr) * dataLayout.getTypeSize(currentType);

  for (auto index : llvm::drop_begin(indices)) {
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

// TODO this works for single-block regions where SSA values are not used across
// blocks but will fail when a value defined in `block` is used in another
// block.
static affine::AffineScopeOp insertAffineScope(Block *block,
                                               ValueRange operands) {
  assert(block->getParent()->getBlocks().size() == 1);
  IRRewriter rewriter(block->getParentOp()->getContext());
  rewriter.setInsertionPointToStart(block);
  auto scope = rewriter.create<affine::AffineScopeOp>(
      block->getParentOp()->getLoc(), block->getTerminator()->getOperandTypes(),
      operands);
  Block *innerBlock = rewriter.createBlock(
      &scope.getRegion(), {}, operands.getTypes(),
      llvm::map_to_vector(operands, [](Value v) { return v.getLoc(); }));
  while (scope->getNextNode() != &block->back())
    rewriter.moveOpBefore(scope->getNextNode(), innerBlock, innerBlock->end());
  rewriter.setInsertionPointToEnd(innerBlock);
  Operation *terminator = block->getTerminator();
  rewriter.create<affine::AffineYieldOp>(terminator->getLoc(),
                                         terminator->getOperands());
  terminator->setOperands(scope->getResults());
  return scope;
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
  aa.base = addr;
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

    MemrefConverter mc;
    IndexConverter ic;

    SmallVector<std::unique_ptr<AffineAccessBuilder>> aabs;
    bool legalizeSymbols = true;

    auto handleOp = [&](Operation *op, PtrVal addr) {
      LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << op
                              << " for address " << addr << "\n");
      aabs.push_back(
          std::make_unique<AffineAccessBuilder>(op, legalizeSymbols));
      AffineAccessBuilder &aab = *aabs.back();
      auto dl = dataLayoutAnalysis.getAtOrAbove(op);
      auto res = aab.build(dl, addr);
      if (failed(res))
        aabs.pop_back();
    };

    op->walk([&](LLVM::StoreOp store) {
      PtrVal addr = store.getAddr();
      handleOp(store, addr);
    });
    op->walk([&](LLVM::LoadOp load) {
      PtrVal addr = load.getAddr();
      handleOp(load, addr);
    });
    // TODO should also gather other mem operations such as memory intrinsics
    // TODO should we shrink the scope to where no other memory operations
    // exist?

    if (legalizeSymbols) {
      SmallVector<Block *> blocksToScope;
      for (auto &aabp : aabs)
        for (auto illegalSym : aabp->illegalSymbols)
          blocksToScope.push_back(illegalSym.getParentBlock());
      SmallPtrSet<Block *, 8> innermostBlocks;
      for (Block *b : blocksToScope) {
        SmallVector<Block *> toRemove;
        bool isInnermost = true;
        for (Block *existing : innermostBlocks) {
          if (existing->getParent()->isProperAncestor(b->getParent()))
            toRemove.push_back(existing);
          if (b->getParent()->isAncestor(existing->getParent()))
            isInnermost = false;
        }
        for (Block *r : toRemove)
          innermostBlocks.erase(r);
        if (isInnermost)
          innermostBlocks.insert(b);
      }

      for (Block *b : innermostBlocks) {
        SmallPtrSet<Value, 6> symbols;
        for (auto &aabp : aabs)
          aabp->collectSymbolsForScope(b->getParent(), symbols);
        SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
        auto scope = insertAffineScope(b, symbolsVec);
        for (auto &aabp : aabs) {
          aabp->rescope(scope);
        }
      }
    }

    IRMapping mapping;
    for (auto &aabp : aabs) {
      AffineAccessBuilder &aab = *aabp;
      // TODO add a test where some operations are left illegal
      if (!aab.isLegal())
        continue;

      auto dl = dataLayoutAnalysis.getAtOrAbove(aab.accessOp);
      if (auto load = dyn_cast<LLVM::LoadOp>(aab.accessOp)) {
        IRRewriter rewriter(load);
        auto vty = VectorType::get({(int64_t)dl.getTypeSize(load.getType())},
                                   rewriter.getI8Type());
        auto vecLoad = rewriter.create<affine::AffineVectorLoadOp>(
            load.getLoc(), vty, mc(aab.base), aab.map, ic(aab.operands));
        Operation *newLoad;
        if (isa<LLVM::LLVMPointerType>(load.getType())) {
          Type intTy = rewriter.getIntegerType(
              (int64_t)dl.getTypeSize(load.getType()) * 8);
          auto cast =
              rewriter.create<LLVM::BitcastOp>(load.getLoc(), intTy, vecLoad);
          newLoad = rewriter.create<LLVM::IntToPtrOp>(load.getLoc(),
                                                      load.getType(), cast);
        } else {
          newLoad = rewriter.create<LLVM::BitcastOp>(load.getLoc(),
                                                     load.getType(), vecLoad);
        }
        mapping.map(load, newLoad);
      } else if (auto store = dyn_cast<LLVM::StoreOp>(aab.accessOp)) {
        Type ty = store.getValue().getType();
        IRRewriter rewriter(store);
        auto vty = VectorType::get({(int64_t)dl.getTypeSize(ty)},
                                   rewriter.getI8Type());
        Value cast;
        if (isa<LLVM::LLVMPointerType>(ty)) {
          Type intTy = rewriter.getIntegerType((int64_t)dl.getTypeSize(ty) * 8);
          cast = rewriter.create<LLVM::PtrToIntOp>(store.getLoc(), intTy,
                                                   store.getValue());
          cast = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty, cast);
        } else {
          cast = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty,
                                                  store.getValue());
        }
        Operation *newStore = rewriter.create<affine::AffineVectorStoreOp>(
            store.getLoc(), cast, mc(aab.base), aab.map, ic(aab.operands));
        mapping.map(store.getOperation(), newStore);
      } else {
        llvm_unreachable("");
      }
    }

    IRRewriter rewriter(op->getContext());
    for (auto &&[oldOp, newOp] : mapping.getOperationMap()) {
      rewriter.replaceOp(oldOp, newOp);
    }
  }
};

std::unique_ptr<Pass> mlir::createLLVMToAffineAccessPass() {
  return std::make_unique<LLVMToAffineAccessPass>();
}
