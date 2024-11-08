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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
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
#include "llvm/Support/LogicalResult.h"
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

static std::optional<int64_t> getConstant(Operation *op) {
  if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    return cst.value();
  }
  return {};
}

static std::optional<int64_t> getConstant(Value v) {
  Operation *op = v.getDefiningOp();
  if (op)
    return getConstant(op);
  return {};
}

static LogicalResult
convertLLVMAllocaToMemrefAlloca(LLVM::AllocaOp alloc, RewriterBase &rewriter,
                                const DataLayout &dataLayout) {
  if (!alloc.getRes().hasOneUse())
    return failure();

  auto sizeVal = getConstant(alloc.getArraySize());
  if (!sizeVal)
    return failure();

  Type elType = rewriter.getI8Type();
  int64_t elNum = dataLayout.getTypeSize(alloc.getElemType()) * (*sizeVal);

  auto atAddr =
      dyn_cast<memref::AtAddrOp>(alloc.getRes().use_begin()->getOwner());
  if (!atAddr)
    return failure();

  assert(elType == atAddr.getResult().getType().getElementType());

  SmallVector<int64_t, 1> sizes = {elNum};
  auto memrefType =
      MemRefType::get(sizes, elType, MemRefLayoutAttrInterface{},
                      atAddr.getResult().getType().getMemorySpace());
  auto newAlloca =
      rewriter.create<memref::AllocaOp>(alloc->getLoc(), memrefType);
  rewriter.replaceAllUsesWith(atAddr.getResult(), newAlloca.getResult());
  rewriter.eraseOp(atAddr);
  rewriter.eraseOp(alloc);
  return success();
}

namespace {
struct ConvertLLVMAllocaToMemrefAlloca
    : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;
  const DataLayoutAnalysis &dl;
  ConvertLLVMAllocaToMemrefAlloca(MLIRContext *context,
                                  const DataLayoutAnalysis &dl)
      : OpRewritePattern<LLVM::AllocaOp>(context), dl(dl) {}

  LogicalResult matchAndRewrite(LLVM::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto dataLayout = dl.getAtOrAbove(alloc);
    return convertLLVMAllocaToMemrefAlloca(alloc, rewriter, dataLayout);
  }
};
} // namespace

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
  Attribute addrSpace;
  if (addr.getType().getAddressSpace() == 0)
    addrSpace = nullptr;
  else
    addrSpace = IntegerAttr::get(IntegerType::get(addr.getContext(), 64),
                                 addr.getType().getAddressSpace());
  // TODO we can actually plug in the size of the memref here if `addr` is
  // defined by an llvm.alloca
  auto atAddr = builder.create<memref::AtAddrOp>(
      addr.getLoc(), addr,
      MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                      MemRefLayoutAttrInterface{}, Attribute(addrSpace)));
  return cast<MemRefVal>(atAddr.getResult());
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

static BlockArgument getScopeRemap(affine::AffineScopeOp scope, Value v) {
  for (unsigned i = 0; i < scope->getNumOperands(); i++)
    if (scope->getOperand(i) == v)
      return scope.getRegion().begin()->getArgument(i);
  return nullptr;
}

/// See llvm/Support/Alignment.h
static AffineExpr alignTo(AffineExpr expr, uint64_t a) {
  return (expr + a - 1).floorDiv(a) * a;
}

// TODO To preserve correctness, we need to keep track of values for which
// converting indexing to the index type preserves the semantics, i.e. no
// overflows or underflows or trucation etc and insert a runtime guard against
// that
struct AffineExprBuilder {
  AffineExprBuilder(Operation *user, bool legalizeSymbols)
      : user(user), legalizeSymbols(legalizeSymbols) {}
  Operation *user;

  SmallPtrSet<Value, 4> illegalSymbols;

  DenseMap<Value, unsigned> symToPos;
  DenseMap<Value, unsigned> dimToPos;
  SmallVector<Value> symbolOperands;
  SmallVector<Value> dimOperands;

  // Options
  bool legalizeSymbols;

  SmallVector<Value> symbolsForScope;
  unsigned scopedIllegalSymbols = 0;
  bool scoped = false;

  bool isLegal() {
    return illegalSymbols.size() == 0 ||
           (illegalSymbols.size() == scopedIllegalSymbols && scoped);
  }

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    assert(region->getBlocks().size() == 1);
    SmallVector<AffineExpr> newExprs;
    if (!region->isAncestor(user->getParentRegion()))
      return;
    // An illegal symbol will be legalized either by defining in at the top
    // level in a region, or by remapping it in the scope
    for (auto sym : illegalSymbols) {
      assert(sym.getParentRegion()->isAncestor(region));
      bool isOutsideRegion = sym.getParentRegion()->isProperAncestor(region);
      auto ba = dyn_cast<BlockArgument>(sym);
      bool isTopLevelBlockArg = ba && ba.getOwner()->getParent() == region;
      [[maybe_unused]] bool isTopLevelOp =
          !ba && sym.getParentRegion() == region;
      assert((unsigned)isOutsideRegion + (unsigned)isTopLevelBlockArg +
                 (unsigned)isTopLevelOp ==
             1);
      scopedIllegalSymbols++;
      if (isOutsideRegion || isTopLevelBlockArg)
        symbols.insert(sym);
    }
    if (!region->isProperAncestor(user->getParentRegion()))
      return;
    // We redefine dims to be symbols in this scope
    for (auto dim : dimOperands) {
      if (dim.getParentRegion()->isProperAncestor(region)) {
        symbols.insert(dim);
        symbolsForScope.push_back(dim);
      }
    }
    // TODO we may have a state like this:
    //
    // func.func () {
    //   %sym = ...
    //   region: {
    //     ...
    //   }
    // }
    //
    // and `sym` was mot marked illegal because func.func is an affine scope.
    // Should we rescope it to the new scope?
  }

  AffineExpr rescopeExprImpl(AffineExpr expr, affine::AffineScopeOp scope) {
    auto newExpr = expr;
    for (auto sym : symbolsForScope) {
      unsigned dimPos = getDimPosition(sym);
      assert(dimOperands[dimPos] == sym);
      BlockArgument newSym = getScopeRemap(scope, sym);
      assert(newSym);
      unsigned newSymPos = getSymbolPosition(newSym);
      AffineExpr dimExpr = getAffineDimExpr(dimPos, user->getContext());
      AffineExpr newSymExpr = getAffineDimExpr(newSymPos, user->getContext());
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
    return newExpr;
  }

  void rescopeExpr(affine::AffineScopeOp scope) {
    expr = rescopeExprImpl(expr, scope);
    assert(!scoped);
    scoped = true;
  }

  unsigned getPosition(Value v, SmallVectorImpl<Value> &operands,
                       DenseMap<Value, unsigned> toPos) {
    auto it = toPos.find(v);
    if (it != toPos.end())
      return it->getSecond();
    unsigned newPos = operands.size();
    toPos.insert({v, newPos});
    operands.push_back(v);
    return newPos;
  }

  unsigned getSymbolPosition(Value v) {
    return getPosition(v, symbolOperands, symToPos);
  }
  unsigned getDimPosition(Value v) {
    return getPosition(v, dimOperands, dimToPos);
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
    }
    bool isIndexTy = v.getType().isIndex();
    Value oldV = v;
    if (!isIndexTy)
      v = convertToIndex(v);
    if (affine::isValidSymbol(v)) {
      return getAffineSymbolExpr(getSymbolPosition(v), v.getContext());
    } else if (affine::isValidDim(v)) {
      return getAffineDimExpr(getDimPosition(v), v.getContext());
    }
    if (!isIndexTy) {
      v.getDefiningOp()->erase();
      v = oldV;
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

    for (auto &use : v.getUses()) {
      if (auto affineScope = dyn_cast<affine::AffineScopeOp>(use.getOwner())) {
        if (affineScope->isAncestor(user))
          // TODO should we try to find the inner-most one?
          return getAffineSymbolExpr(
              getSymbolPosition(affineScope.getRegion().front().getArgument(
                  use.getOperandNumber())),
              v.getContext());
      }
    }

    if (legalizeSymbols) {
      illegalSymbols.insert(v);
      return getAffineSymbolExpr(getSymbolPosition(v), context);
    }

    return failure();
  }

  FailureOr<AffineExpr> getExpr(llvm::PointerUnion<IntegerAttr, Value> index) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    if (constIndex) {
      return getAffineConstantExpr(constIndex.getInt(), user->getContext());
    } else {
      return buildExpr(cast<Value>(index));
    }
  }

  AffineExpr expr;
  LogicalResult build(llvm::PointerUnion<IntegerAttr, Value> index) {
    auto mexpr = getExpr(index);
    if (failed(mexpr))
      return failure();
    expr = *mexpr;
    return success();
  }

  struct MapAndOperands {
    AffineMap map;
    SmallVector<Value> operands;
  };
  AffineExpr getExpr() {
    assert(isLegal());
    return expr;
  }
  MapAndOperands getMap() {
    assert(isLegal());
    AffineMap map = AffineMap::get(dimOperands.size(), symbolOperands.size(),
                                   expr, user->getContext());
    auto concat = llvm::concat<Value>(dimOperands, symbolOperands);
    SmallVector<Value> operands =
        SmallVector<Value>(concat.begin(), concat.end());
    affine::canonicalizeMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    return {map, operands};
  }
};

struct AffineAccessBuilder : AffineExprBuilder {
private:
  struct AffineAccess {
    PtrVal base;
    AffineExpr expr;
  };

public:
  AffineAccessBuilder(Operation *accessOp, bool legalizeSymbols)
      : AffineExprBuilder(accessOp, legalizeSymbols) {}

  PtrVal base = nullptr;

  LogicalResult build(const DataLayout &dataLayout, PtrVal addr) {
    auto aa = buildAffineAccess(dataLayout, addr);
    if (failed(aa))
      return failure();
    expr = aa->expr;
    base = aa->base;

    LLVM_DEBUG(llvm::dbgs() << "Built expr: " << expr << "\n");
    return success();
  }

  AffineExprBuilder::MapAndOperands getMap() {
    return AffineExprBuilder::getMap();
  }

  PtrVal getBase() {
    assert(base);
    return base;
  }

  void rescope(affine::AffineScopeOp scope) {
    if (!scope->isAncestor(user))
      return;
    rescopeExpr(scope);
  }

private:
  std::optional<AffineExpr> getGepAffineExpr(const DataLayout &dataLayout,
                                             LLVM::GEPOp gep) {
    // TODO what happens if we get a negative index
    auto indicesRange = gep.getIndices();
    auto indices = SmallVector<LLVM::GEPIndicesAdaptor<ValueRange>::value_type>(
        indicesRange.begin(), indicesRange.end());
    assert(indices.size() > 0);
    Type currentType = gep.getElemType();
    auto expr = getExpr(indices[0]);
    if (failed(expr))
      return std::nullopt;
    AffineExpr offset = (*expr) * dataLayout.getTypeSize(currentType);

    for (auto index : llvm::drop_begin(indices)) {
      bool shouldCancel =
          TypeSwitch<Type, bool>(currentType)
              .Case([&](LLVM::LLVMArrayType arrayType) {
                auto expr = getExpr(index);
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
                    offset = alignTo(offset,
                                     dataLayout.getTypeABIAlignment(body[i]));
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

  FailureOr<AffineAccess> buildAffineAccess(const DataLayout &dataLayout,
                                            PtrVal addr) {
    if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(addr.getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "gep " << gep << "\n");
      auto base = cast<PtrVal>(gep.getBase());

      auto gepExpr = getGepAffineExpr(dataLayout, gep);
      if (!gepExpr)
        return failure();

      auto aa = buildAffineAccess(dataLayout, base);
      if (failed(aa))
        return failure();

      AffineAccess newAA;
      newAA.base = aa->base;
      newAA.expr = aa->expr + *gepExpr;
      LLVM_DEBUG(llvm::dbgs() << "added " << newAA.expr << "\n");
      return newAA;
    } else if (auto addrSpaceCast = dyn_cast_or_null<LLVM::AddrSpaceCastOp>(
                   addr.getDefiningOp())) {
      return buildAffineAccess(dataLayout,
                               cast<PtrVal>(addrSpaceCast.getArg()));
    }

    AffineAccess aa;
    aa.base = addr;
    aa.expr = getAffineConstantExpr(0, addr.getContext());
    LLVM_DEBUG(llvm::dbgs() << "base " << aa.expr << "\n");
    return aa;
  }
};

struct AffineForBuilder {
public:
  AffineForBuilder(scf::ForOp forOp, bool legalizeSymbols)
      : lbBuilder(forOp, legalizeSymbols), ubBuilder(forOp, legalizeSymbols),
        forOp(forOp) {}

  AffineExprBuilder lbBuilder;
  AffineExprBuilder ubBuilder;

  scf::ForOp forOp;
  int64_t step;

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    lbBuilder.collectSymbolsForScope(region, symbols);
    ubBuilder.collectSymbolsForScope(region, symbols);
  }

  SmallPtrSet<Value, 4> getIllegalSymbols() {
    auto set = lbBuilder.illegalSymbols;
    set.insert(ubBuilder.illegalSymbols.begin(),
               ubBuilder.illegalSymbols.end());
    return set;
  }

  LogicalResult build() {
    auto cstStep = getConstant(forOp.getStep());
    if (!cstStep)
      return failure();
    step = *cstStep;

    if (failed(ubBuilder.build(forOp.getUpperBound())) ||
        failed(lbBuilder.build(forOp.getLowerBound())))
      return failure();

    return success();
  }

  AffineExprBuilder::MapAndOperands getUbMap() { return ubBuilder.getMap(); }

  AffineExprBuilder::MapAndOperands getLbMap() { return lbBuilder.getMap(); }

  int64_t getStep() { return step; }

  void rescope(affine::AffineScopeOp scope) {
    if (!scope->isAncestor(forOp))
      return;
    SmallVector<AffineExpr> newExprs;

    lbBuilder.rescopeExpr(scope);
    ubBuilder.rescopeExpr(scope);
  }
};

struct AffineIfBuilder {
public:
  scf::IfOp ifOp;
  bool legalizeSymbols;
  AffineIfBuilder(scf::IfOp ifOp, bool legalizeSymbols)
      : ifOp(ifOp), legalizeSymbols(legalizeSymbols) {}

  struct Constraint {
    arith::CmpIPredicate pred;
    Value lhs;
    Value rhs;
  };

  struct SetAndOperands {
    IntegerSet set;
    SmallVector<Value> operands;
  } sao;

  SmallVector<AffineExprBuilder, 0> builders;

  LogicalResult build() {
    Value cond = ifOp.getCondition();

    SmallVector<Constraint> constraints;
    if (failed(getConstraints(cond, constraints)))
      return failure();

    SmallVector<AffineExpr> exprs;
    SmallVector<bool> eqs;
    unsigned numDims = 0;
    unsigned numSymbols = 0;
    SmallVector<Value> dimOperands;
    SmallVector<Value> symbolOperands;
    for (auto c : constraints) {

      builders.push_back({ifOp, legalizeSymbols});
      AffineExprBuilder &blhs = builders.back();
      if (failed(blhs.build(c.lhs)))
        return failure();
      auto lhs = blhs.getExpr();
      lhs = lhs.shiftDims(blhs.dimOperands.size(), numDims);
      lhs = lhs.shiftSymbols(blhs.symbolOperands.size(), numSymbols);
      numDims += blhs.dimOperands.size();
      numSymbols += blhs.symbolOperands.size();
      dimOperands.append(blhs.dimOperands);
      symbolOperands.append(blhs.symbolOperands);

      builders.push_back({ifOp, legalizeSymbols});
      AffineExprBuilder &brhs = builders.back();
      if (failed(brhs.build(c.rhs)))
        return failure();
      auto rhs = brhs.getExpr();
      rhs = rhs.shiftDims(brhs.dimOperands.size(), numDims);
      rhs = rhs.shiftSymbols(brhs.symbolOperands.size(), numSymbols);
      numDims += brhs.dimOperands.size();
      numSymbols += brhs.symbolOperands.size();
      dimOperands.append(brhs.dimOperands);
      symbolOperands.append(brhs.symbolOperands);

      AffineExpr expr = getAffineConstantExpr(0, ifOp->getContext());
      switch (c.pred) {
      case arith::CmpIPredicate::eq:
        exprs.push_back(rhs - lhs);
        eqs.push_back(true);
        break;
      case arith::CmpIPredicate::ne:
        // TODO not very sure about this, write some tests
        exprs.push_back(rhs - lhs + 1);
        eqs.push_back(false);
        exprs.push_back(lhs - rhs + 1);
        eqs.push_back(false);
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        expr = expr - 1;
        [[fallthrough]];
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        expr = expr + rhs - lhs;
        exprs.push_back(expr);
        eqs.push_back(false);
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        expr = expr - 1;
        [[fallthrough]];
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        expr = expr + lhs - rhs;
        exprs.push_back(expr);
        eqs.push_back(false);
        break;
      }
    }
    sao.set = IntegerSet::get(numDims, numSymbols, exprs, eqs);
    sao.operands = dimOperands;
    sao.operands.append(symbolOperands);
    affine::canonicalizeSetAndOperands(&sao.set, &sao.operands);

    return success();
  }

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    for (auto &builder : builders)
      builder.collectSymbolsForScope(region, symbols);
  }

  SmallPtrSet<Value, 4> getIllegalSymbols() {
    SmallPtrSet<Value, 4> set;
    for (auto &builder : builders)
      set.insert(builder.illegalSymbols.begin(), builder.illegalSymbols.end());
    return set;
  }

  void rescope(affine::AffineScopeOp scope) {
    if (!scope->isAncestor(ifOp))
      return;
    SmallVector<AffineExpr> newExprs;

    for (auto &builder : builders)
      builder.rescopeExpr(scope);
  }

  SetAndOperands getSet() { return sao; }

  LogicalResult getConstraints(Value conjunction,
                               SmallVectorImpl<Constraint> &constraints) {
    Operation *op = conjunction.getDefiningOp();
    if (!op)
      return failure();
    if (isa<LLVM::AndOp, arith::AndIOp>(op)) {
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      if (succeeded(getConstraints(lhs, constraints)) &&
          succeeded(getConstraints(rhs, constraints)))
        return success();
      else
        return failure();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      Constraint c;
      c.pred = cmp.getPredicate();
      c.lhs = cmp.getLhs();
      c.rhs = cmp.getRhs();
      constraints.push_back(c);
      return success();
    }
    return failure();
  }
};

// TODO this works for single-block regions where SSA values are not used across
// blocks but will fail when a value defined in `block` is used in another
// block.
static affine::AffineScopeOp appendToScope(affine::AffineScopeOp oldScope,
                                           ValueRange operands) {
  IRRewriter rewriter(oldScope);
  assert(llvm::all_of(operands, [&](Value a) {
    return llvm::all_of(oldScope->getOperands(),
                        [&](Value b) { return a != b; });
  }));
  SmallVector<Value> newOperands(oldScope->getOperands());
  Block *b = &oldScope.getRegion().front();
  for (Value v : operands) {
    if (llvm::find(newOperands, v) == newOperands.end()) {
      b->addArgument(v.getType(), v.getLoc());
      newOperands.push_back(v);
    }
  }
  auto scope = rewriter.create<affine::AffineScopeOp>(
      oldScope.getLoc(), oldScope->getResultTypes(), newOperands);
  rewriter.inlineRegionBefore(oldScope.getRegion(), scope.getRegion(),
                              scope.getRegion().begin());
  rewriter.replaceOp(oldScope, scope);
  return scope;
}

template <typename T>
SmallVector<Location> getLocs(T values) {
  return llvm::map_to_vector(values, [](Value v) { return v.getLoc(); });
}

static affine::AffineScopeOp insertAffineScope(Block *block,
                                               ValueRange operands) {
  assert(block->getParent()->getBlocks().size() == 1);

  assert(!isa<affine::AffineScopeOp>(block->getParentOp()));
  if (auto scope = dyn_cast<affine::AffineScopeOp>(block->front())) {
    assert(scope->getNextNode() == scope->getBlock()->getTerminator());
    return appendToScope(scope, operands);
  }

  IRRewriter rewriter(block->getParentOp()->getContext());
  rewriter.setInsertionPointToStart(block);
  auto scope = rewriter.create<affine::AffineScopeOp>(
      block->getParentOp()->getLoc(), block->getTerminator()->getOperandTypes(),
      operands);
  Block *innerBlock = rewriter.createBlock(
      &scope.getRegion(), {}, operands.getTypes(), getLocs(operands));
  while (scope->getNextNode() != &block->back())
    rewriter.moveOpBefore(scope->getNextNode(), innerBlock, innerBlock->end());
  rewriter.setInsertionPointToEnd(innerBlock);
  Operation *terminator = block->getTerminator();
  rewriter.create<affine::AffineYieldOp>(terminator->getLoc(),
                                         terminator->getOperands());
  terminator->setOperands(scope->getResults());
  return scope;
}

static constexpr bool useVectorLoadStore = true;

static Operation *createVectorStore(OpBuilder &b, Location loc,
                                    TypedValue<VectorType> v, MemRefVal m,
                                    AffineMap map, ValueRange mapOperands) {
  if (useVectorLoadStore)
    return b.create<affine::AffineVectorStoreOp>(loc, v, m, map, mapOperands);

  SmallVector<Value> newMapOperands(mapOperands);

  VectorType vty = v.getType();
  assert(!vty.isScalable());
  unsigned rank = vty.getRank();
  SmallVector<AffineMap> lbs(rank,
                             AffineMap::getConstantMap(0, b.getContext()));
  SmallVector<int64_t> steps(rank, 1);
  SmallVector<AffineMap> ubs;
  ubs.reserve(rank);
  for (auto size : vty.getShape())
    ubs.push_back(AffineMap::getConstantMap(size, b.getContext()));
  auto par = b.create<affine::AffineParallelOp>(
      loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{}, lbs, ValueRange(),
      ubs, ValueRange(), steps);
  par->setAttr("affine.vector.store", b.getUnitAttr());

  b.setInsertionPointToStart(par.getBody());

  assert(map.getNumResults() == rank);
  SmallVector<AffineExpr> newExprs;
  SmallVector<OpFoldResult> idxs;
  for (unsigned i = 0; i < rank; i++) {
    auto expr = map.getResult(i);
    expr = expr + getAffineDimExpr(map.getNumDims() + i, b.getContext());
    newExprs.push_back(expr);
    newMapOperands.push_back(par.getIVs()[i]);
    idxs.push_back(par.getIVs()[i]);
  }
  AffineMap newMap = AffineMap::get(
      map.getNumDims() + rank, map.getNumSymbols(), newExprs, b.getContext());
  auto el = b.create<vector::ExtractOp>(loc, v, idxs);
  b.create<affine::AffineStoreOp>(loc, el, m, newMap, newMapOperands);
  return par;
}

static Value createVectorLoad(OpBuilder &b, Location loc, VectorType vty,
                              MemRefVal m, AffineMap map,
                              ValueRange mapOperands) {
  if (useVectorLoadStore)
    return b.create<affine::AffineVectorLoadOp>(loc, vty, m, map, mapOperands);

  SmallVector<Value> newMapOperands(mapOperands);

  std::array<arith::AtomicRMWKind, 1> reds = {
      arith::AtomicRMWKind::vector_insert};
  assert(!vty.isScalable());
  unsigned rank = vty.getRank();
  SmallVector<AffineMap> lbs(rank,
                             AffineMap::getConstantMap(0, b.getContext()));
  SmallVector<int64_t> steps(rank, 1);
  SmallVector<AffineMap> ubs;
  ubs.reserve(rank);
  for (auto size : vty.getShape())
    ubs.push_back(AffineMap::getConstantMap(size, b.getContext()));
  auto par = b.create<affine::AffineParallelOp>(
      loc, TypeRange{vty}, reds, lbs, ValueRange(), ubs, ValueRange(), steps);
  par->setAttr("affine.vector.load", b.getUnitAttr());
  b.setInsertionPointToStart(par.getBody());

  assert(map.getNumResults() == rank);
  SmallVector<AffineExpr> newExprs;
  for (unsigned i = 0; i < rank; i++) {
    auto expr = map.getResult(i);
    expr = expr + getAffineDimExpr(map.getNumDims() + i, b.getContext());
    newExprs.push_back(expr);
    newMapOperands.push_back(par.getIVs()[i]);
  }
  AffineMap newMap = AffineMap::get(
      map.getNumDims() + rank, map.getNumSymbols(), newExprs, b.getContext());
  auto load = b.create<affine::AffineLoadOp>(loc, m, newMap, newMapOperands);
  auto bc = b.create<vector::BroadcastOp>(loc, vty, load);
  b.create<affine::AffineYieldOp>(loc, ValueRange{bc});
  b.setInsertionPointAfter(par);
  return par.getResult(0);
}

namespace mlir {
LogicalResult
convertLLVMToAffineAccess(Operation *op,
                          const DataLayoutAnalysis &dataLayoutAnalysis,
                          bool legalizeSymbols) {
  if (!legalizeSymbols && !op->hasTrait<OpTrait::AffineScope>()) {
    LLVM_DEBUG(llvm::errs() << "Must be called with an affine scope root when "
                               "not legelizing symbols\n");
    return failure();
  }

  MLIRContext *context = op->getContext();

  MemrefConverter mc;
  IndexConverter ic;

  // TODO Pretty slow but annoying to implement as we wrap the operation in
  // the callback
  while (true) {
    auto res = op->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      AffineForBuilder forBuilder(forOp, legalizeSymbols);
      if (failed(forBuilder.build()))
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "Converting\n" << forOp << "\n");
      if (legalizeSymbols) {
        SmallPtrSet<Block *, 8> blocksToScope;
        for (auto illegalSym : forBuilder.getIllegalSymbols())
          blocksToScope.insert(illegalSym.getParentBlock());
        for (Block *b : blocksToScope) {
          SmallPtrSet<Value, 6> symbols;
          forBuilder.collectSymbolsForScope(b->getParent(), symbols);
          SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
          auto scope = insertAffineScope(b, symbolsVec);
          forBuilder.rescope(scope);
        }
      }
      IRRewriter rewriter(forOp);
      auto lb = forBuilder.getLbMap();
      auto ub = forBuilder.getUbMap();
      auto affineForOp = rewriter.create<affine::AffineForOp>(
          forOp.getLoc(), ic(lb.operands), lb.map, ic(ub.operands), ub.map,
          forBuilder.getStep(), forOp.getInitArgs());
      if (!affineForOp.getRegion().empty())
        affineForOp.getRegion().front().erase();
      Block *block = forOp.getBody();
      SmallVector<Type> blockArgTypes = {rewriter.getIndexType()};
      auto iterArgTypes = forOp.getInitArgs().getTypes();
      blockArgTypes.insert(blockArgTypes.end(), iterArgTypes.begin(),
                           iterArgTypes.end());
      SmallVector<Location> blockArgLocs =
          getLocs(forOp.getBody()->getArguments());
      auto newBlock = rewriter.createBlock(&affineForOp.getRegion(), {},
                                           blockArgTypes, blockArgLocs);
      SmallVector<Value> newBlockArgs(newBlock->getArguments());
      auto origIVType = forOp.getInductionVar().getType();
      if (origIVType != rewriter.getIndexType()) {
        rewriter.setInsertionPointToStart(newBlock);
        newBlockArgs[0] = rewriter.create<arith::IndexCastOp>(
            newBlockArgs[0].getLoc(), origIVType, newBlockArgs[0]);
      }
      rewriter.inlineBlockBefore(block, newBlock, newBlock->end(),
                                 newBlockArgs);
      rewriter.replaceOp(forOp, affineForOp);
      auto yield = cast<scf::YieldOp>(newBlock->getTerminator());
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yield,
                                                         yield.getOperands());
      return WalkResult::interrupt();
    });
    if (!res.wasInterrupted())
      break;
  }

  while (true) {
    auto res = op->walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
      AffineIfBuilder ifBuilder(ifOp, legalizeSymbols);
      if (failed(ifBuilder.build()))
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "Converting\n" << ifOp << "\n");
      if (legalizeSymbols) {
        SmallPtrSet<Block *, 8> blocksToScope;
        for (auto illegalSym : ifBuilder.getIllegalSymbols())
          blocksToScope.insert(illegalSym.getParentBlock());
        for (Block *b : blocksToScope) {
          SmallPtrSet<Value, 6> symbols;
          ifBuilder.collectSymbolsForScope(b->getParent(), symbols);
          SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
          auto scope = insertAffineScope(b, symbolsVec);
          ifBuilder.rescope(scope);
        }
      }
      IRRewriter rewriter(ifOp);
      auto sao = ifBuilder.getSet();
      auto affineIfOp = rewriter.create<affine::AffineIfOp>(
          ifOp.getLoc(), ifOp.getResultTypes(), sao.set, ic(sao.operands),
          ifOp.elseBlock());
      for (auto [newRegion, oldRegion] :
           llvm::zip(affineIfOp.getRegions(), ifOp.getRegions())) {
        if (!newRegion->empty())
          newRegion->front().erase();
        if (oldRegion->empty())
          continue;
        Block *block = &oldRegion->front();
        auto newBlock = rewriter.createBlock(newRegion);
        rewriter.inlineBlockBefore(block, newBlock, newBlock->end(), {});
        auto yield = cast<scf::YieldOp>(newBlock->getTerminator());
        rewriter.setInsertionPoint(yield);
        rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yield,
                                                           yield.getOperands());
      }
      rewriter.replaceOp(ifOp, affineIfOp);
      return WalkResult::interrupt();
    });
    if (!res.wasInterrupted())
      break;
  }

  SmallVector<std::unique_ptr<AffineAccessBuilder>> accessBuilders;
  auto handleOp = [&](Operation *op, PtrVal addr) {
    LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << op
                            << " for address " << addr << "\n");
    accessBuilders.push_back(
        std::make_unique<AffineAccessBuilder>(op, legalizeSymbols));
    AffineAccessBuilder &aab = *accessBuilders.back();
    auto dl = dataLayoutAnalysis.getAtOrAbove(op);
    auto res = aab.build(dl, addr);
    if (failed(res))
      accessBuilders.pop_back();
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
    SmallPtrSet<Block *, 8> blocksToScope;
    for (auto &aabp : accessBuilders)
      for (auto illegalSym : aabp->illegalSymbols)
        blocksToScope.insert(illegalSym.getParentBlock());
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

    // TODO this looks terribly slow
    for (Block *b : innermostBlocks) {
      SmallPtrSet<Value, 6> symbols;
      for (auto &aabp : accessBuilders)
        aabp->collectSymbolsForScope(b->getParent(), symbols);
      SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
      auto scope = insertAffineScope(b, symbolsVec);
      for (auto &aabp : accessBuilders) {
        aabp->rescope(scope);
      }
    }
  }

  IRMapping mapping;
  for (auto &aabp : accessBuilders) {
    AffineAccessBuilder &aab = *aabp;
    // TODO add a test where some operations are left illegal
    if (!aab.isLegal())
      continue;

    auto mao = aab.getMap();

    auto dl = dataLayoutAnalysis.getAtOrAbove(aab.user);
    if (auto load = dyn_cast<LLVM::LoadOp>(aab.user)) {
      IRRewriter rewriter(load);
      auto vty = VectorType::get({(int64_t)dl.getTypeSize(load.getType())},
                                 rewriter.getI8Type());
      auto vecLoad =
          createVectorLoad(rewriter, load.getLoc(), vty, mc(aab.getBase()),
                           mao.map, ic(mao.operands));
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
    } else if (auto store = dyn_cast<LLVM::StoreOp>(aab.user)) {
      Type ty = store.getValue().getType();
      IRRewriter rewriter(store);
      auto vty =
          VectorType::get({(int64_t)dl.getTypeSize(ty)}, rewriter.getI8Type());
      Value v;
      if (isa<LLVM::LLVMPointerType>(ty)) {
        Type intTy = rewriter.getIntegerType((int64_t)dl.getTypeSize(ty) * 8);
        v = rewriter.create<LLVM::PtrToIntOp>(store.getLoc(), intTy,
                                              store.getValue());
        v = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty, v);
      } else {
        v = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty,
                                             store.getValue());
      }
      Operation *newStore = createVectorStore(
          rewriter, store.getLoc(), cast<TypedValue<VectorType>>(v),
          mc(aab.base), mao.map, ic(mao.operands));
      mapping.map(store.getOperation(), newStore);
    } else {
      llvm_unreachable("");
    }
  }

  IRRewriter rewriter(context);
  for (auto &&[oldOp, newOp] : mapping.getOperationMap()) {
    rewriter.replaceOp(oldOp, newOp);
  }

  RewritePatternSet patterns(context);
  patterns.insert<ConvertLLVMAllocaToMemrefAlloca>(context, dataLayoutAnalysis);
  GreedyRewriteConfig config;
  return applyPatternsAndFoldGreedily(op, std::move(patterns), config);
}
} // namespace mlir

// This should scheduled on individual functions
struct LLVMToAffineAccessPass
    : public impl::LLVMToAffineAccessPassBase<LLVMToAffineAccessPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    if (failed(convertLLVMToAffineAccess(op, dataLayoutAnalysis, true)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createLLVMToAffineAccessPass() {
  return std::make_unique<LLVMToAffineAccessPass>();
}
