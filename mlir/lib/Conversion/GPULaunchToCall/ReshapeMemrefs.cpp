#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "reshape-memrefs"

namespace mlir {
#define GEN_PASS_DEF_RESHAPEMEMREFS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct AddExpr {
  AffineExpr divisible;
  AffineExpr remainder;
};

LogicalResult getOpIndexSet(Operation *op,
                            affine::FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  affine::getEnclosingAffineOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

} // namespace

struct ReshapeMemrefsPass
    : public impl::ReshapeMemrefsBase<ReshapeMemrefsPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    op->walk([&](memref::AllocaOp alloca) {
      using namespace mlir::affine;

      SmallVector<MemRefAccess> accesses;

      bool foundAllUses = true;
      for (auto user : alloca.getResult().getUsers()) {
        if (auto load = dyn_cast<AffineLoadOp>(user)) {
          accesses.push_back(MemRefAccess(load));
        } else if (auto store = dyn_cast<AffineStoreOp>(user)) {
          accesses.push_back(MemRefAccess(store));
        } else {
          foundAllUses = false;
          break;
        }
      }

      if (!foundAllUses)
        return;

      llvm::SmallSetVector<int64_t, 16> constantsSet;
      llvm::SmallSetVector<Value, 16> symbols;
      for (auto access : accesses) {
        AffineValueMap valueMap;
        access.getAccessMap(&valueMap);
        AffineMap map = valueMap.getAffineMap();
        for (AffineExpr result : map.getResults()) {
          result.walk([&](AffineExpr expr) {
            if (auto cst = dyn_cast<AffineConstantExpr>(expr))
              constantsSet.insert(cst.getValue());
            if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
              symbols.insert(valueMap.getOperand(valueMap.getNumDims() +
                                                 sym.getPosition()));
          });
        }
      }

      SmallVector<int64_t> constants(constantsSet.begin(), constantsSet.end());
      llvm::sort(constants);

      auto checkShapeCandidate = [&](int64_t cst, unsigned resultId) {
        LLVM_DEBUG(llvm::dbgs() << "Checking shape candidate " << cst << " at "
                                << resultId << "\n");
        bool allValid = true;
        for (auto access : accesses) {
          AffineValueMap accessAvm;
          access.getAccessMap(&accessAvm);
          AffineExpr expr = accessAvm.getResult(resultId);
          LLVM_DEBUG(llvm::dbgs() << "For access " << *access.opInst
                                  << " with expr " << expr << "\n");
          auto mod = expr % cst;
          auto floor = expr.floorDiv(cst);
          LLVM_DEBUG(llvm::dbgs() << "Mod: " << mod << "\n");
          LLVM_DEBUG(llvm::dbgs() << "Floor: " << floor << "\n");

          auto res = mod.walk([&](AffineExpr expr) {
            AffineBinaryOpExpr binexpr = dyn_cast<AffineBinaryOpExpr>(expr);
            if (!binexpr)
              return WalkResult::advance();
            if (binexpr.getKind() != AffineExprKind::Mod)
              return WalkResult::advance();
            if (binexpr.getRHS() != getAffineConstantExpr(cst, ctx))
              return WalkResult::advance();
            auto lhs = binexpr.getLHS();
            auto lhsMap = AffineMap::get(accessAvm.getNumDims(),
                                         accessAvm.getNumSymbols(), lhs);
            AffineValueMap lhsAvm(lhsMap, accessAvm.getOperands());
            lhsAvm.composeSimplifyAndCanonicalize();
            LLVM_DEBUG(llvm::dbgs()
                       << "Nested mod: " << lhsAvm.getAffineMap() << "\n");
            affine::FlatAffineValueConstraints domain;
            if (failed(getOpIndexSet(access.opInst, &domain)))
              return WalkResult::interrupt();
            if (failed(domain.composeMap(&lhsAvm)))
              return WalkResult::interrupt();
            LLVM_DEBUG(llvm::dbgs() << "Composed domain: ");
            LLVM_DEBUG(domain.dump());
            domain.setDimSymbolSeparation(domain.getNumDimAndSymbolVars() - 1);
            domain.simplify();
            SmallVector<Value, 4> vars;
            domain.getValues(domain.getNumDimVars(),
                             domain.getNumDimAndSymbolVars(), &vars);
            for (Value var : vars)
              if ((affine::isAffineInductionVar(var)))
                domain.projectOut(var);
            domain.constantFoldVarRange(
                /*pos=*/1,
                /*num=*/domain.getNumDimAndSymbolVars() - 1);
            domain.removeTrivialRedundancy();
            auto bounds = domain.getLowerAndUpperBound(
                0, 0, 1, domain.getNumDimVars(), {}, ctx);
            auto lbExpr = simplifyAffineExpr(bounds.first.getResult(0),
                                             1 + accessAvm.getNumDims(),
                                             accessAvm.getNumSymbols());
            auto ubExpr = simplifyAffineExpr(bounds.second.getResult(0),
                                             1 + accessAvm.getNumDims(),
                                             accessAvm.getNumSymbols());
            LLVM_DEBUG(llvm::dbgs() << "LB: " << lbExpr << "\n");
            LLVM_DEBUG(llvm::dbgs() << "UB: " << ubExpr << "\n");
            auto cLb = dyn_cast<AffineConstantExpr>(lbExpr);
            auto cUb = dyn_cast<AffineConstantExpr>(ubExpr);
            if (!cLb || !cUb)
              return WalkResult::interrupt();
            auto lb = cLb.getValue();
            auto ub = cUb.getValue();

            if (lb >= 0 && lb <= cst && ub >= 0 && ub <= cst)
              return WalkResult::advance();
            return WalkResult::interrupt();
          });
          bool isValid = !res.wasInterrupted();

          if (!isValid) {
            allValid = false;
            break;
          }
        }

        if (!allValid)
          return;

        LLVM_DEBUG(llvm::dbgs()
                   << "Found valid shape candidate" << cst << "\n");
      };

      bool changed;
      do {
        changed = false;
        for (auto cst : constants) {
          if (cst == 1 || cst == 0)
            continue;
          checkShapeCandidate(cst, 0);
        }
      } while (changed);
    });
  }
};

std::unique_ptr<Pass> mlir::createReshapeMemrefsPass() {
  return std::make_unique<ReshapeMemrefsPass>();
}
