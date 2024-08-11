#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
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

      std::function<SmallVector<AffineExpr>(int64_t, AffineExpr, AffineExpr)>
          getNonMultipleAddOperand;
      getNonMultipleAddOperand =
          [&](int64_t cst, AffineExpr expr,
              AffineExpr operand) -> SmallVector<AffineExpr> {
        if (expr.isMultipleOf(cst))
          return {operand};
        if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
          if (binExpr.getKind() == AffineExprKind::Add) {
            auto lhs = binExpr.getLHS();
            auto rhs = binExpr.getRHS();
            auto nmaos = getNonMultipleAddOperand(cst, lhs, operand + rhs);
            nmaos.append(getNonMultipleAddOperand(cst, rhs, operand + lhs));
            return nmaos;
          }
        }
        return {};
      };

      auto checkShapeCandidate = [&](int64_t cst, unsigned resultId) {
        LLVM_DEBUG(llvm::dbgs() << "Checking shape candidate " << cst << " at "
                                << resultId << "\n");
        for (auto access : accesses) {
          AffineValueMap valueMap;
          access.getAccessMap(&valueMap);
          AffineExpr expr = valueMap.getResult(resultId);
          LLVM_DEBUG(llvm::dbgs() << "for access " << *access.opInst
                                  << " with expr " << expr << "\n");
          auto nmaos = getNonMultipleAddOperand(cst, expr,
                                                getAffineConstantExpr(0, ctx));
          for (auto nmao : nmaos)
            LLVM_DEBUG(llvm::dbgs() << "nmao " << nmao << "\n");
        }
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
