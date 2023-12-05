//===- OMPOpt.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO doc

#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <mlir/IR/Diagnostics.h>

namespace fir {
#define GEN_PASS_DEF_OMPOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-omp-opt"

struct CoexecuteToSingle : public mlir::OpRewritePattern<mlir::omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::omp::CoexecuteOp coexecute,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = coexecute->getLoc();
    auto teams = llvm::dyn_cast<mlir::omp::TeamsOp>(coexecute->getParentOp());
    if (!teams) {
      mlir::emitError(loc, "coexecute not nested in teams\n");
      return mlir::failure();
    }
    if (coexecute.getRegion().getBlocks().size() != 1) {
      mlir::emitError(loc, "coexecute with multiple blocks\n");
      return mlir::failure();
    }
    if (teams.getRegion().getBlocks().size() != 1) {
      mlir::emitError(loc, "teams with multiple blocks\n");
      return mlir::failure();
    }
    if (teams.getRegion().getBlocks().front().getOperations().size() != 2) {
      mlir::emitError(loc, "teams with multiple nested ops\n");
      return mlir::failure();
    }
    mlir::Block *coexecuteBlock = &coexecute.getRegion().front();
    rewriter.eraseOp(coexecuteBlock->getTerminator());
    rewriter.inlineBlockBefore(coexecuteBlock, teams);
    rewriter.eraseOp(teams);
    return mlir::success();
  }
};

class OMPOptPass
    : public fir::impl::OMPOptBase<OMPOptPass> {
public:
  void runOnOperation() override;
};

void OMPOptPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  mlir::MLIRContext &context = getContext();
  mlir::RewritePatternSet patterns(&context);
  mlir::GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = false;

  patterns.insert<CoexecuteToSingle>(&context);
  mlir::Operation *op = getOperation();
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    mlir::emitError(op->getLoc(), "error in OpenMP optimizations\n");
    signalPassFailure();
  }
}

/// OpenMP optimizations
std::unique_ptr<mlir::Pass> fir::createOMPOptPass() {
  return std::make_unique<OMPOptPass>();
}
