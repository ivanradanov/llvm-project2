//===- OMPOpt.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO doc

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace fir {
#define GEN_PASS_DEF_OMPOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-omp-opt"
#define TAG ("[" DEBUG_TYPE "] ")

/// This is the single source of truth about whether we should parallelize an
/// operation nested in an omp.execute region.
static bool shouldParallelize(mlir::Operation *op) {
  // TODO Currently we cannot parallelize operations with results that have uses
  // - we need additional handling in the fission for that i.e. a way to access
  // that result outside the
  if (llvm::any_of(op->getResults(),
                   [](mlir::OpResult v) -> bool { return !v.use_empty(); }))
    return false;
  // We will parallelize unordered loops - these come from array syntax
  if (auto loop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
    auto unordered = loop.getUnordered();
    if (!unordered)
      return false;
    return *unordered;
  }
  if (auto callOp = mlir::dyn_cast<fir::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    auto *func = op->getParentOfType<mlir::ModuleOp>().lookupSymbol(*callee);
    // TODO need to insert a check here whether it is a call we can actually
    // parallelize currently
    if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
      return true;
    return false;
  }
  // We cannot parallise anything else
  return false;
}

struct FissionCoexecute
    : public mlir::OpRewritePattern<mlir::omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::omp::CoexecuteOp coexecute,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = coexecute->getLoc();
    auto teams = mlir::dyn_cast<mlir::omp::TeamsOp>(coexecute->getParentOp());
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

    LLVM_DEBUG(llvm::dbgs() << TAG << "Fission " << coexecute << "\n");

    auto *teamsBlock = &teams.getRegion().front();

    // While we have unhandled operations in the original coexecute
    auto *coexecuteBlock = &coexecute.getRegion().front();
    auto *terminator = coexecuteBlock->getTerminator();
    bool changed = false;
    while (&coexecuteBlock->front() != terminator) {
      rewriter.setInsertionPoint(teams);
      mlir::IRMapping mapping;
      llvm::SmallVector<mlir::Operation *> hoisted;
      mlir::Operation *parallelize = nullptr;
      for (auto &op : coexecute.getOps()) {
        if (&op == terminator) {
          break;
        }
        if (shouldParallelize(&op)) {
          LLVM_DEBUG(llvm::dbgs() << TAG << "Will parallelize " << op << "\n");
          parallelize = &op;
          break;
        } else {
          rewriter.clone(op, mapping);
          LLVM_DEBUG(llvm::dbgs() << TAG << "Hoisting " << op << "\n");
          hoisted.push_back(&op);
          changed = true;
        }
      }

      for (auto *op : hoisted)
        rewriter.replaceOp(op, mapping.lookup(op));

      if (parallelize && hoisted.empty() &&
          parallelize->getNextNode() == terminator)
        break;
      if (parallelize) {
        // TODO do we need any special handling for teams region, block args
        // etc?
        auto newTeams = rewriter.cloneWithoutRegions(teams);
        auto *newTeamsBlock = rewriter.createBlock(
            &newTeams.getRegion(), newTeams.getRegion().begin(), {}, {});
        for (auto arg : teamsBlock->getArguments())
          newTeamsBlock->addArgument(arg.getType(), arg.getLoc());
        auto newCoexecute = rewriter.create<mlir::omp::CoexecuteOp>(loc);
        rewriter.create<mlir::omp::TerminatorOp>(loc);
        rewriter.createBlock(&newCoexecute.getRegion(),
                             newCoexecute.getRegion().begin(), {}, {});
        auto *cloned = rewriter.clone(*parallelize);
        rewriter.replaceOp(parallelize, cloned);
        rewriter.create<mlir::omp::TerminatorOp>(loc);
        changed = true;
      }
    }

    LLVM_DEBUG({
      if (changed) {
        llvm::dbgs() << TAG << "After fission:\n"
                     << *teams->getParentOp() << "\n";
      }
    });

    return mlir::success(changed);
  }
};

struct CoexecuteToSingle
    : public mlir::OpRewritePattern<mlir::omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::omp::CoexecuteOp coexecute,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = coexecute->getLoc();
    auto teams = mlir::dyn_cast<mlir::omp::TeamsOp>(coexecute->getParentOp());
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

class OMPOptPass : public fir::impl::OMPOptBase<OMPOptPass> {
public:
  void runOnOperation() override;
};

void OMPOptPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  mlir::Operation *op = getOperation();

  LLVM_DEBUG({
    using llvm::dbgs;
    dbgs() << "Dumping memory effects\n";
    op->walk([](mlir::omp::CoexecuteOp coexecute) {
      dbgs() << "For " << coexecute << ":\n";
      for (auto &op : coexecute.getOps()) {
        dbgs() << op << ": ";
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
        mlir::MemoryEffectOpInterface interface =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
        if (!interface) {
          dbgs() << "No memory effect interface\n";
          continue;
        }
        interface.getEffects(effects);
        if (effects.empty())
          dbgs() << "Pure";
        for (auto &effect : effects) {
          if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
            dbgs() << "Read(" << effect.getValue() << ") ";
          } else if (mlir::isa<mlir::MemoryEffects::Write>(
                         effect.getEffect())) {
            dbgs() << "Write(" << effect.getValue() << ") ";
          } else if (mlir::isa<mlir::MemoryEffects::Allocate>(
                         effect.getEffect())) {
            dbgs() << "Alloc(" << effect.getValue() << ") ";
          } else if (mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {
            dbgs() << "Free(" << effect.getValue() << ") ";
          }
        }
        dbgs() << "\n";
      }
    });
  });

  // TODO we should assert here that all of our coexecute regions have a single
  // block and that they are perfectly nested in a teams region so as not to
  // duplicate that check (that check should probably be in a LLVM_DEBUG?)

  mlir::MLIRContext &context = getContext();
  mlir::RewritePatternSet patterns(&context);
  mlir::GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = false;

  patterns.insert<FissionCoexecute, CoexecuteToSingle>(&context);
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(op, std::move(patterns),
                                                      config))) {
    mlir::emitError(op->getLoc(), "error in OpenMP optimizations\n");
    signalPassFailure();
  }
}

/// OpenMP optimizations
std::unique_ptr<mlir::Pass> fir::createOMPOptPass() {
  return std::make_unique<OMPOptPass>();
}
