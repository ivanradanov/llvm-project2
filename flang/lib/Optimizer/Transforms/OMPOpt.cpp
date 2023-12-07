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
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace fir {
#define GEN_PASS_DEF_FIROMPOPT
#define GEN_PASS_DEF_LLVMOMPOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-omp-opt"
#define TAG ("[" DEBUG_TYPE "] ")

using llvm::dbgs;
using namespace mlir;

/// This is the single source of truth about whether we should parallelize an
/// operation nested in an omp.execute region.
static bool shouldParallelize(Operation *op) {
  // TODO Currently we cannot parallelize operations with results that have uses
  // - we need additional handling in the fission for that i.e. a way to access
  // that result outside the
  if (llvm::any_of(op->getResults(),
                   [](OpResult v) -> bool { return !v.use_empty(); }))
    return false;
  // We will parallelize unordered loops - these come from array syntax
  if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
    auto unordered = loop.getUnordered();
    if (!unordered)
      return false;
    return *unordered;
  }
  if (auto callOp = dyn_cast<fir::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    auto *func = op->getParentOfType<ModuleOp>().lookupSymbol(*callee);
    // TODO need to insert a check here whether it is a call we can actually
    // parallelize currently
    if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
      return true;
    return false;
  }
  // We cannot parallise anything else
  return false;
}

static omp::TeamsOp getTeamsOpToIsolate(omp::TargetOp targetOp) {
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto &op : *targetBlock) {
    if (auto teamsOp = dyn_cast<omp::TeamsOp>(&op)) {
      if (teamsOp->getNextNode() != targetBlock->getTerminator() &&
          teamsOp != &*targetBlock->begin())
        return teamsOp;
      return nullptr;
    }
  }
  return nullptr;
}

mlir::LLVM::ConstantOp
genI32Constant(mlir::Location loc, mlir::PatternRewriter &rewriter, int value) {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

/// Isolates the first teams{coexecute{}} nest in its own omp.target op
struct FissionTarget : public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::TargetOp targetOp,
                                PatternRewriter &rewriter) const override {
    auto teamsOp = getTeamsOpToIsolate(targetOp);
    if (!teamsOp) {
      LLVM_DEBUG(dbgs() << TAG << "No teams op to isolate\n");
      return failure();
    }

    auto loc = targetOp->getLoc();

    auto allocTemp = [&](Type ty) {
      // TODO This should be a omp_target_alloc or something equivalent which we
      // currently do not have an easy way to generate so I am ignoring this
      // problem for now
      return rewriter.create<LLVM::AllocaOp>(
          loc, LLVM::LLVMPointerType::get(ty.getContext()), ty,
          genI32Constant(loc, rewriter, 1));
    };

    auto *targetBlock = &targetOp.getRegion().front();
    auto usedOutsideSplit = [&](Value v, Operation *splitBefore) {
      for (auto *user : v.getUsers()) {
        while (user->getBlock() != targetBlock) {
          user = user->getParentOp();
        }
        if (!user->isBeforeInBlock(splitBefore))
          return true;
      }
      return false;
    };
    auto splitBeforeOp = [&](Operation *splitBefore) {
      rewriter.setInsertionPoint(targetOp);
      IRMapping toReplace;
      SmallVector<std::pair<Value, Value>> allocs;
      auto mapOperands = SmallVector<Value>(targetOp.getMapOperands());
      for (auto &op : *targetBlock) {
        if (&op == splitBefore)
          break;
        for (auto res : op.getResults()) {
          if (usedOutsideSplit(res, splitBefore)) {
            auto alloc = allocTemp(res.getType());
            allocs.push_back({alloc, res});
            toReplace.map(res, alloc);
            mapOperands.push_back(alloc);
          }
        }
      }

      // Split into two blocks with additional mapppings for the values to be
      // passed in memory across the regions

      auto preTargetOp = rewriter.create<omp::TargetOp>(
          loc, targetOp.getIfExpr(), targetOp.getDevice(),
          targetOp.getThreadLimit(), targetOp.getNowait(), mapOperands);
      auto *preTargetBlock = rewriter.createBlock(
          &preTargetOp.getRegion(), preTargetOp.getRegion().begin(), {}, {});
      IRMapping preMapping;
      for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
        auto originalArg = targetBlock->getArgument(i);
        auto newArg = preTargetBlock->addArgument(originalArg.getType(),
                                                  originalArg.getLoc());
        preMapping.map(originalArg, newArg);
      }
      for (auto it = targetBlock->begin(); it != splitBefore->getIterator();
           it++)
        rewriter.clone(*it, preMapping);
      for (auto pair : allocs) {
        auto alloc = std::get<0>(pair);
        auto original = std::get<0>(pair);
        rewriter.create<LLVM::StoreOp>(loc, preMapping.lookup(original), alloc);
      }
      rewriter.create<omp::TerminatorOp>(loc);

      auto postTargetOp = rewriter.create<omp::TargetOp>(
          loc, targetOp.getIfExpr(), targetOp.getDevice(),
          targetOp.getThreadLimit(), targetOp.getNowait(), mapOperands);
      auto *postTargetBlock = rewriter.createBlock(
          &postTargetOp.getRegion(), postTargetOp.getRegion().begin(), {}, {});
      IRMapping postMapping;
      for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
        auto originalArg = targetBlock->getArgument(i);
        auto newArg = postTargetBlock->addArgument(originalArg.getType(),
                                                   originalArg.getLoc());
        postMapping.map(originalArg, newArg);
      }
      for (auto pair : allocs) {
        auto alloc = std::get<0>(pair);
        auto original = std::get<0>(pair);
        auto newArg =
            postTargetBlock->addArgument(alloc.getType(), alloc.getLoc());
        postMapping.map(original, rewriter.create<LLVM::LoadOp>(
                                      loc, original.getType(), newArg));
      }
      for (auto it = splitBefore->getIterator(); it != targetBlock->end(); it++)
        rewriter.clone(*it, postMapping);

      rewriter.eraseOp(targetOp);
      targetOp = postTargetOp;
    };

    auto *nextOp = teamsOp->getNextNode();
    bool splitAfter = nextOp == teamsOp->getBlock()->getTerminator();
    splitBeforeOp(teamsOp);
    if (splitAfter)
      splitBeforeOp(targetOp.getRegion().front().front().getNextNode());

    return success();
  }
};

/// If B() and D() are parallelizable,
///
/// omp.teams {
///   omp.coexecute {
///     A()
///     B()
///     C()
///     D()
///     E()
///   }
/// }
///
/// becomes
///
/// A()
/// omp.teams {
///   omp.coexecute {
///     B()
///   }
/// }
/// C()
/// omp.teams {
///   omp.coexecute {
///     D()
///   }
/// }
/// E()
struct FissionCoexecute : public OpRewritePattern<omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::CoexecuteOp coexecute,
                                PatternRewriter &rewriter) const override {
    auto loc = coexecute->getLoc();
    auto teams = dyn_cast<omp::TeamsOp>(coexecute->getParentOp());
    if (!teams) {
      emitError(loc, "coexecute not nested in teams\n");
      return failure();
    }
    if (coexecute.getRegion().getBlocks().size() != 1) {
      emitError(loc, "coexecute with multiple blocks\n");
      return failure();
    }
    if (teams.getRegion().getBlocks().size() != 1) {
      emitError(loc, "teams with multiple blocks\n");
      return failure();
    }
    if (teams.getRegion().getBlocks().front().getOperations().size() != 2) {
      emitError(loc, "teams with multiple nested ops\n");
      return failure();
    }

    LLVM_DEBUG(dbgs() << TAG << "Fission " << coexecute << "\n");

    auto *teamsBlock = &teams.getRegion().front();

    // While we have unhandled operations in the original coexecute
    auto *coexecuteBlock = &coexecute.getRegion().front();
    auto *terminator = coexecuteBlock->getTerminator();
    bool changed = false;
    while (&coexecuteBlock->front() != terminator) {
      rewriter.setInsertionPoint(teams);
      IRMapping mapping;
      llvm::SmallVector<Operation *> hoisted;
      Operation *parallelize = nullptr;
      for (auto &op : coexecute.getOps()) {
        if (&op == terminator) {
          break;
        }
        if (shouldParallelize(&op)) {
          LLVM_DEBUG(dbgs() << TAG << "Will parallelize " << op << "\n");
          parallelize = &op;
          break;
        } else {
          rewriter.clone(op, mapping);
          LLVM_DEBUG(dbgs() << TAG << "Hoisting " << op << "\n");
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
        auto newCoexecute = rewriter.create<omp::CoexecuteOp>(loc);
        rewriter.create<omp::TerminatorOp>(loc);
        rewriter.createBlock(&newCoexecute.getRegion(),
                             newCoexecute.getRegion().begin(), {}, {});
        auto *cloned = rewriter.clone(*parallelize);
        rewriter.replaceOp(parallelize, cloned);
        rewriter.create<omp::TerminatorOp>(loc);
        changed = true;
      }
    }

    LLVM_DEBUG({
      if (changed) {
        dbgs() << TAG << "After fission:\n" << *teams->getParentOp() << "\n";
      }
    });

    return success(changed);
  }
};

struct CoexecuteToSingle : public OpRewritePattern<omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::CoexecuteOp coexecute,
                                PatternRewriter &rewriter) const override {
    auto loc = coexecute->getLoc();
    auto teams = dyn_cast<omp::TeamsOp>(coexecute->getParentOp());
    if (!teams) {
      emitError(loc, "coexecute not nested in teams\n");
      return failure();
    }
    if (coexecute.getRegion().getBlocks().size() != 1) {
      emitError(loc, "coexecute with multiple blocks\n");
      return failure();
    }
    if (teams.getRegion().getBlocks().size() != 1) {
      emitError(loc, "teams with multiple blocks\n");
      return failure();
    }
    if (teams.getRegion().getBlocks().front().getOperations().size() != 2) {
      emitError(loc, "teams with multiple nested ops\n");
      return failure();
    }
    Block *coexecuteBlock = &coexecute.getRegion().front();
    rewriter.eraseOp(coexecuteBlock->getTerminator());
    rewriter.inlineBlockBefore(coexecuteBlock, teams);
    rewriter.eraseOp(teams);
    return success();
  }
};

static void dumpMemoryEffects(Operation *op) {
  dbgs() << "For " << *op << ":\n";
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto &op : block) {
        dbgs() << op << ": ";
        llvm::SmallVector<MemoryEffects::EffectInstance> effects;
        MemoryEffectOpInterface interface =
            dyn_cast<MemoryEffectOpInterface>(op);
        if (!interface) {
          dbgs() << "No memory effect interface\n";
          continue;
        }
        interface.getEffects(effects);
        if (effects.empty())
          dbgs() << "Pure";
        for (auto &effect : effects) {
          if (isa<MemoryEffects::Read>(effect.getEffect())) {
            dbgs() << "Read(" << effect.getValue() << ") ";
          } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
            dbgs() << "Write(" << effect.getValue() << ") ";
          } else if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
            dbgs() << "Alloc(" << effect.getValue() << ") ";
          } else if (isa<MemoryEffects::Free>(effect.getEffect())) {
            dbgs() << "Free(" << effect.getValue() << ") ";
          }
        }
        dbgs() << "\n";
      }
    }
  }
}

class LLVMOMPOptPass : public ::fir::impl::LLVMOMPOptBase<LLVMOMPOptPass> {
public:
  void runOnOperation() override;
};

class FIROMPOptPass : public ::fir::impl::FIROMPOptBase<FIROMPOptPass> {
public:
  void runOnOperation() override;
};

void LLVMOMPOptPass::runOnOperation() {
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  Operation *op = getOperation();
  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context);
  GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = false;

  patterns.insert<FissionTarget>(&context);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    emitError(op->getLoc(), "error in OpenMP optimizations\n");
    signalPassFailure();
  }
}

void FIROMPOptPass::runOnOperation() {
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE " ===\n");

  Operation *op = getOperation();

  LLVM_DEBUG({
    dbgs() << "Dumping memory effects\n";
    op->walk([](omp::CoexecuteOp coexecute) { dumpMemoryEffects(coexecute); });
  });

  // TODO we should assert here that all of our coexecute regions have a single
  // block and that they are perfectly nested in a teams region so as not to
  // duplicate that check (that check should probably be in a LLVM_DEBUG?)

  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context);
  GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = false;

  patterns.insert<FissionCoexecute>(&context);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    emitError(op->getLoc(), "error in OpenMP optimizations\n");
    signalPassFailure();
  }
}

/// OpenMP optimizations
std::unique_ptr<Pass> fir::createLLVMOMPOptPass() {
  return std::make_unique<LLVMOMPOptPass>();
}
/// OpenMP optimizations
std::unique_ptr<Pass> fir::createFIROMPOptPass() {
  return std::make_unique<FIROMPOptPass>();
}
