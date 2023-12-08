//===- OMPOpt.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Handles coexecute lowering
///
/// Currently it happens in two stages:
///
/// 1. At FIR level:
/// We identify parallelism available in omp.coexecute ops, and we fission
/// coexecute op so that we can parallelise the individual chunks.
/// teams{coexecute{do_unordered{}}} nests become parallel{for{}} nests (they
/// should probably become teams{distribute} nests, but currently we do not have
/// that construct available in the omp dialect). teams{coexecute{runtime_func}}
/// become just runtime_func.
///
/// 2. At LLVM level:
/// We fission omp.target ops so that we isolate separale parallel{for}s nested
/// in it in separate omp.target regions. This way we can later replace runtime
/// functions with optimized gpu kernels and preserve the semantics with the
/// target executing the code between the parallel chunks only on one thread.
///
/// NOTE this is done at LLVM level because we do not have an easy way to
/// allocate temporaries on the target
///

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

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

static std::optional<std::tuple<Operation *, bool, bool>>
getNestedOpToIsolate(omp::TargetOp targetOp) {
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto &op : *targetBlock) {
    if (isa<omp::TeamsOp, omp::ParallelOp>(&op)) {
      bool first = &op == &*targetBlock->begin();
      bool last = op.getNextNode() == targetBlock->getTerminator();
      if (first && last)
        return std::nullopt;
      return {{&op, first, last}};
    }
  }
  return std::nullopt;
}

mlir::LLVM::ConstantOp
genI64Constant(mlir::Location loc, mlir::PatternRewriter &rewriter, int value) {
  mlir::Type i64Ty = rewriter.getI64Type();
  mlir::IntegerAttr attr = rewriter.getI64IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, attr);
}

/// Isolates the first target{parallel|teams{}} nest in its own omp.target op
///
/// TODO should we clean up the attributes
/// TODO when splitting a target we need to properly tweak omp.map_info's
/// TODO we should hoist out allocations
struct FissionTarget : public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::TargetOp targetOp,
                                PatternRewriter &rewriter) const override {
    if (targetOp->getAttr("fission_target_already_handled")) {
      return failure();
    }
    auto tuple = getNestedOpToIsolate(targetOp);
    if (!tuple) {
      LLVM_DEBUG(dbgs() << TAG << "No op to isolate\n");
      return failure();
    }

    Operation *toIsolate = std::get<0>(*tuple);
    bool splitBefore = !std::get<1>(*tuple);
    bool splitAfter = !std::get<2>(*tuple);

    LLVM_DEBUG(dbgs() << TAG << "Will isolate " << *toIsolate << " from "
                      << targetOp << "\n");

    auto loc = targetOp->getLoc();

    auto ptrTy = LLVM::LLVMPointerType::get(targetOp.getContext());
    auto allocTemp = [&](Type ty) {
      // TODO This should be a omp_target_alloc or something equivalent which we
      // currently do not have an easy way to generate so I am ignoring this
      // problem for now
      auto alloca = rewriter.create<LLVM::AllocaOp>(
          loc, ptrTy, ty, genI64Constant(loc, rewriter, 1));
      SmallVector<Value> bounds = {rewriter.create<omp::DataBoundsOp>(
          loc, rewriter.getType<mlir::omp::DataBoundsType>(),
          genI64Constant(loc, rewriter, 0), genI64Constant(loc, rewriter, 1),
          nullptr, nullptr, false, nullptr)};
      auto mapInfo = rewriter.create<omp::MapInfoOp>(
          loc, ptrTy, alloca, ty, /*var_ptr_ptr=*/nullptr, bounds,
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, false),
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
                  llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM)),
          rewriter.getAttr<mlir::omp::VariableCaptureKindAttr>(
              mlir::omp::VariableCaptureKind::ByCopy),
          rewriter.getStringAttr("coexecute_tmp"));
      return mapInfo;
    };

    auto usedOutsideSplit = [&](Value v, Operation *splitBefore) {
      auto targetOp = cast<omp::TargetOp>(splitBefore->getParentOp());
      auto *targetBlock = &targetOp.getRegion().front();
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
      auto targetOp = cast<omp::TargetOp>(splitBefore->getParentOp());
      auto *targetBlock = &targetOp.getRegion().front();
      rewriter.setInsertionPoint(targetOp);
      SmallVector<std::tuple<Value, unsigned>> allocs;
      SmallVector<LLVM::LoadOp> toClone;
      auto mapOperands = SmallVector<Value>(targetOp.getMapOperands());
      for (auto it = targetBlock->begin(); it != splitBefore->getIterator();
           it++) {
        // Skip if it is already a load from a mapped argument to the target
        // region
        if (auto loadOp = dyn_cast<LLVM::LoadOp>(it))
          if (auto blockArg = dyn_cast<BlockArgument>(loadOp.getAddr()))
            if (blockArg.getOwner() == targetBlock) {
              toClone.push_back(loadOp);
              continue;
            }
        for (auto res : it->getResults()) {
          if (usedOutsideSplit(res, splitBefore)) {
            auto alloc = allocTemp(res.getType());
            allocs.push_back({res, mapOperands.size()});
            mapOperands.push_back(alloc);
          }
        }
      }

      // Split into two blocks with additional mapppings for the values to be
      // passed in memory across the regions

      rewriter.setInsertionPoint(targetOp);
      auto preTargetOp = rewriter.create<omp::TargetOp>(
          loc, targetOp.getIfExpr(), targetOp.getDevice(),
          targetOp.getThreadLimit(), targetOp.getNowait(), mapOperands);
      preTargetOp->setAttr("fission_target_already_handled",
                           rewriter.getUnitAttr());
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
      for (auto tup : allocs) {
        auto original = std::get<0>(tup);
        auto newArg = preTargetBlock->addArgument(ptrTy, original.getLoc());
        rewriter.create<LLVM::StoreOp>(loc, preMapping.lookup(original),
                                       newArg);
      }
      rewriter.create<omp::TerminatorOp>(loc);

      rewriter.setInsertionPoint(targetOp);
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
      for (auto loadOp : toClone)
        rewriter.clone(*loadOp, postMapping);
      for (auto tup : allocs) {
        auto original = std::get<0>(tup);
        auto newArg = postTargetBlock->addArgument(ptrTy, original.getLoc());
        postMapping.map(original, rewriter.create<LLVM::LoadOp>(
                                      loc, original.getType(), newArg));
      }
      for (auto it = splitBefore->getIterator(); it != targetBlock->end(); it++)
        rewriter.clone(*it, postMapping);

      rewriter.eraseOp(targetOp);
      return postMapping.lookup(splitBefore);
    };

    if (splitBefore && splitAfter) {
      auto *newToIsolate = splitBeforeOp(toIsolate);
      splitBeforeOp(newToIsolate->getNextNode());
      return success();
    }
    if (splitBefore) {
      splitBeforeOp(toIsolate);
      return success();
    }
    if (splitAfter) {
      splitBeforeOp(toIsolate->getNextNode());
      return success();
    }
    llvm_unreachable("we should not have had an op to isolate");
  }
};

template <typename T>
static T getPerfectlyNested(Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  auto &region = op->getRegion(0);
  if (region.getBlocks().size() != 1)
    return nullptr;
  auto *block = &region.front();
  auto *firstOp = &block->front();
  if (auto nested = dyn_cast<T>(firstOp))
    if (firstOp->getNextNode() == block->getTerminator())
      return nested;
  return nullptr;
}

struct TeamsCoexecuteLowering : public OpRewritePattern<omp::TeamsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::TeamsOp teamsOp,
                                PatternRewriter &rewriter) const override {
    auto teamsLoc = teamsOp->getLoc();
    auto coexecuteOp = getPerfectlyNested<omp::CoexecuteOp>(teamsOp);
    if (!coexecuteOp) {
      LLVM_DEBUG(dbgs() << TAG << "No coexecute nested\n");
      return failure();
    }
    auto coexecuteLoc = teamsOp->getLoc();
    assert(teamsOp.getAllReductionVars().empty());

    auto loopOp = getPerfectlyNested<fir::DoLoopOp>(coexecuteOp);
    if (loopOp && shouldParallelize(loopOp)) {
      auto parallelOp = rewriter.create<omp::ParallelOp>(
          teamsLoc, teamsOp.getIfExpr(), /*num_threads_var=*/nullptr,
          teamsOp.getAllocateVars(), teamsOp.getAllocatorsVars(),
          /*reduction_vars=*/ValueRange(), /*reductions=*/nullptr,
          /*proc_bind_val=*/nullptr);
      rewriter.createBlock(&parallelOp.getRegion(),
                           parallelOp.getRegion().begin(), {}, {});
      auto wsLoop = rewriter.create<omp::WsLoopOp>(
          coexecuteLoc, loopOp.getLowerBound(), loopOp.getUpperBound(),
          loopOp.getStep());
      wsLoop.setInclusive(true);
      rewriter.create<omp::TerminatorOp>(coexecuteLoc);
      auto *loopTerminator = loopOp.getBody()->getTerminator();
      assert(loopTerminator->getNumResults() == 0);
      rewriter.setInsertionPoint(loopTerminator);
      rewriter.replaceOpWithNewOp<omp::YieldOp>(
          loopOp.getBody()->getTerminator(), ValueRange());
      rewriter.inlineRegionBefore(loopOp.getRegion(), wsLoop.getRegion(),
                                  wsLoop.getRegion().begin());
      // Currently the number of args in the wsloop block matches the number of
      // args in the do loop, so we do not need to remap any arguments or add
      // new ones, but we may need to when we involve reductions
      rewriter.replaceOp(teamsOp, parallelOp);
      return success();
    }

    auto callOp = getPerfectlyNested<fir::CallOp>(coexecuteOp);
    if (callOp && shouldParallelize(callOp)) {
      Block *coexecuteBlock = &coexecuteOp.getRegion().front();
      rewriter.eraseOp(coexecuteBlock->getTerminator());
      rewriter.inlineBlockBefore(coexecuteBlock, teamsOp);
      rewriter.eraseOp(teamsOp);
      return success();
    }

    // TODO need to handle reductions/ops with results (shouldParallelize has to
    // be update too)

    LLVM_DEBUG(dbgs() << TAG
                      << "Unhandled case in teams{coexecute{}} lowering\n");
    return failure();
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

struct TeamsCoexecuteToSingle : public OpRewritePattern<omp::TeamsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::TeamsOp teamsOp,
                                PatternRewriter &rewriter) const override {
    auto coexecuteOp = getPerfectlyNested<omp::CoexecuteOp>(teamsOp);
    if (!coexecuteOp) {
      LLVM_DEBUG(dbgs() << TAG << "No coexecute nested\n");
      return failure();
    }

    Block *coexecuteBlock = &coexecuteOp.getRegion().front();
    rewriter.eraseOp(coexecuteBlock->getTerminator());
    rewriter.inlineBlockBefore(coexecuteBlock, teamsOp);
    rewriter.eraseOp(teamsOp);

    coexecuteOp.emitWarning("unable to parallelize coexecute");

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
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE "-llvm ===\n");

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
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE "-fir ===\n");

  Operation *op = getOperation();

  LLVM_DEBUG({
    dbgs() << "Dumping memory effects\n";
    op->walk([](omp::CoexecuteOp coexecute) { dumpMemoryEffects(coexecute); });
  });

  // TODO we should assert here that all of our coexecute regions have a single
  // block and that they are perfectly nested in a teams region so as not to
  // duplicate that check (that check should probably be in a LLVM_DEBUG?)

  MLIRContext &context = getContext();
  GreedyRewriteConfig config;
  // prevent the pattern driver form merging blocks
  config.enableRegionSimplification = false;

  {
    RewritePatternSet patterns(&context);
    patterns.insert<FissionCoexecute>(&context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc(), "error in OpenMP optimizations\n");
      signalPassFailure();
    }
  }
  {
    RewritePatternSet patterns(&context);
    patterns.insert<TeamsCoexecuteLowering, TeamsCoexecuteToSingle>(&context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc(), "error in OpenMP optimizations\n");
      signalPassFailure();
    }
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
