//===- LoopUnrollAndInterleavePass.cpp - Loop unroller pass --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop unroller.  It works best when loops have
// been canonicalized by the -indvars pass, allowing it to determine the trip
// counts of loops easily.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopUnrollAndInterleavePass.h"
#include "llvm-c/Core.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-and-interleave"

static LoopUnrollResult
tryToUnrollLoop(Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
                const TargetTransformInfo &TTI, AssumptionCache &AC,
                OptimizationRemarkEmitter &ORE, BlockFrequencyInfo *BFI,
                ProfileSummaryInfo *PSI) {

  LLVM_DEBUG(dbgs() << "Loop Unroll And Interleave: F["
                    << L->getHeader()->getParent()->getName() << "] Loop %"
                    << L->getHeader()->getName() << "\n");
  // TODO Encode unroll and interleave info in the loop

  if (!L->getExitBlock()->getParent()->getName().ends_with("omp_outlined"))
    return LoopUnrollResult::Unmodified;

  // Find the smallest exact trip count for any exit. This is an upper bound
  // on the loop trip count, but an exit at an earlier iteration is still
  // possible. An unroll by the smallest exact trip count guarantees that all
  // branches relating to at least one exit can be eliminated. This is unlike
  // the max trip count, which only guarantees that the backedge can be broken.
  unsigned TripCount = 0;
  unsigned TripMultiple = 1;
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (BasicBlock *ExitingBlock : ExitingBlocks)
    if (unsigned TC = SE.getSmallConstantTripCount(L, ExitingBlock))
      if (!TripCount || TC < TripCount)
        TripCount = TripMultiple = TC;

  if (!TripCount) {
    // If no exact trip count is known, determine the trip multiple of either
    // the loop latch or the single exiting block.
    // TODO: Relax for multiple exits.
    BasicBlock *ExitingBlock = L->getLoopLatch();
    if (!ExitingBlock || !L->isLoopExiting(ExitingBlock))
      ExitingBlock = L->getExitingBlock();
    if (ExitingBlock)
      TripMultiple = SE.getSmallConstantTripMultiple(L, ExitingBlock);
  }

  // TODO generate epilogue, while making sure to handle convergent insts
  // properly (e.g. __syncthreads())

  // Save loop properties before it is transformed.
  MDNode *OrigLoopID = L->getLoopID();

  BasicBlock *Preheader = L->getLoopPreheader();
  assert(Preheader && "Expected a loop preheader");
  BasicBlock *Header = L->getHeader();
  BasicBlock *LatchBlock = L->getLoopLatch();
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  std::vector<BasicBlock *> OriginalLoopBlocks = L->getBlocks();

  unsigned UnrollFactor = 4;

  // TODO abort if trip countis not divisible by the factor (or use the original
  // non coarsened loop) we expect the runtime to call us with the appropriate
  // trip count

  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> VMaps;
  VMaps.reserve(UnrollFactor);
  for (unsigned I = 0; I < UnrollFactor; I++)
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());

  auto LoopBounds = L->getBounds(SE);
  if (LoopBounds == std::nullopt) {
    LLVM_DEBUG(dbgs() << "Unable to find loop bounds of the omp workshare loop, not coarsening\n");
    return LoopUnrollResult::Unmodified;
  }

  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());

  Value *IVStepVal = LoopBounds->getStepValue();
  Instruction *IVStepInst = dyn_cast_or_null<Instruction>(IVStepVal);
  if (!IVStepInst) {
    LLVM_DEBUG(dbgs() << "Unexpected step definition");
    if (IVStepVal)
      LLVM_DEBUG(dbgs() << *IVStepVal);
    LLVM_DEBUG(dbgs() << "\n");
    return LoopUnrollResult::Unmodified;
  }

  // The new Step is UnrollFactor * Step
  Value *NewStep = PreheaderBuilder.CreateMul(
      IVStepVal, ConstantInt::get(IVStepVal->getType(), UnrollFactor));
  IVStepInst->replaceUsesWithIf(NewStep, [NewStep](Use &U) -> bool {
    return U.getUser() != NewStep;
  });

  // Set up new initial IV values, for now we do initial + stride, initial + 2 * stride, ...,
  // initial + (UnrollFactor - 1) * stride
  Value *InitialIVVal = &LoopBounds->getInitialIVValue();
  Instruction *InitialIVInst = dyn_cast<Instruction>(InitialIVVal) ;
  if (!InitialIVInst) {
    LLVM_DEBUG(dbgs() << "Unexpected initial val definition" << *InitialIVVal << "\n");
    return LoopUnrollResult::Unmodified;
  }
  for (unsigned I = 1; I < UnrollFactor; I++) {
    Value *MultipliedStep = PreheaderBuilder.CreateMul(
        IVStepVal, ConstantInt::get(IVStepVal->getType(), I));
    Value *CoarsenedInitialIV = PreheaderBuilder.CreateAdd(
        InitialIVVal, MultipliedStep);
    (*VMaps[I])[InitialIVVal] = CoarsenedInitialIV;
  }

  // Interleave instructions

  // TODO Currently we do not check whether the control flow may be divergent
  // between the interleaved original "iterations" - we need to duplicate
  // instead of interleave divergent flow
  for (BasicBlock *BB : OriginalLoopBlocks) {
    SmallVector<Instruction *> ToClone;
    for (Instruction &I : *BB) {
      ToClone.push_back(&I);
    }
    for (Instruction *I : ToClone) {
      Instruction *LastI = I;
      for (unsigned It = 1; It < UnrollFactor; It++) {
        bool IsLastClone = It + 1 == UnrollFactor;
        // Only clone the last of the cloned terminators
        if (I->isTerminator() && !IsLastClone) {
          continue;
        }

        Instruction *Cloned = I->clone();
        Cloned->insertAfter(LastI);
        RemapInstruction(Cloned, *VMaps[It], RemapFlags::RF_IgnoreMissingLocals);
        (*VMaps[It])[I] = Cloned;
        LastI = Cloned;

        if (I->isTerminator() ) {
          assert(IsLastClone);
          assert(isa<BranchInst>(I) && "Unhandled case");
          // Remove the original terminator, we will be using the one cloned
          // last
          I->eraseFromParent();
        }
      }
    }
  }

  return LoopUnrollResult::PartiallyUnrolled;
}

PreservedAnalyses LoopUnrollAndInterleavePass::run(Loop &L, LoopAnalysisManager &AM,
                                          LoopStandardAnalysisResults &AR,
                                          LPMUpdater &Updater) {
  // For the new PM, we can't use OptimizationRemarkEmitter as an analysis
  // pass. Function analyses need to be preserved across loop transformations
  // but ORE cannot be preserved (see comment before the pass definition).
  OptimizationRemarkEmitter ORE(L.getHeader()->getParent());

  // Keep track of the previous loop structure so we can identify new loops
  // created by unrolling.
  Loop *ParentL = L.getParentLoop();
  SmallPtrSet<Loop *, 4> OldLoops;
  if (ParentL)
    OldLoops.insert(ParentL->begin(), ParentL->end());
  else
    OldLoops.insert(AR.LI.begin(), AR.LI.end());

  std::string LoopName = std::string(L.getName());

  bool Changed =
      tryToUnrollLoop(&L, AR.DT, &AR.LI, AR.SE, AR.TTI, AR.AC, ORE,
                      /*BFI*/ nullptr, /*PSI*/ nullptr) !=
      LoopUnrollResult::Unmodified;
  if (!Changed)
    return PreservedAnalyses::all();

  // The parent must not be damaged by unrolling!
#ifndef NDEBUG
  if (ParentL)
    ParentL->verifyLoop();
#endif

  // Unrolling can do several things to introduce new loops into a loop nest:
  // - Full unrolling clones child loops within the current loop but then
  //   removes the current loop making all of the children appear to be new
  //   sibling loops.
  //
  // When a new loop appears as a sibling loop after fully unrolling,
  // its nesting structure has fundamentally changed and we want to revisit
  // it to reflect that.
  //
  // When unrolling has removed the current loop, we need to tell the
  // infrastructure that it is gone.
  //
  // Finally, we support a debugging/testing mode where we revisit child loops
  // as well. These are not expected to require further optimizations as either
  // they or the loop they were cloned from have been directly visited already.
  // But the debugging mode allows us to check this assumption.
  bool IsCurrentLoopValid = false;
  SmallVector<Loop *, 4> SibLoops;
  if (ParentL)
    SibLoops.append(ParentL->begin(), ParentL->end());
  else
    SibLoops.append(AR.LI.begin(), AR.LI.end());
  erase_if(SibLoops, [&](Loop *SibLoop) {
    if (SibLoop == &L) {
      IsCurrentLoopValid = true;
      return true;
    }

    // Otherwise erase the loop from the list if it was in the old loops.
    return OldLoops.contains(SibLoop);
  });
  Updater.addSiblingLoops(SibLoops);

  if (!IsCurrentLoopValid) {
    Updater.markLoopAsDeleted(L, LoopName);
  }
  return getLoopPassPreservedAnalyses();
}
