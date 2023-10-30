//===- LoopUnrollAndInterleavePass.cpp - Loop unroller pass----------------===//
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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
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
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-and-interleave"

static bool isWorkShareLoop(Loop *L) {
  if (!L->isOutermost())
    return false;

  Function *F = L->getHeader()->getParent();
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (CallInst *CI = dyn_cast<CallInst>(&I))
        if (Function *CF = CI->getCalledFunction())
          if (CF->getName().starts_with("__kmpc_for_static_init"))
            return true;

  return false;
}

static void setLoopAlreadyCoarsened(Loop *L) {
  LLVMContext &Context = L->getHeader()->getContext();

  MDNode *DisableUnrollMD =
      MDNode::get(Context, MDString::get(Context, "llvm.loop.coarsen.disable"));
  MDNode *LoopID = L->getLoopID();
  MDNode *NewLoopID = makePostTransformationMetadata(
      Context, LoopID, {"llvm.loop.coarsen."}, {DisableUnrollMD});
  L->setLoopID(NewLoopID);
}

static MDNode *getUnrollMetadataForLoop(const Loop *L, StringRef Name) {
  if (MDNode *LoopID = L->getLoopID())
    return GetUnrollMetadata(LoopID, Name);
  return nullptr;
}

static bool getLoopAlreadyCoarsened(Loop *L) {
  return getUnrollMetadataForLoop(L, "llvm.loop.coarsen.disable");
}

static LoopUnrollResult
tryToUnrollLoop(Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
                const TargetTransformInfo &TTI, AssumptionCache &AC,
                OptimizationRemarkEmitter &ORE, BlockFrequencyInfo *BFI,
                ProfileSummaryInfo *PSI) {

  if (getLoopAlreadyCoarsened(L)) {
    LLVM_DEBUG(dbgs() << "Already coarsened\n");
    return LoopUnrollResult::Unmodified;
  }

  Function *F = L->getHeader()->getParent();
  LLVM_DEBUG(dbgs() << "Loop Unroll And Interleave: F["
                    << L->getHeader()->getParent()->getName() << "] Loop %"
                    << L->getHeader()->getName() << " "
                    << "Parallel=" << L->isAnnotatedParallel() << " ");

  if (!isWorkShareLoop(L)) {
    LLVM_DEBUG(dbgs() << "Not work share loop\n");
    return LoopUnrollResult::Unmodified;
  }

  // TODO handle convergent insts properly (e.g. __syncthreads())

  // Save loop properties before it is transformed.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    // We delete the preheader of the epilogue loop so this is currently how we
    // detect that this may be the epilogue loop, because all other loops should
    // have a preheader after being simplified before this pass
    LLVM_DEBUG(dbgs() << "No preheader\n");
    return LoopUnrollResult::Unmodified;
  }
  BasicBlock *LatchBlock = L->getLoopLatch();
  assert(LatchBlock);
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  BasicBlock *ExitBlock = L->getExitBlock();
  std::vector<BasicBlock *> OriginalLoopBlocks = L->getBlocks();

  // Disabled by default
  unsigned UnrollFactor = 1;
  if (char *env = getenv("UNROLL_AND_INTERLEAVE_FACTOR"))
    StringRef(env).getAsInteger(10, UnrollFactor);
  assert(UnrollFactor > 0);
  if (UnrollFactor == 1) {
    LLVM_DEBUG(dbgs() << "Unroll factor of 1 - ignoring\n");
    return LoopUnrollResult::Unmodified;
  }

  bool Chunkify = false;
  if (char *env = getenv("UNROLL_AND_INTERLEAVE_CHUNKIFY")) {
    unsigned Int = 0;
    StringRef(env).getAsInteger(10, Int);
    Chunkify = Int;
  }

  // Change the kmpc_call chunk size
  if (Chunkify) {
    Function *F = L->getHeader()->getParent();
    bool Found = false;
    [&]() {
      for (BasicBlock &BB : *F) {
        for (Instruction &I : BB) {
          if (CallInst *CI = dyn_cast<CallInst>(&I)) {
            if (Function *CF = CI->getCalledFunction()) {
              if (CF->getName() == "__kmpc_for_static_init_8u") {
                unsigned ChunkSizeOperandNum = 8;
                ConstantInt *ConstInt =
                    dyn_cast<ConstantInt>(CI->getOperand(ChunkSizeOperandNum));
                assert(ConstInt);
                CI->setOperand(
                    ChunkSizeOperandNum,
                    ConstantInt::get(
                        CI->getOperand(ChunkSizeOperandNum)->getType(),
                        UnrollFactor * ConstInt->getValue()));
                Found = true;
                return;
              }
            }
          }
        }
      }
    }();
    assert(Found && "Did not find kmpc call");
  }

  // TODO abort if trip countis not divisible by the factor (or use the original
  // non coarsened loop) we expect the runtime to call us with the appropriate
  // trip count

  auto LoopBounds = L->getBounds(SE);
  if (LoopBounds == std::nullopt) {
    LLVM_DEBUG(dbgs() << "Unable to find loop bounds of the omp workshare "
                         "loop, not coarsening\n");
    return LoopUnrollResult::Unmodified;
  }

  // Clone the loop to use as an epilogue, the original one will be coarsened
  // in-place
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *> EpilogueLoopBlocks;
  // TODO insert after the last block of the loop
  Loop *EpilogueLoop =
      cloneLoopWithPreheader(LatchBlock->getNextNode(), Preheader, L, VMap,
                             ".epilogue", LI, &DT, EpilogueLoopBlocks);
  auto IsInEpilogue = [&](Use &U) -> bool {
    if (Instruction *I = dyn_cast<Instruction>(U.getUser())) {
      if (std::find(EpilogueLoopBlocks.begin(), EpilogueLoopBlocks.end(),
                    I->getParent()) != EpilogueLoopBlocks.end())
        return true;
      return false;
    }
    llvm_unreachable("Uses of lb should only be instructions");
  };

  // VMaps for the separate interleaved iterations
  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> VMaps;
  VMaps.reserve(UnrollFactor);
  for (unsigned I = 0; I < UnrollFactor; I++)
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());

  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());

  // The new Step is UnrollFactor * OriginalStep
  Value *IVStepVal = LoopBounds->getStepValue();
  if (Instruction *IVStepInst = dyn_cast_or_null<Instruction>(IVStepVal)) {
    if (!Chunkify) {
      Value *NewStep = PreheaderBuilder.CreateMul(
          IVStepVal, ConstantInt::get(IVStepVal->getType(), UnrollFactor),
          "coarsened.step");
      IVStepInst->replaceUsesWithIf(
          NewStep, [NewStep, IsInEpilogue](Use &U) -> bool {
            return U.getUser() != NewStep && !IsInEpilogue(U);
          });
    }
  } else {
    // If the step is a constant
    auto *IVStepConst = dyn_cast<ConstantInt>(IVStepVal);
    assert(IVStepConst);
    Value *NewStep = ConstantInt::getIntegerValue(
        IntegerType::getInt32Ty(IVStepVal->getContext()),
        UnrollFactor * IVStepConst->getValue());
    Instruction *StepInst = &LoopBounds->getStepInst();
    for (unsigned It = 0; It < StepInst->getNumOperands(); It++)
      if (StepInst->getOperand(It) == IVStepVal)
        StepInst->setOperand(It, NewStep);
  }

  // Set up new initial IV values, for now we do initial + stride, initial + 2 *
  // stride, ..., initial + (UnrollFactor - 1) * stride
  Value *InitialIVVal = &LoopBounds->getInitialIVValue();
  Instruction *InitialIVInst = dyn_cast<Instruction>(InitialIVVal);
  if (!InitialIVInst) {
    LLVM_DEBUG(dbgs() << "Unexpected initial val definition" << *InitialIVVal
                      << "\n");
    return LoopUnrollResult::Unmodified;
  }
  for (unsigned I = 1; I < UnrollFactor; I++) {
    Value *CoarsenedInitialIV;
    if (Chunkify) {
      CoarsenedInitialIV = PreheaderBuilder.CreateAdd(
          InitialIVVal, ConstantInt::get(IVStepVal->getType(), I));
    } else {
      Value *MultipliedStep = PreheaderBuilder.CreateMul(
          IVStepVal, ConstantInt::get(IVStepVal->getType(), I));
      CoarsenedInitialIV = PreheaderBuilder.CreateAdd(
          InitialIVVal, MultipliedStep,
          "initial.iv.coarsened." + std::to_string(I));
    }
    (*VMaps[I])[InitialIVVal] = CoarsenedInitialIV;
  }

  // Interleave instructions

  SmallVector<SmallVector<Instruction *>> ClonedInsts;
  ClonedInsts.reserve(UnrollFactor);
  for (unsigned I = 0; I < UnrollFactor; I++)
    ClonedInsts.push_back({});

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
        // Do not clone terminators - we use the contrl flow of the existing
        // iteration (we assume all iterations follow the same control flow)
        if (I->isTerminator()) {
          continue;
        }

        Instruction *Cloned = I->clone();
        Cloned->insertAfter(LastI);
        if (!Cloned->getType()->isVoidTy())
          Cloned->setName(I->getName() + ".coarsened." + std::to_string(It));
        ClonedInsts[It].push_back(Cloned);
        (*VMaps[It])[I] = Cloned;
        LastI = Cloned;
      }
    }
  }

  for (unsigned It = 1; It < UnrollFactor; It++)
    for (Instruction *I : ClonedInsts[It])
      RemapInstruction(I, *VMaps[It], RemapFlags::RF_IgnoreMissingLocals);

  // Plumbing around the coarsened and epilogue loops

  BasicBlock *EpiloguePH = cast<BasicBlock>(VMap[Preheader]);
  EpilogueLoopBlocks.erase(std::find(EpilogueLoopBlocks.begin(),
                                     EpilogueLoopBlocks.end(), EpiloguePH));
  for (Instruction &I : *Preheader) {
    VMap.erase(&I);
  }
  VMap.erase(Preheader);
  remapInstructionsInBlocks(EpilogueLoopBlocks, VMap);

  auto HoistEnd = [&](Value *End) -> Value * {
    Instruction *I = dyn_cast<Instruction>(End);
    if (!I)
      return End;

    BasicBlock *BB = I->getParent();
    if (BB == Preheader ||
        std::find(OriginalLoopBlocks.begin(), OriginalLoopBlocks.end(), BB) ==
            OriginalLoopBlocks.end())
      return End;

    if (Instruction *I = dyn_cast<Instruction>(End)) {
      // TODO need to make sure this is legal - it should be the case for an omp
      // workshare loop (why didnt we licm it?)
      Instruction *Cloned = I->clone();
      Cloned->insertBefore(Preheader->getTerminator());
      Cloned->setName(I->getName() + ".hoisted.ub");
      I->replaceAllUsesWith(Cloned);
      I->eraseFromParent();
      return Cloned;
    }
    return End;
  };

  // Calc the start value for the epilogue loop, should be:
  // Start + (ceil((End - Start) / Stride) / UnrollFactor) * UnrollFactor *
  // Stride.
  // I.e. when we are done with our iterations of the coarsened loop
  Value *EpilogueStart;
  Value *Start = &LoopBounds->getInitialIVValue();
  Value *End = HoistEnd(&LoopBounds->getFinalIVValue());
  Value *Stride = LoopBounds->getStepValue();
  Value *One = ConstantInt::get(Start->getType(), 1);
  Value *UnrollFactorCst = ConstantInt::get(Start->getType(), UnrollFactor);
  {
    Value *Diff = PreheaderBuilder.CreateSub(End, Start);
    Value *CeilDiv = PreheaderBuilder.CreateUDiv(
        PreheaderBuilder.CreateSub(
            PreheaderBuilder.CreateAdd(Diff, LoopBounds->getStepValue()), One),
        Stride);
    Value *FloorDiv = PreheaderBuilder.CreateUDiv(CeilDiv, UnrollFactorCst);
    Value *Offset = PreheaderBuilder.CreateNSWMul(
        PreheaderBuilder.CreateNSWMul(FloorDiv, UnrollFactorCst), Stride);
    EpilogueStart =
        PreheaderBuilder.CreateAdd(Offset, Start, "epilogue.start.iv");
  }

  // Jump to epilogue from preheader if we are already at its start
  {
    BranchInst *Incoming = dyn_cast<BranchInst>(Preheader->getTerminator());
    assert(Incoming && !Incoming->isConditional());
    IRBuilder<> Builder(Incoming);

    assert(LoopBounds->getDirection() !=
           Loop::LoopBounds::Direction::Decreasing);
    Value *IsAtEpilogueStart = Builder.CreateCmp(
        CmpInst::Predicate::ICMP_EQ, Start, EpilogueStart, "is.epilogue.start");
    Builder.CreateCondBr(IsAtEpilogueStart, EpilogueLoop->getHeader(),
                         Incoming->getSuccessor(0));
    Incoming->eraseFromParent();
  }

  // Jump to epilogue from coarsened latch if we are at its start
  {
    // Note we need to check for the loop end first and exit the loop
    // alltogether if we are at the end because if all iterations are handled by
    // the coarsened loop the final IV will be equal to the epilogue start IV
    BranchInst *BackEdge = dyn_cast<BranchInst>(LatchBlock->getTerminator());
    assert(BackEdge && BackEdge->isConditional());
    BasicBlock *PrevBB = LatchBlock->splitBasicBlockBefore(BackEdge);
    Instruction *BI = PrevBB->getTerminator();
    IRBuilder<> Builder(BI);
    Value *IsAtEpilogueStart = Builder.CreateCmp(
        CmpInst::Predicate::ICMP_EQ, &LoopBounds->getStepInst(), EpilogueStart,
        "is.epilogue.start");
    Value *EpilogueBranch = Builder.CreateCondBr(
        IsAtEpilogueStart, EpilogueLoop->getHeader(), LatchBlock);
    BI->eraseFromParent();
    EpilogueLoop->getHeader()->replacePhiUsesWith(Preheader, PrevBB);

    // TODO Instead of introducing this redundant branch and bb we can redirect
    // the original branch instead so that if it doesnt jump to the exit to do a
    // check for the epilogue loop start first
    BasicBlock *EndCheckBB =
        PrevBB->splitBasicBlockBefore(cast<Instruction>(EpilogueBranch));
    Instruction *EndCheckBI = EndCheckBB->getTerminator();
    IRBuilder<> EpilogueCheckBuilder(EndCheckBI);
    EpilogueCheckBuilder.CreateCondBr(BackEdge->getCondition(), PrevBB,
                                      ExitBlock);
    EndCheckBI->eraseFromParent();
  }

  // Start the epilogue loop from the iteration the coarsened version ended with
  // instead of the original lb
  {
    // TODO should we give it EpilogueStart instead of the StepInst? (they
    // should be equal)
    SmallVector<std::pair<PHINode *, unsigned>> ToHandle;
    for (Use &U : InitialIVVal->uses()) {
      if (!IsInEpilogue(U))
        continue;
      if (PHINode *PN = dyn_cast<PHINode>(U.getUser())) {
        ToHandle.push_back(std::make_pair(PN, U.getOperandNo()));
      } else {
        llvm_unreachable("Non PHI use of lb");
      }
    }
    for (auto &Pair : ToHandle) {
      PHINode *PN = Pair.first;
      unsigned OpNo = Pair.second;
      PN->addIncoming(InitialIVVal, Preheader);
      PN->setOperand(OpNo, &LoopBounds->getStepInst());
    }
  }

  EpiloguePH->eraseFromParent();

  LLVM_DEBUG(llvm::dbgs() << "After unroll and interleave:\n" << *F);

  simplifyLoop(L, nullptr, nullptr, nullptr, nullptr, nullptr, false);
  simplifyLoop(EpilogueLoop, nullptr, nullptr, nullptr, nullptr, nullptr, false);

  setLoopAlreadyCoarsened(L);



  return LoopUnrollResult::PartiallyUnrolled;
}

PreservedAnalyses
LoopUnrollAndInterleavePass::run(Loop &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &Updater) {
  OptimizationRemarkEmitter ORE(L.getHeader()->getParent());

  bool Changed = tryToUnrollLoop(&L, AR.DT, &AR.LI, AR.SE, AR.TTI, AR.AC, ORE,
                                 /*BFI*/ nullptr, /*PSI*/ nullptr) !=
                 LoopUnrollResult::Unmodified;
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
