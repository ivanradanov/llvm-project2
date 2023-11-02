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
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
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
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-and-interleave"

class LoopUnrollAndInterleave {
private:
  OptimizationRemarkEmitter &ORE;
  LoopInfo *LI;
  DominatorTree *DT;
  PostDominatorTree *PDT;
  Loop *TheLoop;

  // Loop data
  BasicBlock *CombinedLatchExiting;

  struct MergedDivergentRegion {
    SmallPtrSet<BasicBlock *, 8> Blocks;
  };

  /// Divergent groups are the maximal sets of basic blocks with the following
  /// properties:
  ///
  /// * There is a divergent branch from a block outside the region (From) to a
  ///   block (Entry) in the set
  ///
  /// * All blocks in the set are post-dominated by a single exiting block (To)
  ///
  /// (We assume a single combined exiting/latch block)
  struct DivergentRegion {
    // Outside the group
    BasicBlock *From;
    BasicBlock *To;

    // Entry block in the region, jumped to from From
    BasicBlock *Entry;
    // Exit block from the region, jumps to To
    BasicBlock *Exit;
    // The blocks in the divergent region
    SmallPtrSet<BasicBlock *, 8> Blocks;
  };

  SmallPtrSet<Instruction *, 8> DivergentBranches;
  // A DivergentRegion is uniquely identified by the convergent to divergent
  // edge
  SmallPtrSet<BranchInst *, 8> ConvergentToDivergentEdges;
  // Whereas a MergedDivergentRegion is uniquely identified by the divergent to
  // convergent edge
  SmallPtrSet<BranchInst *, 8> DivergentToConvergentEdges;
  SmallVector<DivergentRegion> DivergentRegions;
  SmallVector<MergedDivergentRegion> MergedDivergentRegions;
  SmallPtrSet<BasicBlock *, 8> ConvergingBlocks;

  void populateDivergentRegions();
  bool isLegalToCoarsen(Loop *TheLoop, LoopInfo *LI);

public:
  LoopUnrollAndInterleave(OptimizationRemarkEmitter &ORE) : ORE(ORE) {}
  LoopUnrollResult tryToUnrollLoop(Loop *L, DominatorTree &DT, LoopInfo *LI,
                                   ScalarEvolution &SE,
                                   const TargetTransformInfo &TTI,
                                   PostDominatorTree &PDT);
};

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

#define DBGS llvm::dbgs() << "LUAI: "
#define DBGS_FAIL llvm::dbgs() << "LUAI: FAIL: "

#pragma push_macro("ILLEGAL")
#define ILLEGAL()                                                              \
  do {                                                                         \
    if (DoExtraAnalysis)                                                       \
      Result = false;                                                          \
    else                                                                       \
      return false;                                                            \
  } while (0)

bool LoopUnrollAndInterleave::isLegalToCoarsen(Loop *TheLoop, LoopInfo *LI) {
  const bool DoExtraAnalysis = true;
  bool Result = true;
  for (BasicBlock *BB : TheLoop->blocks()) {
    // Check whether the BB terminator is a BranchInst. Any other terminator is
    // not supported yet.
    auto *Term = BB->getTerminator();
    auto *Br = dyn_cast<BranchInst>(Term);
    auto *Sw = dyn_cast<SwitchInst>(Term);
    if (!Br && !Sw) {
      LLVM_DEBUG(DBGS_FAIL << "Unsupported basic block terminator" << *Term
                           << "\n");
      ILLEGAL();
    }

    // TODO we need to check for syncs and divergent branches, because I think
    // they have legality implications
    //
    // TODO we can do better if we know there is no
    // synchronisation as we do not need to care about stores from other
    // iterations
    if (Br && Br->isConditional() &&
        !TheLoop->isLoopInvariant(Br->getCondition()) &&
        !LI->isLoopHeader(Br->getSuccessor(0)) &&
        !LI->isLoopHeader(Br->getSuccessor(1))) {
      DivergentBranches.insert(Term);
      LLVM_DEBUG(DBGS << "Divergent branch found:" << *Br << "\n");
    }
    if (Sw && !TheLoop->isLoopInvariant(Sw->getCondition())) {
      DivergentBranches.insert(Term);
      LLVM_DEBUG(DBGS << "Divergent switch found:" << *Sw << "\n");
    }
  }

  CombinedLatchExiting = TheLoop->getExitingBlock();
  if (CombinedLatchExiting != TheLoop->getLoopLatch()) {
    LLVM_DEBUG(DBGS << "Expected a combined exiting and latch block\n");
    ILLEGAL();
  }

  return Result;
}
#pragma pop_macro("ILLEGAL")

template <typename T, typename S> static bool contains(S &C, T A) {
  return std::find(C.begin(), C.end(), A) != C.end();
}

static void findReachableFromTo(BasicBlock *From, BasicBlock *To,
                                SmallPtrSetImpl<BasicBlock *> &Reachable) {
  std::queue<BasicBlock *> Queue;
  Queue.push(From);
  Reachable.insert(From);
  while (!Queue.empty()) {
    BasicBlock *SrcBB = Queue.front();
    Queue.pop();
    for (BasicBlock *DstBB : children<BasicBlock *>(SrcBB)) {
      if (Reachable.insert(DstBB).second && DstBB != To)
        Queue.push(DstBB);
    }
  }
}

void LoopUnrollAndInterleave::populateDivergentRegions() {
  for (Instruction *Term : DivergentBranches) {
    BasicBlock *Entry = Term->getParent();
    auto *ConvergeBlock = PDT->getNode(Entry)->getIDom()->getBlock();
    assert(ConvergeBlock &&
           PDT->dominates(CombinedLatchExiting, ConvergeBlock));
    // Split the entry blocks to have the first part be convergent and then
    // diverge in the second part
    auto *From = Entry->splitBasicBlockBefore(Entry->getTerminator());
    From->setName(Entry->getName() + ".divergent.preentry.split");
    TheLoop->addBasicBlockToLoop(From, *LI);
    if (TheLoop->getHeader() == Entry)
      TheLoop->moveToHeader(From);

    SmallPtrSet<BasicBlock *, 8> Reachable;
    findReachableFromTo(Entry, ConvergeBlock, Reachable);

    Reachable.erase(ConvergeBlock);

    DivergentRegion Region = {From, ConvergeBlock, Entry,
                              /* We will insert the Exit later */ nullptr,
                              std::move(Reachable)};
    DivergentRegions.push_back(Region);
  }

  // Split the converging blocks to have the first part be divergent and
  // reconverg in the second part
  for (auto &DR : DivergentRegions) {
    BasicBlock *LastDivergent;
    if (ConvergingBlocks.contains(DR.To)) {
      auto Preds = predecessors(DR.To);
      assert(std::next(Preds.begin()) == Preds.end());
      LastDivergent = *Preds.begin();
    } else {
      LastDivergent = DR.To->splitBasicBlockBefore(DR.To->getFirstNonPHI());
      LastDivergent->setName(DR.To->getName() + ".divergent.exit.split");
      TheLoop->addBasicBlockToLoop(LastDivergent, *LI);
      DivergentToConvergentEdges.insert(
          cast<BranchInst>(LastDivergent->getTerminator()));
    }
    DR.Blocks.insert(LastDivergent);
    DR.Exit = LastDivergent;
  }

  LLVM_DEBUG({
    for (auto &DR : DivergentRegions) {
      DBGS << "Divergent region for entry %" << DR.Entry->getName()
           << " from block %" << DR.From->getName() << " to block %"
           << DR.To->getName() << ":\n";
      for (auto *BB : DR.Blocks)
        dbgs() << "%" << BB->getName() << ", ";
      dbgs() << "\n";
    }
  });

  // Merge overlapping divergent regions so as to generate them only once
  for (auto &DR : DivergentRegions) {
    // TODO Performance of this isnt very good
    bool Found = false;
    for (auto &MDR : MergedDivergentRegions) {
      if (!set_intersection(DR.Blocks, MDR.Blocks).empty()) {
        MDR.Blocks.insert(DR.Blocks.begin(), DR.Blocks.end());
        Found = true;
        break;
      }
    }
    if (!Found) {
      MergedDivergentRegion MDR = {DR.Blocks};
      MergedDivergentRegions.push_back(std::move(MDR));
    }
  }
  LLVM_DEBUG({
    DBGS << "Merged divergent regions (" << MergedDivergentRegions.size()
         << "):\n";
    for (auto &MDR : MergedDivergentRegions) {
      dbgs() << "Region: ";
      for (auto *BB : MDR.Blocks)
        dbgs() << "%" << BB->getName() << ", ";
      dbgs() << "\n";
    }
  });

  for (auto *DB : DivergentBranches) {
    auto Preds = predecessors(DB->getParent());
    assert(std::next(Preds.begin()) == Preds.end());
    ConvergentToDivergentEdges.insert(
        cast<BranchInst>((*Preds.begin())->getTerminator()));
  }
}

LoopUnrollResult LoopUnrollAndInterleave::tryToUnrollLoop(
    Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
    const TargetTransformInfo &TTI, PostDominatorTree &PDT) {
  this->LI = LI;
  this->TheLoop = L;
  this->DT = &DT;
  this->PDT = &PDT;

  // Disabled by default
  unsigned UnrollFactor = 1;
  if (char *Env = getenv("UNROLL_AND_INTERLEAVE_FACTOR"))
    StringRef(Env).getAsInteger(10, UnrollFactor);
  assert(UnrollFactor > 0);
  if (UnrollFactor == 1) {
    LLVM_DEBUG(DBGS << "Unroll factor of 1 - ignoring\n");
    return LoopUnrollResult::Unmodified;
  }

  // TODO currently doesnt work
  bool Chunkify = false;
  if (char *Env = getenv("UNROLL_AND_INTERLEAVE_CHUNKIFY")) {
    unsigned Int = 0;
    StringRef(Env).getAsInteger(10, Int);
    Chunkify = Int;
  }

  Function *F = L->getHeader()->getParent();
  auto &Ctx = F->getContext();
  LLVM_DEBUG(DBGS << "F[" << F->getName() << "] Loop %"
                  << L->getHeader()->getName() << " "
                  << "Parallel=" << L->isAnnotatedParallel() << "\n");

  if (getLoopAlreadyCoarsened(L)) {
    LLVM_DEBUG(DBGS_FAIL << "Already coarsened\n");
    return LoopUnrollResult::Unmodified;
  }

  if (!isWorkShareLoop(L)) {
    LLVM_DEBUG(DBGS_FAIL << "Not work share loop\n");
    return LoopUnrollResult::Unmodified;
  }

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP"))
    LLVM_DEBUG(DBGS << "Before unroll and interleave:\n" << *F);

  if (!isLegalToCoarsen(L, LI))
    return LoopUnrollResult::Unmodified;

  auto LoopBounds = L->getBounds(SE);
  if (LoopBounds == std::nullopt) {
    LLVM_DEBUG(DBGS_FAIL << "Unable to find loop bounds of the omp workshare "
                            "loop, not coarsening\n");
    return LoopUnrollResult::Unmodified;
  }

  populateDivergentRegions();

  // TODO Pretty bad, SE is also invalidated but I /think/ we dont need it any
  // more
  DT.recalculate(*F);

  // TODO handle convergent insts properly (e.g. __syncthreads())

  // Save loop properties before it is transformed.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    // We delete the preheader of the epilogue loop so this is currently how we
    // detect that this may be the epilogue loop, because all other loops should
    // have a preheader after being simplified before this pass
    LLVM_DEBUG(DBGS_FAIL << "No preheader\n");
    return LoopUnrollResult::Unmodified;
  }
  BasicBlock *ExitBlock = L->getExitBlock();
  std::vector<BasicBlock *> OriginalLoopBlocks = L->getBlocks();

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

  // We need the upper bound of the loop to be defined before we enter it in
  // order to know whether we should enter the coarsened version or the
  // epilogue.
  Value *End = [&](Value *End) -> Value * {
    Instruction *I = dyn_cast<LoadInst>(End);
    if (!I)
      return End;

    BasicBlock *BB = I->getParent();
    if (BB == Preheader ||
        std::find(OriginalLoopBlocks.begin(), OriginalLoopBlocks.end(), BB) ==
            OriginalLoopBlocks.end())
      return End;

    return nullptr;
  }(&LoopBounds->getFinalIVValue());
  if (End == nullptr) {
    LLVM_DEBUG(DBGS_FAIL << "Unusable FinalIVValue define in the loop\n");
    return LoopUnrollResult::Unmodified;
  }

  Value *InitialIVVal = &LoopBounds->getInitialIVValue();
  Instruction *InitialIVInst = dyn_cast<Instruction>(InitialIVVal);
  if (!InitialIVInst) {
    LLVM_DEBUG(DBGS_FAIL << "Unexpected initial val definition" << *InitialIVVal
                         << "\n");
    return LoopUnrollResult::Unmodified;
  }

  // Clone the loop to use as an epilogue, the original one will be coarsened
  // in-place
  ValueToValueMapTy EpilogueVMap;
  SmallVector<BasicBlock *> EpilogueLoopBlocks;
  Loop *EpilogueLoop =
      cloneLoopWithPreheader(ExitBlock, Preheader, L, EpilogueVMap, ".epilogue",
                             LI, &DT, EpilogueLoopBlocks);
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

  BasicBlock *EpiloguePH = cast<BasicBlock>(EpilogueVMap[Preheader]);
  EpilogueLoopBlocks.erase(std::find(EpilogueLoopBlocks.begin(),
                                     EpilogueLoopBlocks.end(), EpiloguePH));
  for (Instruction &I : *Preheader) {
    EpilogueVMap.erase(&I);
  }
  EpilogueVMap.erase(Preheader);
  remapInstructionsInBlocks(EpilogueLoopBlocks, EpilogueVMap);

  // Find all values used in the divergent region but defined outside and insert
  // explicit phi nodes that bring in the values in the Entry block
  // TODO
  // insertMDRIncomingPHINodes(EpilogueLoop, EpilogueVMap,
  // MergedDivergentRegions);

  for (auto *BI : ConvergentToDivergentEdges) {
    assert(!BI->isConditional());

    BasicBlock *Entry = BI->getSuccessor(0);
    BasicBlock *EpilogueEntry =
        cast<BasicBlock>(EpilogueVMap[BI->getSuccessor(0)]);

    auto &DR = *std::find_if(DivergentRegions.begin(), DivergentRegions.begin(),
                             [&](auto &DR) { return DR.Entry == Entry; });

    BasicBlock *EpilogueExit = cast<BasicBlock>(EpilogueVMap[DR.Exit]);
    BasicBlock *EpilogueTo = cast<BasicBlock>(EpilogueVMap[DR.To]);
    BasicBlock *EpilogueFrom = cast<BasicBlock>(EpilogueVMap[DR.From]);

    for (auto *BB : DR.Blocks) {
      auto *EpilogueBB = cast<BasicBlock>(EpilogueVMap[BB]);
      (void)EpilogueBB;
    }

    Type *CoarsenedIdentifierTy = IntegerType::getInt32Ty(Ctx);
    PHINode *IsFromCoarsened = PHINode::Create(
        CoarsenedIdentifierTy, /*NumReservedValues=*/UnrollFactor + 1,
        "is.from.coarsened", EpilogueEntry->getFirstNonPHI());
    IsFromCoarsened->addIncoming(ConstantInt::get(CoarsenedIdentifierTy, -1),
                                 EpilogueFrom);

    // Multiple DivergentRegion entries may converge at the same location,
    // create the exit switch only once
    auto *ToOutroSw = dyn_cast<SwitchInst>(EpilogueExit->getTerminator());
    int NumSwCases;
    if (!ToOutroSw) {
      ToOutroSw = SwitchInst::Create(IsFromCoarsened, EpilogueTo, 0);
      EpilogueExit->getTerminator()->eraseFromParent();
      ToOutroSw->insertInto(EpilogueExit, EpilogueExit->end());
      NumSwCases = 0;
    } else {
      NumSwCases = ToOutroSw->getNumCases();
    }

    for (unsigned It = 0; It < UnrollFactor; It++) {
      auto *Intro = BasicBlock::Create(
          Ctx, EpilogueEntry->getName() + ".div.intro." + std::to_string(It), F,
          EpilogueEntry);
      auto *FromIntroBI = BranchInst::Create(EpilogueEntry);
      FromIntroBI->insertInto(Intro, Intro->end());
      auto *ToIntroBI = BranchInst::Create(Intro);
      DR.From->getTerminator()->eraseFromParent();
      ToIntroBI->insertInto(DR.From, DR.From->end());

      auto *ThisFactorIdentifier = cast<ConstantInt>(
          ConstantInt::get(CoarsenedIdentifierTy, NumSwCases + It));
      IsFromCoarsened->addIncoming(ThisFactorIdentifier, Intro);

      // TODO Handle entry PHIs

      auto *Outro = BasicBlock::Create(
          Ctx,
          EpilogueExit->getName() + ".div.outro." + std::to_string(It) + "." +
              std::to_string(NumSwCases / UnrollFactor),
          F, EpilogueTo);
      auto *FromOutroBI = BranchInst::Create(DR.To);
      FromOutroBI->insertInto(Outro, Outro->end());

      ToOutroSw->addCase(ThisFactorIdentifier, Outro);
    }
  }

  // TODO Remove
  LLVM_DEBUG(DBGS << "After dr intro/outro:\n" << *F);

  // Plumbing around the coarsened and epilogue loops

  for (BasicBlock::iterator I = ExitBlock->begin(); isa<PHINode>(I);) {
    PHINode *PN = cast<PHINode>(I++);

    Value *IncomingVal = PN->getIncomingValueForBlock(CombinedLatchExiting);
    PN->addIncoming(EpilogueVMap[IncomingVal],
                    cast<BasicBlock>(EpilogueVMap[CombinedLatchExiting]));
  }

  // Calc the start value for the epilogue loop, should be:
  // Start + (ceil((End - Start) / Stride) / UnrollFactor) * UnrollFactor *
  // Stride.
  // I.e. when we are done with our iterations of the coarsened loop
  Value *EpilogueStart;
  Value *Start = &LoopBounds->getInitialIVValue();
  // Value *End = GetEnd(&LoopBounds->getFinalIVValue()); // Defined above.
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
    BranchInst *BackEdge =
        dyn_cast<BranchInst>(CombinedLatchExiting->getTerminator());
    assert(BackEdge && BackEdge->isConditional());
    BasicBlock *PrevBB = CombinedLatchExiting->splitBasicBlockBefore(BackEdge);
    Instruction *BI = PrevBB->getTerminator();
    IRBuilder<> Builder(BI);
    Value *IsAtEpilogueStart = Builder.CreateCmp(
        CmpInst::Predicate::ICMP_EQ, &LoopBounds->getStepInst(), EpilogueStart,
        "is.epilogue.start");
    Value *EpilogueBranch = Builder.CreateCondBr(
        IsAtEpilogueStart, EpilogueLoop->getHeader(), CombinedLatchExiting);
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

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP"))
    LLVM_DEBUG(DBGS << "After unroll and interleave:\n" << *F);

  setLoopAlreadyCoarsened(L);

  LLVM_DEBUG(DBGS << "SUCCESS\n");

  return LoopUnrollResult::PartiallyUnrolled;
}

PreservedAnalyses
LoopUnrollAndInterleavePass::run(Loop &L, LoopAnalysisManager &AM,
                                 LoopStandardAnalysisResults &AR,
                                 LPMUpdater &Updater) {
  auto *F = L.getHeader()->getParent();
  OptimizationRemarkEmitter ORE(F);

  // auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(L, AR);
  auto PDT = PostDominatorTree(*F);
  bool Changed = LoopUnrollAndInterleave(ORE).tryToUnrollLoop(
                     &L, AR.DT, &AR.LI, AR.SE, AR.TTI, PDT) !=
                 LoopUnrollResult::Unmodified;
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
