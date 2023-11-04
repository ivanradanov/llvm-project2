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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
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
#include "llvm/IR/DerivedTypes.h"
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
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-and-interleave"

template <typename OutTy, typename CastTy, typename InTy>
static OutTy mapContainer(InTy &Container, ValueToValueMapTy &VMap) {
  auto MappedRange = llvm::map_range(
      Container, [&VMap](Value *V) { return cast<CastTy>(VMap[V]); });
  OutTy Mapped(MappedRange.begin(), MappedRange.end());
  return std::move(Mapped);
}

class LoopUnrollAndInterleave {
private:
  OptimizationRemarkEmitter &ORE;
  LoopInfo *LI;
  DominatorTree *DT;
  PostDominatorTree *PDT;

  // Loop data
  Loop *TheLoop;
  BasicBlock *CombinedLatchExiting;
  BasicBlock *Preheader;

  // Options
  unsigned UnrollFactor;
  bool UseDynamicConvergence = false;

  struct MergedDivergentRegion {
    // Populated at the start
    SmallPtrSet<BasicBlock *, 8> Entries;
    SmallPtrSet<BasicBlock *, 8> Exits;
    SmallPtrSet<BasicBlock *, 8> Blocks;
  };

  /// Divergent groups are the maximal sets of basic blocks with the following
  /// properties:
  ///
  /// * There is a divergent branch from a block outside the region (From) to a
  ///   block (Entry) in the set
  /// * There is a path from the Entry to all blocks in the set
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

    MergedDivergentRegion *MDR;

    // Used for transformations later
    std::unique_ptr<ValueToValueMapTy> DefinedOutsideDemotedVMap;
    std::unique_ptr<ValueToValueMapTy> DefinedInsideDemotedVMap;
  };

  SmallPtrSet<Instruction *, 8> DemotedRegs;
  ValueToValueMapTy DemotedRegsVMap;

  SmallPtrSet<Instruction *, 8> DivergentBranches;
  // A DivergentRegion is uniquely identified by the convergent to divergent
  // edge
  SmallPtrSet<BranchInst *, 8> ConvergentToDivergentEdges;
  SmallPtrSet<BranchInst *, 8> DivergentToConvergentEdges;
  SmallVector<DivergentRegion, 0> DivergentRegions;
  SmallVector<MergedDivergentRegion, 0> MergedDivergentRegions;

  void demoteDRRegs(Loop *NCL, ValueToValueMapTy &VMap, Loop *CL);
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

  SmallPtrSet<BasicBlock *, 8> ConvergingBlocks;
  SmallPtrSet<BasicBlock *, 8> EntryBlocks;

  for (Instruction *Term : DivergentBranches) {
    BasicBlock *Entry = Term->getParent();
    auto *ConvergeBlock = PDT->getNode(Entry)->getIDom()->getBlock();
    assert(ConvergeBlock &&
           PDT->dominates(CombinedLatchExiting, ConvergeBlock));

    SmallPtrSet<BasicBlock *, 8> Reachable;
    findReachableFromTo(Entry, ConvergeBlock, Reachable);

    // We will insert the Entry and Exit later
    DivergentRegion Region = {Entry, ConvergeBlock, nullptr, nullptr,
                              std::move(Reachable)};
    ConvergingBlocks.insert(ConvergeBlock);
    EntryBlocks.insert(Entry);
    DivergentRegions.push_back(std::move(Region));
  }

  SmallPtrSet<BasicBlock *, 8> EntryAndConvergingBlocks =
      set_intersection(ConvergingBlocks, EntryBlocks);

  // Split blocks that are both an entry and an exit to a DR in this way:
  // DivergentExit -> Convergent -> DivergentEntry
  // Where we will later insert insert jumps to/from a DR at the two new edges.
  for (auto *TheBlock : EntryAndConvergingBlocks) {
    std::string BlockName = TheBlock->getName().str();
    auto *DivergentEntry = TheBlock;
    auto *Convergent =
        DivergentEntry->splitBasicBlockBefore(DivergentEntry->getTerminator());
    auto *DivergentExit =
        Convergent->splitBasicBlockBefore(Convergent->getFirstNonPHI());
    DivergentExit->setName(BlockName + ".divergent.exit");
    DivergentEntry->setName(BlockName + ".divergent.entry");
    Convergent->setName(BlockName);

    TheLoop->addBasicBlockToLoop(DivergentExit, *LI);
    TheLoop->addBasicBlockToLoop(Convergent, *LI);
    assert(TheLoop->getHeader() != TheBlock &&
           "A Converging block cannot be the header.");

    for (auto &DR : llvm::make_filter_range(
             DivergentRegions,
             [TheBlock](DivergentRegion &DR) { return DR.To == TheBlock; })) {
      DR.Blocks.erase(DR.To);
      DR.Blocks.insert(DivergentExit);
      DR.Exit = DivergentExit;
      DR.To = Convergent;
    }
    auto *DR = find_if(DivergentRegions, [TheBlock](DivergentRegion &DR) {
      return DR.From == TheBlock;
    });
    assert(DR);
    DR->From = Convergent;
    DR->Entry = DivergentEntry;
  }

  for (auto *TheBlock : EntryBlocks) {
    // Already handled
    if (EntryAndConvergingBlocks.contains(TheBlock))
      continue;
    std::string BlockName = TheBlock->getName().str();
    auto *DivergentEntry = TheBlock;
    auto *Convergent =
        DivergentEntry->splitBasicBlockBefore(DivergentEntry->getTerminator());
    DivergentEntry->setName(BlockName + ".divergent.entry");
    Convergent->setName(BlockName);

    TheLoop->addBasicBlockToLoop(Convergent, *LI);
    if (TheLoop->getHeader() == TheBlock)
      TheLoop->moveToHeader(Convergent);

    auto *DR = find_if(DivergentRegions, [TheBlock](DivergentRegion &DR) {
      return DR.From == TheBlock;
    });
    assert(DR);
    DR->From = Convergent;
    DR->Entry = DivergentEntry;
  }

  for (auto *TheBlock : ConvergingBlocks) {
    // Already handled
    if (EntryAndConvergingBlocks.contains(TheBlock))
      continue;
    std::string BlockName = TheBlock->getName().str();
    auto *Convergent = TheBlock;
    auto *DivergentExit =
        Convergent->splitBasicBlockBefore(Convergent->getFirstNonPHI());
    Convergent->setName(BlockName);
    DivergentExit->setName(BlockName + ".divergent.exit");

    TheLoop->addBasicBlockToLoop(DivergentExit, *LI);

    for (auto &DR : llvm::make_filter_range(
             DivergentRegions,
             [TheBlock](DivergentRegion &DR) { return DR.To == TheBlock; })) {
      DR.Blocks.erase(DR.To);
      DR.Blocks.insert(DivergentExit);
      DR.Exit = DivergentExit;
      DR.To = Convergent;
    }
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
        DR.MDR = &MDR;
        MDR.Entries.insert(DR.Entry);
        MDR.Exits.insert(DR.Exit);
        MDR.Blocks.insert(DR.Blocks.begin(), DR.Blocks.end());
        Found = true;
        break;
      }
    }
    if (!Found) {
      MergedDivergentRegion MDR = {{DR.Entry}, {DR.Exit}, DR.Blocks};
      MergedDivergentRegions.push_back(std::move(MDR));
      DR.MDR = &MergedDivergentRegions.back();
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

void LoopUnrollAndInterleave::demoteDRRegs(Loop *NCL, ValueToValueMapTy &VMap,
                                            Loop *CL) {

  // Demote values defined outside a DR used inside it in the Non Coarsened Loop
  // (NCL)
  for (auto &DR : DivergentRegions) {
    auto MappedMDRBlocks =
        mapContainer<SmallPtrSet<BasicBlock *, 8>, BasicBlock>(DR.Blocks,
                                                               VMap);

    // Find values defined outside the DR and used inside it
    DR.DefinedOutsideDemotedVMap.reset(new ValueToValueMapTy());
    for (auto *BB : NCL->getBlocks()) {
      if (MappedMDRBlocks.contains(BB))
        continue;

      for (auto &I : *BB) {
        if (I.use_empty())
          continue;
        for (auto *User : I.users()) {
          auto *UserI = dyn_cast<Instruction>(User);
          if (!UserI)
            continue;
          if (MappedMDRBlocks.contains(UserI->getParent())) {
            auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[&I]);
            if (!Demoted) {
              Demoted = DemoteRegToStack(
                  I, /*VolatileLoads=*/false, Preheader->getFirstNonPHI());
              DemotedRegsVMap[&I] = Demoted;
            }
            (*DR.DefinedOutsideDemotedVMap)[&I] = Demoted;
          }
        }
      }
    }
  }

  // Demote values defined inside a DR and used outside it in the Coarsened Loop
  // (CL)
  for (auto &DR : DivergentRegions) {
    // Find values defined outside the DR and used inside it
    DR.DefinedInsideDemotedVMap.reset(new ValueToValueMapTy());
    for (auto *BB : DR.Blocks) {
      for (auto &I : *BB) {
        if (I.use_empty())
          continue;
        for (auto *User : I.users()) {
          auto *UserI = dyn_cast<Instruction>(User);
          if (!UserI)
            continue;
          BasicBlock *UserBB = UserI->getParent();
          if (DR.Blocks.contains(UserBB))
            continue;
          auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[&I]);
          if (!Demoted) {
            Demoted = DemoteRegToStack(
                I, /*VolatileLoads=*/false, Preheader->getFirstNonPHI());
            DemotedRegsVMap[&I] = Demoted;
          }
          (*DR.DefinedInsideDemotedVMap)[&I] = Demoted;
        }
      }
    }
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
  UnrollFactor = 1;
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
  LLVM_DEBUG(DBGS << "After populateDivergentRegions:\n" << *F);

  // TODO Pretty bad, SE is also invalidated but I /think/ we dont need it any
  // more
  DT.recalculate(*F);

  // TODO handle convergent insts properly (e.g. __syncthreads())

  // Save loop properties before it is transformed.
  Preheader = L->getLoopPreheader();
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

  // Clone the loop to use the blocks for divergent regions
  ValueToValueMapTy DivergentRegionsVMap;
  SmallVector<BasicBlock *> DivergentRegionsLoopBlocks;
  Loop *DivergentRegionsLoop = cloneLoopWithPreheader(
      ExitBlock, &F->getEntryBlock(), L, DivergentRegionsVMap, ".drs", LI, &DT,
      DivergentRegionsLoopBlocks);
  remapInstructionsInBlocks(DivergentRegionsLoopBlocks, DivergentRegionsVMap);

  // Clone the loop once more to use as an epilogue, the original one will be
  // coarsened in-place
  ValueToValueMapTy EpilogueVMap;
  SmallVector<BasicBlock *> EpilogueLoopBlocks;
  Loop *EpilogueLoop =
      cloneLoopWithPreheader(ExitBlock, &F->getEntryBlock(), L, EpilogueVMap,
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
  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> ReverseVMaps;
  VMaps.reserve(UnrollFactor);
  ReverseVMaps.reserve(UnrollFactor);
  for (unsigned I = 0; I < UnrollFactor; I++) {
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
    ReverseVMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
  }

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
  (*VMaps[0])[InitialIVVal] = InitialIVVal;
  (*ReverseVMaps[0])[InitialIVVal] = InitialIVVal;
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
    (*ReverseVMaps[I])[CoarsenedInitialIV] = InitialIVVal;
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
      (*VMaps[0])[I] = I;
      (*ReverseVMaps[0])[I] = I;
      for (unsigned It = 1; It < UnrollFactor; It++) {
        if (I->isTerminator()) {
          if (UseDynamicConvergence) {
            // TODO here we would check if all original iterations agree on the
            // next block and take the coarsened branch if they do. Currently
            // unsupported
            assert(0 && "Unsupported");
          } else {
            // Do not clone terminators - we use the control flow of the
            // existing iteration (if the branch is divergent we will insert
            // an entry to a divergent region here later)
            continue;
          }
        }

        Instruction *Cloned = I->clone();
        Cloned->insertAfter(LastI);
        if (!Cloned->getType()->isVoidTy())
          Cloned->setName(I->getName() + ".coarsened." + std::to_string(It));
        ClonedInsts[It].push_back(Cloned);
        (*VMaps[It])[I] = Cloned;
        (*ReverseVMaps[It])[Cloned] = I;
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

  DT.recalculate(*F); // TODO another recalculation...
  // Find all values used in the divergent region but defined outside and demote
  // them to memory - now we can change the CFG easier.
  demoteDRRegs(DivergentRegionsLoop, DivergentRegionsVMap, TheLoop);

  ValueToValueMapTy ReverseDRLVMap;
  for (auto M : DivergentRegionsVMap) {
    ReverseDRLVMap[cast<Value>(M.second)] = const_cast<Value *>(M.first);
  }

  Type *CoarsenedIdentifierTy = IntegerType::getInt32Ty(Ctx);
  auto *CoarsenedIdentPtr = PreheaderBuilder.CreateAlloca(
      CoarsenedIdentifierTy, /*ArraySize*/ nullptr, "coarsened.ident");

  // Hook into the the blocks from the DR loop to do the divergent part of the
  // coarsened computation. Generates Intro and Outro blocks which bring in the
  // appropriate coarsened values into the DR in the uncoarsened part, and then
  // bring out those values for use in the subsequent coarsened computation.
  for (auto &DR : DivergentRegions) {
    BasicBlock *DREntry = cast<BasicBlock>(DivergentRegionsVMap[DR.Entry]);
    BasicBlock *DRExit = cast<BasicBlock>(DivergentRegionsVMap[DR.Exit]);
    BasicBlock *DRTo = cast<BasicBlock>(DivergentRegionsVMap[DR.To]);
    BasicBlock *DRFrom = cast<BasicBlock>(DivergentRegionsVMap[DR.From]);

    new StoreInst(ConstantInt::get(CoarsenedIdentifierTy, -1),
                  CoarsenedIdentPtr, DRFrom->getTerminator());

    // Multiple DivergentRegion entries may converge at the same location,
    // create the exit switch only once
    auto *ToOutroSw = dyn_cast<SwitchInst>(DRExit->getTerminator());
    int NumSwCases;
    if (!ToOutroSw) {
      auto *Ld = new LoadInst(CoarsenedIdentifierTy, CoarsenedIdentPtr,
                              "coarsene.ident.load", DRExit->getTerminator());
      // We will override the default dest later, set something random
      ToOutroSw = SwitchInst::Create(Ld, DREntry, 0);
      DRExit->getTerminator()->eraseFromParent();
      ToOutroSw->insertInto(DRExit, DRExit->end());
    }
    NumSwCases = ToOutroSw->getNumCases();

    BasicBlock *LastOutro = nullptr;
    for (unsigned It = 0; It < UnrollFactor; It++) {
      auto *ThisFactorIdentifier = cast<ConstantInt>(
          ConstantInt::get(CoarsenedIdentifierTy, NumSwCases + It));

      auto *Intro = BasicBlock::Create(
          Ctx, DREntry->getName() + ".div.intro." + std::to_string(It), F,
          DREntry);
      IRBuilder<> IntroBuilder(Intro);
      IntroBuilder.CreateStore(ThisFactorIdentifier, CoarsenedIdentPtr);

      auto *ToIntroBI = BranchInst::Create(Intro);
      if (It == 0) {
        assert(!LastOutro);
        DR.From->getTerminator()->eraseFromParent();
        ToIntroBI->insertInto(DR.From, DR.From->end());
      } else {
        ToIntroBI->insertInto(LastOutro, LastOutro->end());
      }

      for (auto P : *DR.DefinedOutsideDemotedVMap) {
        auto *I = P.first;
        auto *OriginalValue = cast<Instruction>(ReverseDRLVMap[I]);
        Instruction *CoarsenedValue =
            cast<Instruction>((*VMaps[It])[OriginalValue]);
        IntroBuilder.CreateStore(CoarsenedValue, P.second);
      }

      IntroBuilder.CreateBr(DREntry);

      auto *Outro = LastOutro = BasicBlock::Create(
          Ctx,
          DRExit->getName() + ".div.outro." + std::to_string(It) + "." +
              std::to_string(NumSwCases / UnrollFactor),
          F, DRTo);
      for (auto P : *DR.DefinedInsideDemotedVMap) {
        auto *CoarsenedValue = P.first;
        auto *OriginalValue =
            cast_or_null<Instruction>((*ReverseVMaps[It])[CoarsenedValue]);
        // When the defined inside value is from another original iteration
        if (!OriginalValue)
          continue;
        auto *DRValue = cast<Instruction>(DivergentRegionsVMap[OriginalValue]);
        new StoreInst(DRValue, P.second, Outro);
      }
      if (It == UnrollFactor - 1) {
        auto *FromOutroBI = BranchInst::Create(DR.To);
        FromOutroBI->insertInto(Outro, Outro->end());
      }

      if (It == 0)
        ToOutroSw->setDefaultDest(Outro);
      else
        ToOutroSw->addCase(ThisFactorIdentifier, Outro);
    }
  }
  LLVM_DEBUG(DBGS << "1:\n" << *F);
  for (auto *DR : DemotedRegs) {
    auto *AI = cast<AllocaInst>(DemotedRegsVMap[DR]);
    DBGS << "demoted ai:" << *AI << "\n";
  }

  // Now that we have done the plumbing around the divergent regions loop, erase
  // the remainders
  SmallPtrSet<BasicBlock *, 8> NotUsedDRBlocks(
      DivergentRegionsLoopBlocks.begin(), DivergentRegionsLoopBlocks.end());
  for (auto &DR : DivergentRegions) {
    auto MappedRange =
        llvm::map_range(DR.Blocks, [&DivergentRegionsVMap](BasicBlock *BB) {
          return cast<BasicBlock>(DivergentRegionsVMap[BB]);
        });
    SmallPtrSet<BasicBlock *, 8> MappedMDRBlocks(MappedRange.begin(),
                                                 MappedRange.end());
    set_subtract(NotUsedDRBlocks, MappedMDRBlocks);
  }
  {
    SmallVector<BasicBlock *> Tmp(NotUsedDRBlocks.begin(),
                                  NotUsedDRBlocks.end());
    DeleteDeadBlocks(Tmp);
  }
  LLVM_DEBUG(DBGS << "3:\n" << *F);
  for (auto *DR : DemotedRegs) {
    auto *AI = cast<AllocaInst>(DemotedRegsVMap[DR]);
    DBGS << "demoted ai:" << *AI << "\n";
  }

  // If we do not use dynamic convergence then the coarsened versions of the DRs
  // are unused and we have to delete them
  // TODO not sure if we can have convergent flow go into a DR and exit through
  // its exit - I think it is possible and we have to handle that case (not only
  // here, but in the DR plumbing that we do around the exits above as well
  if (!UseDynamicConvergence) {
    SmallPtrSet<BasicBlock *, 8> ToDelete;
    for (auto &DR : DivergentRegions) {
      set_union(ToDelete, DR.Blocks);
    }
    SmallVector<BasicBlock *> Tmp(ToDelete.begin(), ToDelete.end());
    DeleteDeadBlocks(Tmp);
  }

  LLVM_DEBUG(DBGS << "4:\n" << *F);
  for (auto *DR : DemotedRegs) {
    auto *AI = cast<AllocaInst>(DemotedRegsVMap[DR]);
    DBGS << "demoted ai:" << *AI << "\n";
  }

  // Now that we are done with the aggressive CFG restructuring we can
  // re-promote the regs we demoted earlier
  // TODO can we check we were able to promote everything without undefs?
  DT.recalculate(*F); // TODO another recalculation...
  PromoteMemToReg(mapContainer<SmallVector<AllocaInst *>, AllocaInst>(
                      DemotedRegs, DemotedRegsVMap),
                  DT);

  LLVM_DEBUG(DBGS << "2:\n" << *F);

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
