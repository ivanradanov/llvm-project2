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
#include "llvm/ADT/MapVector.h"
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
#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
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
#include <memory>

using namespace llvm;
using std::make_unique;
using std::unique_ptr;

#define DEBUG_TYPE "loop-unroll-and-interleave"

static cl::opt<int> UnrollFactorOpt("luai-factor", cl::init(1), cl::Hidden,
                                    cl::desc("Factor to coarsen with"));

static cl::opt<bool> UseDynamicConvergenceOpt(
    "luai-use-dynamic-convergence", cl::init(false), cl::Hidden,
    cl::desc("Runtime check for convergence between coarsened iterations"));

template <typename OutTy, typename CastTy, typename InTy>
static OutTy mapContainer(InTy &Container, ValueToValueMapTy &VMap) {
  auto MappedRange = llvm::map_range(
      Container, [&VMap](Value *V) { return cast<CastTy>(VMap[V]); });
  OutTy Mapped(MappedRange.begin(), MappedRange.end());
  return std::move(Mapped);
}

typedef SmallPtrSet<BasicBlock *, 8> BBSet;

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
  const unsigned UnrollFactor;
  const bool UseDynamicConvergence;

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
    BBSet Blocks;

    // Properties
    bool IsNested;

    // Used for transformations later
    unique_ptr<ValueToValueMapTy> DRLVMap;
    unique_ptr<ValueToValueMapTy> ReverseDRLVMap;
    SmallVector<BasicBlock *> DivergentRegionsLoopBlocks;
    Loop *DivergentRegionsLoop;

    BBSet ExecutedByCL;
    unique_ptr<ValueToValueMapTy> DefinedOutsideDemotedVMap;
    unique_ptr<ValueToValueMapTy> DefinedInsideDemotedVMap;

    AllocaInst *IdentPtr;

    DivergentRegion()
        : From(nullptr), To(nullptr), Entry(nullptr), Exit(nullptr),
          IsNested(false), DRLVMap(new ValueToValueMapTy),
          ReverseDRLVMap(new ValueToValueMapTy), DivergentRegionsLoop(nullptr),
          DefinedOutsideDemotedVMap(new ValueToValueMapTy),
          DefinedInsideDemotedVMap(new ValueToValueMapTy) {}
  };

  ValueToValueMapTy DemotedRegsVMap;

  SmallPtrSet<Instruction *, 8> DivergentBranches;
  // A DivergentRegion is uniquely identified by the convergent to divergent
  // edge
  SmallPtrSet<BranchInst *, 8> ConvergentToDivergentEdges;
  SmallPtrSet<BranchInst *, 8> DivergentToConvergentEdges;
  // Needs to be a list because we store pointers to it that should not get
  // invalidated
  std::list<DivergentRegion> DivergentRegions;

  void demoteDRRegs(Loop *CL);
  void populateDivergentRegions();
  bool isLegalToCoarsen(Loop *TheLoop, LoopInfo *LI);

public:
  LoopUnrollAndInterleave(OptimizationRemarkEmitter &ORE, unsigned UnrollFactor,
                          bool UseDynamicConvergence)
      : ORE(ORE), UnrollFactor(UnrollFactor),
        UseDynamicConvergence(UseDynamicConvergence) {}
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
  // `From` is reachable only if we reach it through its successors and not just
  // by starting at it, so we do not insert it into `Reachable`. We have to
  // differentiate these cases as the From block should be included in the DR
  // only if it is reachable in the above sense.
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

  BBSet ConvergingBlocks;
  BBSet EntryBlocks;

  for (Instruction *Term : DivergentBranches) {
    BasicBlock *Entry = Term->getParent();
    auto *ConvergeBlock = PDT->getNode(Entry)->getIDom()->getBlock();
    assert(ConvergeBlock &&
           PDT->dominates(CombinedLatchExiting, ConvergeBlock));

    BBSet Reachable;
    findReachableFromTo(Entry, ConvergeBlock, Reachable);
    Reachable.erase(ConvergeBlock);

    // We will insert the Entry and Exit later
    DivergentRegion Region;
    Region.From = Entry;
    Region.To = ConvergeBlock;
    Region.Blocks = std::move(Reachable);
    DivergentRegions.push_back(std::move(Region));

    ConvergingBlocks.insert(ConvergeBlock);
    EntryBlocks.insert(Entry);
  }

  auto AddExisting = [&](BasicBlock *TheBlock, BasicBlock *BB1,
                         BasicBlock *BB2 = nullptr) {
    for (auto &DR : llvm::make_filter_range(
             DivergentRegions, [TheBlock](DivergentRegion &DR) {
               return DR.Blocks.contains(TheBlock);
             })) {
      DR.Blocks.insert(BB1);
      if (BB2)
        DR.Blocks.insert(BB2);
    }
  };
  auto AddExit = [&](BasicBlock *TheBlock, BasicBlock *DivergentExit,
                     BasicBlock *Convergent) {
    for (auto &DR : llvm::make_filter_range(
             DivergentRegions,
             [TheBlock](DivergentRegion &DR) { return DR.To == TheBlock; })) {
      DR.Blocks.erase(Convergent);
      DR.Blocks.insert(DivergentExit);
      DR.Exit = DivergentExit;
      DR.To = Convergent;
    }
  };
  auto AddEntry = [&](BasicBlock *TheBlock, BasicBlock *Convergent,
                      BasicBlock *DivergentEntry) {
    auto DR = find_if(DivergentRegions, [TheBlock](DivergentRegion &DR) {
      return DR.From == TheBlock;
    });
    assert(DR != DivergentRegions.end());
    DR->Blocks.insert(DivergentEntry);
    DR->From = Convergent;
    DR->Entry = DivergentEntry;
  };

  BBSet EntryAndConvergingBlocks =
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

    AddExisting(TheBlock, Convergent, DivergentExit);
    AddExit(TheBlock, DivergentExit, Convergent);
    AddEntry(TheBlock, Convergent, DivergentEntry);
  }

  for (auto *TheBlock : EntryBlocks) {
    if (EntryAndConvergingBlocks.contains(TheBlock))
      // Already handled
      continue;
    std::string BlockName = TheBlock->getName().str();
    auto *DivergentEntry = TheBlock;
    BasicBlock *Convergent =
        DivergentEntry->splitBasicBlockBefore(DivergentEntry->getTerminator());
    DivergentEntry->setName(BlockName + ".divergent.entry");
    Convergent->setName(BlockName);

    TheLoop->addBasicBlockToLoop(Convergent, *LI);
    if (TheLoop->getHeader() == TheBlock)
      TheLoop->moveToHeader(Convergent);

    AddExisting(TheBlock, Convergent);
    AddEntry(TheBlock, Convergent, DivergentEntry);
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

    AddExisting(TheBlock, DivergentExit);
    AddExit(TheBlock, DivergentExit, Convergent);
  }

  for (auto &DR : DivergentRegions)
    DR.IsNested = any_of(DivergentRegions, [&](DivergentRegion &OtherDR) {
      return &OtherDR != &DR && OtherDR.Blocks.contains(DR.From);
    });

  LLVM_DEBUG({
    for (auto &DR : DivergentRegions) {
      DBGS << "Divergent region for entry %" << DR.Entry->getName()
           << " and exit %" << DR.Exit->getName() << " from block %"
           << DR.From->getName() << " to block %" << DR.To->getName()
           << " nested=" << DR.IsNested << ":\n";
      for (auto *BB : DR.Blocks)
        dbgs() << "%" << BB->getName() << ", ";
      dbgs() << "\n";
    }
  });

  for (auto &DR : DivergentRegions) {
    ConvergentToDivergentEdges.insert(
        cast<BranchInst>(DR.From->getTerminator()));
  }

  if (!UseDynamicConvergence) {
    // If we are not going to use dynamic convergence, then all DRs that have
    // entries in another DR are unreachable - delete them.
    for (auto It = DivergentRegions.begin(); It != DivergentRegions.end();) {
      bool Erase = false;
      for (auto &DR : DivergentRegions) {
        if (&DR == &*It)
          continue;
        if (DR.Blocks.contains(It->Entry)) {
          Erase = true;
          break;
        }
      }
      if (Erase)
        It = DivergentRegions.erase(It);
      else
        ++It;
    }
  }
}

void LoopUnrollAndInterleave::demoteDRRegs(Loop *CL) {

  for (auto &DR : DivergentRegions) {

    Loop *NCL = DR.DivergentRegionsLoop;
    ValueToValueMapTy &VMap = *DR.DRLVMap;
    SmallVector<Instruction *> ToDemote;

    // Demote values defined outside a DR used inside it in the Non Coarsened
    // Loop (NCL)
    auto MappedDRBlocks = mapContainer<BBSet, BasicBlock>(DR.Blocks, VMap);

    DR.ExecutedByCL = {};
    // The blocks that dominate the Entry and are in the DR will be executed by
    // the coarsened loop first, which means the values that are defined in them
    // are considered to be "defined outside" the DR
    auto *DomBlock = DR.From;
    while (DR.Blocks.contains(DomBlock)) {
      DR.ExecutedByCL.insert(DomBlock);
      DomBlock = DT->getNode(DomBlock)->getIDom()->getBlock();
    }

    auto MappedExecutedByCL =
        mapContainer<BBSet, BasicBlock>(DR.ExecutedByCL, VMap);

    // Find values defined outside the DR and used inside it
    for (auto *BB : NCL->getBlocks()) {
      if (!MappedExecutedByCL.contains(BB) && MappedDRBlocks.contains(BB))
        continue;

      for (auto &I : *BB) {
        if (I.use_empty())
          continue;
        for (auto *User : I.users()) {
          auto *UserI = dyn_cast<Instruction>(User);
          if (!UserI)
            continue;
          if (MappedDRBlocks.contains(UserI->getParent())) {
            ToDemote.push_back(&I);
            break;
          }
        }
      }
    }
    for (auto *I : ToDemote) {
      auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[I]);
      if (!Demoted) {
        Demoted = DemoteRegToStack(*I, /*VolatileLoads=*/false,
                                   Preheader->getFirstNonPHI());
        DemotedRegsVMap[I] = Demoted;
      }
      (*DR.DefinedOutsideDemotedVMap)[I] = Demoted;
    }

    // Demote values defined inside a DR and used outside it in the Coarsened
    // Loop (CL)
    ToDemote.clear();
    // Find values defined outside the DR and used inside it
    for (auto *BB : DR.Blocks) {
      SmallVector<Instruction *> ToHandle;
      for (auto &I : *BB) {
        ToHandle.push_back(&I);
      }
      for (auto *II : ToHandle) {
        auto &I = *II;
        if (I.use_empty())
          continue;
        for (auto *User : I.users()) {
          auto *UserI = dyn_cast<Instruction>(User);
          if (!UserI)
            continue;
          BasicBlock *UserBB = UserI->getParent();
          if (DR.Blocks.contains(UserBB))
            continue;
          ToDemote.push_back(&I);
          break;
        }
      }
    }
    for (auto *I : ToDemote) {
      auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[I]);
      if (!Demoted) {
        Demoted = DemoteRegToStack(*I, /*VolatileLoads=*/false,
                                   Preheader->getFirstNonPHI());
        DemotedRegsVMap[I] = Demoted;
      }
      (*DR.DefinedInsideDemotedVMap)[I] = Demoted;
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

  // We need the upper bound of the loop to be defined before we enter it in
  // order to know whether we should enter the coarsened version or the
  // epilogue.
  Value *End = [&](Value *End) -> Value * {
    Instruction *I = dyn_cast<Instruction>(End);
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
  // TODO Figure out what should happen to alloca's we clone...
  unsigned It = 0;
  for (auto &DR : DivergentRegions) {
    DR.DivergentRegionsLoop = cloneLoopWithPreheader(
        ExitBlock, &F->getEntryBlock(), L, *DR.DRLVMap,
        ".drs." + std::to_string(It++), LI, &DT, DR.DivergentRegionsLoopBlocks);
    // Do not remap the preheader values, the DR blocks should refer to the
    // original preheader
    for (Instruction &I : *Preheader) {
      DR.DRLVMap->erase(&I);
    }
    remapInstructionsInBlocks(DR.DivergentRegionsLoopBlocks, *DR.DRLVMap);
  }

  // Clone the loop once more to use as an epilogue, the original one will be
  // coarsened in-place
  // TODO Figure out what should happen to alloca's we clone...
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
    Value *NewStep = PreheaderBuilder.CreateMul(
        IVStepVal, ConstantInt::get(IVStepVal->getType(), UnrollFactor),
        "coarsened.step");
    cast<Instruction>(NewStep)->moveAfter(IVStepInst);
    IVStepInst->replaceUsesWithIf(
        NewStep, [NewStep, IsInEpilogue](Use &U) -> bool {
          return U.getUser() != NewStep && !IsInEpilogue(U);
        });
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
    Value *MultipliedStep = PreheaderBuilder.CreateMul(
        IVStepVal, ConstantInt::get(IVStepVal->getType(), I));
    CoarsenedInitialIV =
        PreheaderBuilder.CreateAdd(InitialIVVal, MultipliedStep,
                                   "initial.iv.coarsened." + std::to_string(I));
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

      bool IsTerminator = I->isTerminator();
      bool IsConvergentToDivergentBr = false;
      Instruction *DivergentCond = nullptr;
      if (IsTerminator && UseDynamicConvergence) {
        BranchInst *BI = dyn_cast<BranchInst>(I);
        if (ConvergentToDivergentEdges.contains(BI)) {
          assert(BI && !BI->isConditional());
          IsConvergentToDivergentBr = true;
          auto *DivBr = BI->getSuccessor(0)->getTerminator();
          if (auto *Sw = dyn_cast<SwitchInst>(DivBr))
            DivergentCond = cast<Instruction>(Sw->getCondition());
          else if (auto *BI = dyn_cast<BranchInst>(DivBr))
            DivergentCond = cast<Instruction>(BI->getCondition());
          else
            llvm_unreachable(
                "Divergent branches must be either branch or switch insts");
        }
      }

      for (unsigned It = 1; It < UnrollFactor; It++) {
        if (IsTerminator) {
          // Do not clone terminators - we use the control flow of the
          // existing iteration (if the branch is divergent we will insert
          // an entry to a divergent region here later)
          continue;
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

      // The current state:
      //
      // ConvergentBlock:
      //   ...
      //   br %DivergentEntry ( = I)
      // DivergentEntry:
      //   ...
      //   br %cond %... ( = DivBr)
      //
      // We change `I` to be a conditional branch with the condition of whether
      // all the coarsened iterations agreed on the `DivBr` condition
      if (IsConvergentToDivergentBr) {
        Value *AllSame = ConstantInt::get(IntegerType::getInt1Ty(Ctx), true);
        Value *Cond = DivergentCond;
        IRBuilder<> Builder(I);
        for (unsigned It = 1; It < UnrollFactor; It++) {
          // The condition has to be an instruction otherwise it cannot be
          // divergent
          auto *CoarsenedCond = cast<Instruction>((*VMaps[It])[Cond]);
          auto *Same = Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ,
                                         CoarsenedCond, Cond);
          AllSame = Builder.CreateAnd(Same, AllSame);
        }
        // If the branches agree on the condition, branch to the convergent
        // region, else, we branch to the divergent region (we will insert that
        // successor later)
        Builder.CreateCondBr(AllSame, I->getSuccessor(0), I->getSuccessor(0));
        I->eraseFromParent();
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
  EpiloguePH->eraseFromParent();

  DT.recalculate(*F); // TODO another recalculation...
  // Find all values used in the divergent region but defined outside and demote
  // them to memory - now we can change the CFG easier.
  demoteDRRegs(TheLoop);

  for (auto &DR : DivergentRegions) {
    ValueToValueMapTy &ReverseDRLVMap = *DR.ReverseDRLVMap;
    for (auto M : *DR.DRLVMap) {
      ReverseDRLVMap[cast<Value>(M.second)] = const_cast<Value *>(M.first);
    }
  }

  // Hook into the the blocks from the DR loop to do the divergent part of the
  // coarsened computation. Generates Intro and Outro blocks which bring in the
  // appropriate coarsened values into the DR in the uncoarsened part, and then
  // bring out those values for use in the subsequent coarsened computation.
  SmallVector<AllocaInst *> DRTmpStorage;
  Type *CoarsenedIdentifierTy = IntegerType::getInt32Ty(Ctx);
  for (auto &DR : DivergentRegions) {
    auto &DivergentRegionsVMap = *DR.DRLVMap;
    auto &ReverseDRLVMap = *DR.ReverseDRLVMap;
    BasicBlock *DREntry = cast<BasicBlock>(DivergentRegionsVMap[DR.Entry]);
    BasicBlock *DRExit = cast<BasicBlock>(DivergentRegionsVMap[DR.Exit]);
    BasicBlock *DRTo = cast<BasicBlock>(DivergentRegionsVMap[DR.To]);

    auto *CoarsenedIdentPtr = DR.IdentPtr = PreheaderBuilder.CreateAlloca(
        CoarsenedIdentifierTy, /*ArraySize*/ nullptr, "dr.coarsened.ident");

    // Multiple DivergentRegion entries may converge at the same location,
    // create the exit switch only once
    auto *ToOutroSw = dyn_cast<SwitchInst>(DRExit->getTerminator());
    int NumSwCases;
    if (!ToOutroSw) {
      auto *Ld = new LoadInst(CoarsenedIdentifierTy, CoarsenedIdentPtr,
                              "coarsened.ident.load", DRExit->getTerminator());
      // We will override the default dest later, set something random
      ToOutroSw = SwitchInst::Create(Ld, DREntry, 0);
      DRExit->getTerminator()->eraseFromParent();
      ToOutroSw->insertInto(DRExit, DRExit->end());
      NumSwCases = 0;
    } else {
      NumSwCases = ToOutroSw->getNumCases() + 1;
    }

    BasicBlock *LastOutro = nullptr;
    SmallVector<BasicBlock *> Intros;
    ValueToValueMapTy SavedCoarsened;
    for (unsigned It = 0; It < UnrollFactor; It++) {
      auto ThisFactorIdentifierVal = NumSwCases + It;
      auto *ThisFactorIdentifier = cast<ConstantInt>(
          ConstantInt::get(CoarsenedIdentifierTy, ThisFactorIdentifierVal));

      auto *Intro = BasicBlock::Create(
          Ctx, DREntry->getName() + ".intro." + std::to_string(It), F, DREntry);
      Intros.push_back(Intro);
      IRBuilder<> IntroBuilder(Intro);
      IntroBuilder.CreateStore(ThisFactorIdentifier, CoarsenedIdentPtr);

      auto *ToIntroBI = BranchInst::Create(Intro);
      if (It == 0) {
        assert(!LastOutro);
        if (UseDynamicConvergence) {
          // If the coarsened versions did _not_ agree on the condition, jump to
          // the divergent region
          DR.From->getTerminator()->setSuccessor(1, Intro);
          ToIntroBI->deleteValue();
        } else {
          DR.From->getTerminator()->eraseFromParent();
          ToIntroBI->insertInto(DR.From, DR.From->end());
        }
      } else {
        ToIntroBI->insertInto(LastOutro, LastOutro->end());
      }

      for (auto P : *DR.DefinedOutsideDemotedVMap) {
        auto *I = P.first;
        auto *OriginalValue = cast<Instruction>(ReverseDRLVMap[I]);
        Instruction *CoarsenedValue =
            cast<Instruction>((*VMaps[It])[OriginalValue]);
        // Reload from the stashes if needed
        IntroBuilder.CreateStore(CoarsenedValue, P.second);
      }

      IntroBuilder.CreateBr(DREntry);

      auto *Outro = LastOutro = BasicBlock::Create(
          Ctx, DRExit->getName() + ".outro." + std::to_string(It), F, DRTo);
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

      if (ThisFactorIdentifierVal == 0)
        ToOutroSw->setDefaultDest(Outro);
      else
        ToOutroSw->addCase(ThisFactorIdentifier, Outro);
    }
  }

  // Grab the demoted allocas before the map is invalidated
  for (auto P : DemotedRegsVMap)
    DRTmpStorage.push_back(cast<AllocaInst>(P.second));

  // Now that we have done the plumbing around the divergent regions loop, erase
  // the remainders
  for (auto &DR : DivergentRegions) {
    BBSet UnusedDRBlocks(DR.DivergentRegionsLoopBlocks.begin(),
                         DR.DivergentRegionsLoopBlocks.end());
    for (auto &DR : DivergentRegions) {
      auto MappedDRBlocks =
          mapContainer<BBSet, BasicBlock>(DR.Blocks, *DR.DRLVMap);
      set_subtract(UnusedDRBlocks, MappedDRBlocks);
    }
    SmallVector<BasicBlock *> Tmp(UnusedDRBlocks.begin(), UnusedDRBlocks.end());
    DeleteDeadBlocks(Tmp);
  }

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
  // I.e. when we are done with our iterations of the coarsened loop.
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
  BasicBlock *EpilogueFrom;
  {
    // Note we need to check for the loop end first and exit the loop
    // alltogether if we are at the end because if all iterations are handled by
    // the coarsened loop the final IV will be equal to the epilogue start IV
    BranchInst *BackEdge =
        dyn_cast<BranchInst>(CombinedLatchExiting->getTerminator());
    assert(BackEdge && BackEdge->isConditional());
    BasicBlock *PrevBB = EpilogueFrom =
        CombinedLatchExiting->splitBasicBlockBefore(BackEdge);
    PrevBB->setName(CombinedLatchExiting->getName() + "epilogue.start.check");
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
    EndCheckBB->setName(PrevBB->getName() + ".original.end.check");
    Instruction *EndCheckBI = EndCheckBB->getTerminator();
    IRBuilder<> EpilogueCheckBuilder(EndCheckBI);

    // TODO Is there a canonical form? Do we know which side the exit block is
    // on?
    BasicBlock *TrueBB = nullptr, *FalseBB = nullptr;
    if (BackEdge->getSuccessor(0) == ExitBlock) {
      TrueBB = ExitBlock;
      FalseBB = PrevBB;
    } else {
      assert(BackEdge->getSuccessor(1) == ExitBlock);
      TrueBB = ExitBlock;
      FalseBB = PrevBB;
    }
    EpilogueCheckBuilder.CreateCondBr(BackEdge->getCondition(), TrueBB,
                                      FalseBB);
    EndCheckBI->eraseFromParent();
  }

  // Start the epilogue loop ivs from the iteration the coarsened version ended
  // with instead of the original lb
  {
    ValueToValueMapTy ReverseEpilogueVMap;
    for (auto M : EpilogueVMap) {
      ReverseEpilogueVMap[cast<Value>(M.second)] = const_cast<Value *>(M.first);
    }
    for (auto &PN : EpilogueLoop->getHeader()->phis()) {
      auto *CoarsenedPN = cast<PHINode>(ReverseEpilogueVMap[&PN]);
      PN.setIncomingValueForBlock(
          EpilogueFrom,
          CoarsenedPN->getIncomingValueForBlock(CombinedLatchExiting));
      PN.addIncoming(CoarsenedPN->getIncomingValueForBlock(Preheader),
                     Preheader);
    }
  }

  // If we do not use dynamic convergence then /some/ of the coarsened versions
  // of the DRs are unused and we have to delete them.
  //
  // We can still have a case like this:
  //
  //    |        |
  // NonDivBB  DivBB
  //         \ /   \
  //          BB   BB
  //           \   /
  //         ConvergeBB
  //             |
  //
  // Where a non-divergent flow joins a divergent regions - this means not all
  // coarsened versions of DRs are dead - just delete the ones unreachable from
  // the entry
  EliminateUnreachableBlocks(*F);

  // Now that we are done with the aggressive CFG restructuring and deleting
  // dead blocks we can re-promote the regs we demoted earlier.
  // TODO can we check we were able to promote everything without undefs?
  DT.recalculate(*F); // TODO another recalculation...
  for (auto &DR : DivergentRegions) {
    DRTmpStorage.push_back(DR.IdentPtr);
  }
  PromoteMemToReg(DRTmpStorage, DT);

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

  unsigned UnrollFactor = UnrollFactorOpt;
  if (char *Env = getenv("UNROLL_AND_INTERLEAVE_FACTOR"))
    StringRef(Env).getAsInteger(10, UnrollFactor);
  if (UnrollFactor == 1) {
    LLVM_DEBUG(DBGS << "Unroll factor of 1 - ignoring\n");
    return PreservedAnalyses::all();
  }
  if (UnrollFactor < 1) {
    LLVM_DEBUG(DBGS << "Unroll factor of less than 1 - ignoring\n");
    return PreservedAnalyses::all();
  }

  bool UseDynamicConvergence = UseDynamicConvergenceOpt;
  if (char *Env = getenv("UNROLL_AND_INTERLEAVE_DYNAMIC_CONVERGENCE")) {
    unsigned Int = 0;
    StringRef(Env).getAsInteger(10, Int);
    UseDynamicConvergence = Int;
  }

  // auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(L, AR);
  auto PDT = PostDominatorTree(*F);
  bool Changed =
      LoopUnrollAndInterleave(ORE, UnrollFactor, UseDynamicConvergence)
          .tryToUnrollLoop(&L, AR.DT, &AR.LI, AR.SE, AR.TTI, PDT) !=
      LoopUnrollResult::Unmodified;
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
