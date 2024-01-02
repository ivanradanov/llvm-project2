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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
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
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
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
#include <string>

using namespace llvm;
using std::make_unique;
using std::unique_ptr;

#define DEBUG_TYPE "loop-unroll-and-interleave"

static cl::opt<int> FactorOpt("luai-factor", cl::init(1), cl::Hidden,
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
typedef SmallVectorImpl<BasicBlock *> BBVecImpl;
typedef SmallVector<BasicBlock *, 8> BBVec;

namespace {
class BBInterleave {
protected:
  OptimizationRemarkEmitter &ORE;
  DominatorTree *DT;
  PostDominatorTree *PDT;

  BasicBlock *DominatingBlock;
  BasicBlock *PostDominatingBlock;
  SmallVector<AllocaInst *> DRTmpStorage;
  Function *F;

  // Optional
  LoopInfo *LI = nullptr;
  Loop *TheLoop = nullptr;

  // Options
  const unsigned Factor;
  const bool UseDynamicConvergence;
  const int InterProceduralInterleavingLevel;

private:
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
    unique_ptr<ValueToValueMapTy> DRVMap;
    unique_ptr<ValueToValueMapTy> ReverseDRVMap;
    SmallVector<BasicBlock *> ClonedBlocks;

    BBSet ExecutedByConvergent;
    unique_ptr<ValueToValueMapTy> DefinedOutsideDemotedVMap;
    unique_ptr<ValueToValueMapTy> DefinedInsideDemotedVMap;

    AllocaInst *IdentPtr;

    DivergentRegion()
        : From(nullptr), To(nullptr), Entry(nullptr), Exit(nullptr),
          IsNested(false), DRVMap(new ValueToValueMapTy),
          ReverseDRVMap(new ValueToValueMapTy),
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

  BBVec BBsToCoarsen;

  void demoteDRRegs();
  void populateDivergentRegions();

public:
  BBInterleave(OptimizationRemarkEmitter &ORE, unsigned Factor,
               bool UseDynamicConvergence, int InterProceduralInterleavingLevel)
      : ORE(ORE), Factor(Factor), UseDynamicConvergence(UseDynamicConvergence),
        InterProceduralInterleavingLevel(InterProceduralInterleavingLevel) {}
  LoopUnrollResult tryToUnrollBBs(
      Loop *L, LoopInfo *LI, BasicBlock *DominatingBlock,
      BasicBlock *PostDominatingBlock,
      const ArrayRef<BasicBlock *> &BBsToCoarsen, DominatorTree &DT,
      PostDominatorTree &PDT,
      SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> &VMaps,
      SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> &ReverseVMaps,
      SmallPtrSet<Instruction *, 8> &DivergentBranches);
  void cleanup();
};

class FunctionInterleave : public BBInterleave {
private:
  SmallPtrSet<Instruction *, 8> DivergentBranches;

public:
  FunctionInterleave(OptimizationRemarkEmitter &ORE, unsigned Factor,
                     bool UseDynamicConvergence,
                     int InterProceduralInterleavingLevel)
      : BBInterleave(ORE, Factor, UseDynamicConvergence,
                     InterProceduralInterleavingLevel) {}
  Function *tryToInterleaveFunction(Function *F);
};

class KmpcParallel51Interleave : public BBInterleave {
private:
  SmallPtrSet<Instruction *, 8> DivergentBranches;

public:
  KmpcParallel51Interleave(OptimizationRemarkEmitter &ORE, unsigned Factor,
                     bool UseDynamicConvergence,
                     int InterProceduralInterleavingLevel)
      : BBInterleave(ORE, Factor, UseDynamicConvergence,
                     InterProceduralInterleavingLevel) {}
  bool tryToInterleave(Instruction *I);
};

class LoopUnrollAndInterleave : public BBInterleave {
private:
  // Loop data
  Loop *TheLoop;
  BasicBlock *CombinedLatchExiting;
  BasicBlock *Preheader;

  SmallPtrSet<Instruction *, 8> DivergentBranches;

  LoopInfo *LI;

  bool collectDivergentBranches(Loop *TheLoop, LoopInfo *LI);

public:
  LoopUnrollAndInterleave(OptimizationRemarkEmitter &ORE, unsigned UnrollFactor,
                          bool UseDynamicConvergence,
                          int InterProceduralInterleavingLevel)
      : BBInterleave(ORE, UnrollFactor, UseDynamicConvergence,
                     InterProceduralInterleavingLevel) {}
  LoopUnrollResult tryToUnrollAndInterleaveLoop(Loop *L, DominatorTree &DT,
                                                LoopInfo &LI,
                                                ScalarEvolution &SE,
                                                PostDominatorTree &PDT);
};

} // namespace

static void setLoopAlreadyCoarsened(Loop *L) {
  LLVMContext &Context = L->getHeader()->getContext();

  MDNode *DisableUnrollMD = MDNode::get(
      Context,
      MDString::get(Context, "llvm.loop.unroll_and_interleave.disable"));
  MDNode *LoopID = L->getLoopID();
  MDNode *NewLoopID = makePostTransformationMetadata(
      Context, LoopID, {"llvm.loop.unroll_and_interleave."}, {DisableUnrollMD});
  L->setLoopID(NewLoopID);
}

static MDNode *getUnrollMetadataForLoop(const Loop *L, StringRef Name) {
  if (MDNode *LoopID = L->getLoopID())
    return GetUnrollMetadata(LoopID, Name);
  return nullptr;
}

static unsigned getLoopCoarseningFactor(Loop *L) {
  MDNode *MD =
      getUnrollMetadataForLoop(L, "llvm.loop.unroll_and_interleave.count");
  if (MD) {
    assert(MD->getNumOperands() == 2 &&
           "Unroll count hint metadata should have two operands.");
    unsigned Count =
        mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
    assert(Count >= 1 && "Unroll and interleave factor must be positive.");
    return Count;
  }
  return 0;
}

static unsigned getLoopCoarseningLevel(Loop *L) {
  MDNode *MD =
      getUnrollMetadataForLoop(L, "llvm.loop.unroll_and_interleave.level");
  if (MD) {
    assert(MD->getNumOperands() == 2 &&
           "Unroll level hint metadata should have two operands.");
    int Level =
        mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
    assert(Level >= -1 && "Unroll and interleave factor must be >= -1.");
    return Level;
  }
  return 0;
}

static bool getLoopAlreadyCoarsened(Loop *L) {
  return getUnrollMetadataForLoop(L, "llvm.loop.unroll_and_interleave.disable");
}

#define DBGS llvm::dbgs() << "LUAI: "
#define DBGS_FAIL llvm::dbgs() << "LUAI: FAIL: "
#define DBGS_DISABLED llvm::dbgs() << "LUAI: DISABLED: "

#pragma push_macro("ILLEGAL")
#define ILLEGAL()                                                              \
  do {                                                                         \
    if (DoExtraAnalysis)                                                       \
      Result = false;                                                          \
    else                                                                       \
      return false;                                                            \
  } while (0)

bool collectDivergentBranches(Function *F, SmallPtrSetImpl<Instruction *> &DivergentBranches) {
  const bool DoExtraAnalysis = true;
  bool Result = true;

  for (BasicBlock &BB : *F) {
    auto *Term = BB.getTerminator();
    auto *Br = dyn_cast<BranchInst>(Term);
    auto *Sw = dyn_cast<SwitchInst>(Term);
    auto *Ret = dyn_cast<ReturnInst>(Term);

    // TODO Should we have a call stack and evaluate whether conditions are
    // invariant? Currently we assume all conditional branches are divergent.
    if (Br) {
      if (Br->isConditional()) {
        DivergentBranches.insert(Term);
        LLVM_DEBUG(DBGS << "Divergent branch found: " << *Br << "\n");
      }
      continue;
    }
    if (Sw) {
      DivergentBranches.insert(Term);
      LLVM_DEBUG(DBGS << "Divergent switch found: " << *Sw << "\n");
      continue;
    }
    if (Ret) {
      continue;
    }
    LLVM_DEBUG(DBGS_FAIL << "Unsupported basic block terminator " << *Term
                         << "\n");
    ILLEGAL();
  }

  return Result;
}

bool LoopUnrollAndInterleave::collectDivergentBranches(Loop *TheLoop,
                                                       LoopInfo *LI) {
  const bool DoExtraAnalysis = true;
  bool Result = true;

  CombinedLatchExiting = TheLoop->getExitingBlock();
  if (CombinedLatchExiting != TheLoop->getLoopLatch()) {
    LLVM_DEBUG(DBGS << "Expected a combined exiting and latch block\n");
    ILLEGAL();
  }

  for (BasicBlock *BB : TheLoop->getBlocks()) {
    auto *Term = BB->getTerminator();
    auto *Br = dyn_cast<BranchInst>(Term);
    auto *Sw = dyn_cast<SwitchInst>(Term);
    if (!Br && !Sw) {
      LLVM_DEBUG(DBGS_FAIL << "Unsupported basic block terminator " << *Term
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
      LLVM_DEBUG(DBGS << "Divergent branch found: " << *Br << "\n");
    }
    if (Sw && !TheLoop->isLoopInvariant(Sw->getCondition())) {
      DivergentBranches.insert(Term);
      LLVM_DEBUG(DBGS << "Divergent switch found: " << *Sw << "\n");
    }
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

// TODO we can probably support functions with non-void returns but it is
// annoying so ignored for now, we should reevaluate how we check for
// legality of coarsening and do the check recursively on these as well
static Function *getInterleavableCallee(Instruction *I) {
  if (auto *CI = dyn_cast<CallInst>(I))
    if (auto *Called = CI->getCalledFunction())
      if (!Called->isDeclaration() &&
          Called->getFunctionType()->getReturnType()->isVoidTy())
        return Called;
  return nullptr;
}

void BBInterleave::populateDivergentRegions() {

  BBSet ConvergingBlocks;
  BBSet EntryBlocks;

  for (Instruction *Term : DivergentBranches) {
    BasicBlock *Entry = Term->getParent();
    auto *ConvergeBlock = PDT->getNode(Entry)->getIDom()->getBlock();
    assert(ConvergeBlock && PDT->dominates(PostDominatingBlock, ConvergeBlock));

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

    if (LI && TheLoop) {
      TheLoop->addBasicBlockToLoop(DivergentExit, *LI);
      TheLoop->addBasicBlockToLoop(Convergent, *LI);
      assert(TheLoop->getHeader() != TheBlock &&
             "A Converging block cannot be the header.");
    }
    BBsToCoarsen.push_back(DivergentExit);
    BBsToCoarsen.push_back(Convergent);

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

    if (LI && TheLoop) {
      TheLoop->addBasicBlockToLoop(Convergent, *LI);
      if (TheLoop->getHeader() == TheBlock)
        TheLoop->moveToHeader(Convergent);
    }
    if (DominatingBlock == TheBlock)
      DominatingBlock = Convergent;
    BBsToCoarsen.push_back(Convergent);

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

    if (LI && TheLoop)
      TheLoop->addBasicBlockToLoop(DivergentExit, *LI);
    BBsToCoarsen.push_back(DivergentExit);

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

void BBInterleave::demoteDRRegs() {
  DenseMap<DivergentRegion *, SmallVector<Instruction *>> ToDemoteDOUIAll;
  DenseMap<DivergentRegion *, SmallVector<Instruction *>> ToDemoteDIUOAll;
  for (auto &DR : DivergentRegions) {
    ValueToValueMapTy &VMap = *DR.DRVMap;

    SmallVector<Instruction *> &ToDemoteDOUI = ToDemoteDOUIAll[&DR] = {};

    // Demote values defined outside a DR used inside it in the non coarsened
    // loop
    auto MappedDRBlocks = mapContainer<BBSet, BasicBlock>(DR.Blocks, VMap);

    DR.ExecutedByConvergent = {};
    // The blocks that dominate the Entry and are in the DR will be executed by
    // the coarsened BBs first, which means the values that are defined in them
    // are considered to be "defined outside" the DR
    auto *DomBlock = DR.From;
    while (DR.Blocks.contains(DomBlock)) {
      DR.ExecutedByConvergent.insert(DomBlock);
      DomBlock = DT->getNode(DomBlock)->getIDom()->getBlock();
    }

    auto MappedExecutedByCL =
        mapContainer<BBSet, BasicBlock>(DR.ExecutedByConvergent, VMap);

    // Find values defined outside the DR and used inside it
    for (auto *BB : DR.ClonedBlocks) {
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
            ToDemoteDOUI.push_back(&I);
            break;
          }
        }
      }
    }

    // Demote values defined inside a DR and used outside it in the Coarsened
    // Loop (CL)
    SmallVector<Instruction *> &ToDemoteDIUO = ToDemoteDIUOAll[&DR] = {};
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
          ToDemoteDIUO.push_back(&I);
          break;
        }
      }
    }
  }
  LLVM_DEBUG({
    for (auto &DR : DivergentRegions) {
      SmallVector<Instruction *> &ToDemoteDOUI = ToDemoteDOUIAll[&DR];
      SmallVector<Instruction *> &ToDemoteDIUO = ToDemoteDIUOAll[&DR];
      DBGS << "To demote DOUI for DR with entry %" << DR.Entry->getName()
           << ":\n";
      for (auto *I : ToDemoteDOUI) {
        dbgs() << *I << "\n";
      }
      DBGS << "To demote DIUO for DR with entry %" << DR.Entry->getName()
           << ":\n";
      for (auto *I : ToDemoteDIUO) {
        dbgs() << *I << "\n";
      }
    }
  });
  for (auto &DR : DivergentRegions) {
    SmallVector<Instruction *> &ToDemoteDOUI = ToDemoteDOUIAll[&DR];
    SmallVector<Instruction *> &ToDemoteDIUO = ToDemoteDIUOAll[&DR];
    for (auto *I : ToDemoteDOUI) {
      auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[I]);
      if (!Demoted) {
        Demoted = DemoteRegToStack(*I, /*VolatileLoads=*/false,
                                   DominatingBlock->getFirstNonPHI());
        DemotedRegsVMap[I] = Demoted;
      }
      (*DR.DefinedOutsideDemotedVMap)[I] = Demoted;
    }
    for (auto *I : ToDemoteDIUO) {
      auto *Demoted = cast_or_null<AllocaInst>(DemotedRegsVMap[I]);
      if (!Demoted) {
        Demoted = DemoteRegToStack(*I, /*VolatileLoads=*/false,
                                   DominatingBlock->getFirstNonPHI());
        DemotedRegsVMap[I] = Demoted;
      }
      (*DR.DefinedInsideDemotedVMap)[I] = Demoted;
    }
  }
}

bool KmpcParallel51Interleave::tryToInterleave(Instruction *I) {
  Function *F = [&]() {
    if (auto *CI = dyn_cast<CallInst>(I))
      if (auto *Called = CI->getCalledFunction())
        if (Called->getName() == "__kmpc_parallel_51")
          if (auto *Callback = dyn_cast<Function>(CI->getArgOperand(5)))
            return Callback;
    return (Function *) nullptr;
  }();
  if (!F)
    return false;

  auto &Ctx = F->getContext();
  std::string NewName =
      F->getName().str() + ".__kmpc_parallel_51.callback.coarsened." + std::to_string(Factor);

  if (Function *NewF = F->getParent()->getFunction(NewName))
    return true;

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "Will interleave __kmpc_parallel_51 callback function:\n" << *F);
  }

  FunctionType *FTy = F->getFunctionType();
  assert(FTy->getReturnType()->isVoidTy());
  LLVM_DEBUG({
    for (auto ParamTy : FTy->params())
      assert(ParamTy->isPointerTy());
    });
  PointerType *PtrTy = PointerType::get(Ctx, 0);
  SmallVector<Type *> ArgTypes(2 + Factor, PtrTy);

  FunctionType *NewFTy = FunctionType::get(Type::getVoidTy(F->getContext()),
                                           ArgTypes, /*isVarArg=*/false);
  Function *NewF =
      Function::Create(NewFTy, F->getLinkage(), NewName, F->getParent());
  assert(Factor <= 16 && "Currently we only support factor up to 16 because of libomptarget limitations.");

  unsigned NumArgs = FTy->getNumParams() - 2;

  ValueToValueMapTy VMap;
  assert(FTy->getNumParams() >= 2);
  BasicBlock *EntryBlock = BasicBlock::Create(Ctx, "entry", NewF);
  IRBuilder<> Builder(EntryBlock);
  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> VMaps;
  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> ReverseVMaps;
  VMaps.reserve(Factor);
  ReverseVMaps.reserve(Factor);
  for (unsigned I = 0; I < Factor; I++) {
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
    ReverseVMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
  }

  for (unsigned ArgN = 0; ArgN < 2; ArgN++) {
    // Global Tid and Bound Tid
    Argument *OldArg = F->getArg(ArgN);
    Argument *NewArg = NewF->getArg(Factor * ArgN);
    assert(OldArg->getType() == NewArg->getType());
    VMap[OldArg] = NewArg;
    (*VMaps[0])[OldArg] = NewArg;
  }

  for (unsigned I = 0; I < Factor; I++) {
    Argument *FactorArgs = NewF->getArg(2 + Factor * I + 1);
    for (unsigned ArgN = 0; ArgN < NumArgs; ArgN++) {
      // The actual args
      Argument *OriginalArg = F->getArg(ArgN);
      Value *Gep = Builder.CreateGEP(PtrTy, FactorArgs,
                                     {Builder.getInt32(ArgN)}, "clonegep");
      Value *NewArg = Builder.CreateLoad(PtrTy, Gep, "clonearg");
      assert(OriginalArg->getType() == NewArg->getType());
      if (I == 0)
        VMap[OriginalArg] = NewArg;
      (*VMaps[I])[VMap[OriginalArg]] = NewArg;
    }
    if (I != 0) {
      for (unsigned ArgN = 0; ArgN < 2; ArgN++) {
        // The actual args
        Argument *OriginalArg = F->getArg(ArgN);
        Value *Gep = Builder.CreateGEP(PtrTy, FactorArgs,
                                       {Builder.getInt32(NumArgs + ArgN)}, "clonegep");
        Value *NewArg = Builder.CreateLoad(PtrTy, Gep, "cloneargtid");
        assert(OriginalArg->getType() == NewArg->getType());
        (*VMaps[I])[VMap[OriginalArg]] = NewArg;
      }
    }
  }

  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  NewF->setVisibility(GlobalValue::DefaultVisibility);
  NewF->setLinkage(GlobalValue::InternalLinkage);

  Builder.CreateBr(cast<BasicBlock>(VMap[&F->getEntryBlock()]));

  BasicBlock *ReturnBB =
      BasicBlock::Create(Ctx, "unified.exit", NewF, &NewF->front());
  ReturnBB->moveAfter(&NewF->back());
  auto *NewRet = ReturnInst::Create(Ctx);
  NewRet->insertInto(ReturnBB, ReturnBB->end());

  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        if (Ret == NewRet)
          break;
        BranchInst::Create(ReturnBB, Ret);
        Ret->eraseFromParent();
        break;
      }
    }
  }

  if (!collectDivergentBranches(NewF, DivergentBranches)) {
    // TODO we should probably do these checks before we clone the function...
    NewF->eraseFromParent();
    return false;
  }

  for (unsigned I = 0; I < Factor; I++)
    for (auto M : *VMaps[I])
      (*ReverseVMaps[I])[cast<Value>(M.second)] = const_cast<Value *>(M.first);

  BBVec BBs;
  for (auto &BB : *NewF)
    if (&BB != ReturnBB)
      BBs.push_back(&BB);
  auto DT = DominatorTree(*NewF);
  auto PDT = PostDominatorTree(*NewF);
  tryToUnrollBBs(nullptr, nullptr, &NewF->getEntryBlock(), ReturnBB, BBs, DT,
                 PDT, VMaps, ReverseVMaps, DivergentBranches);
  cleanup();

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "After interleaving function:\n" << *NewF);
  }



  return true;
}

Function *FunctionInterleave::tryToInterleaveFunction(Function *F) {
  std::string NewName =
      F->getName().str() + ".coarsened." + std::to_string(Factor);

  if (Function *NewF = F->getParent()->getFunction(NewName))
    return NewF;

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "Will interleave function:\n" << *F);
  }

  FunctionType *FTy = F->getFunctionType();
  assert(FTy->getReturnType()->isVoidTy());
  SmallVector<Type *> ArgTypes;
  for (unsigned ArgN = 0; ArgN < FTy->getNumParams(); ArgN++)
    for (unsigned I = 0; I < Factor; I++)
      ArgTypes.push_back(FTy->getParamType(ArgN));

  FunctionType *NewFTy = FunctionType::get(Type::getVoidTy(F->getContext()),
                                           ArgTypes, /*isVarArg=*/false);
  Function *NewF =
      Function::Create(NewFTy, F->getLinkage(), NewName, F->getParent());

  ValueToValueMapTy VMap;
  for (unsigned ArgN = 0; ArgN < FTy->getNumParams(); ArgN++) {
    Argument *OldArg = F->getArg(ArgN);
    Argument *NewArg = NewF->getArg(Factor * ArgN);
    assert(OldArg->getType() == NewArg->getType());
    VMap[OldArg] = NewArg;
  }

  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  NewF->setVisibility(GlobalValue::DefaultVisibility);
  NewF->setLinkage(GlobalValue::InternalLinkage);

  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> VMaps;
  SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> ReverseVMaps;
  VMaps.reserve(Factor);
  ReverseVMaps.reserve(Factor);
  for (unsigned I = 0; I < Factor; I++) {
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
    ReverseVMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
  }

  for (unsigned ArgN = 0; ArgN < FTy->getNumParams(); ArgN++) {
    for (unsigned I = 0; I < Factor; I++) {
      auto *NCArg = NewF->getArg(Factor * ArgN);
      auto *CArg = NewF->getArg(Factor * ArgN + I);
      assert(NCArg->getType() == CArg->getType());
      (*VMaps[I])[NCArg] = CArg;
    }
  }

  auto &Ctx = F->getContext();
  BasicBlock *ReturnBB =
      BasicBlock::Create(Ctx, "unified.exit", NewF, &NewF->front());
  ReturnBB->moveAfter(&NewF->back());
  auto *NewRet = ReturnInst::Create(Ctx);
  NewRet->insertInto(ReturnBB, ReturnBB->end());

  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        if (Ret == NewRet)
          break;
        BranchInst::Create(ReturnBB, Ret);
        Ret->eraseFromParent();
        break;
      }
    }
  }

  if (!collectDivergentBranches(NewF, DivergentBranches)) {
    // TODO we should probably do these checks before we clone the function...
    NewF->eraseFromParent();
    return nullptr;
  }

  for (unsigned I = 0; I < Factor; I++)
    for (auto M : *VMaps[I])
      (*ReverseVMaps[I])[cast<Value>(M.second)] = const_cast<Value *>(M.first);

  BBVec BBs;
  for (auto &BB : *NewF)
    if (&BB != ReturnBB)
      BBs.push_back(&BB);
  auto DT = DominatorTree(*NewF);
  auto PDT = PostDominatorTree(*NewF);
  tryToUnrollBBs(nullptr, nullptr, &NewF->getEntryBlock(), ReturnBB, BBs, DT,
                 PDT, VMaps, ReverseVMaps, DivergentBranches);
  cleanup();

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "After interleaving function:\n" << *NewF);
  }

  return NewF;
}

LoopUnrollResult BBInterleave::tryToUnrollBBs(
    Loop *L, LoopInfo *LI, BasicBlock *DominatingBlockIn,
    BasicBlock *PostDominatingBlockIn, const ArrayRef<BasicBlock *> &BBArr,
    DominatorTree &DT, PostDominatorTree &PDT,
    SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> &VMaps,
    SmallVector<std::unique_ptr<ValueToValueMapTy>, 4> &ReverseVMaps,
    SmallPtrSet<Instruction *, 8> &DivergentBranchesIn) {
  this->DivergentBranches = DivergentBranchesIn;
  this->BBsToCoarsen = BBVec(BBArr.begin(), BBArr.end());
  this->PostDominatingBlock = PostDominatingBlockIn;
  this->DominatingBlock = DominatingBlockIn;
  this->F = PostDominatingBlock->getParent();

  this->LI = LI;
  this->TheLoop = L;

  this->DT = &DT;
  this->PDT = &PDT;

  auto &Ctx = F->getContext();

  populateDivergentRegions();

  // TODO Pretty bad, SE is also invalidated but I /think/ we dont need it any
  // more
  DT.recalculate(*F);

  // TODO handle convergent insts properly (e.g. __syncthreads())

  // Sort the BBs so that they are cloned and inserted in the original order
  std::sort(BBsToCoarsen.begin(), BBsToCoarsen.end(),
            [&](BasicBlock *A, BasicBlock *B) {
              return std::distance(F->begin(), A->getIterator()) <
                     std::distance(F->begin(), B->getIterator());
            });

  // Clone the bbs to use the blocks for divergent regions
  unsigned It = 0;
  for (auto &DR : DivergentRegions) {
    // TODO We can insert the DR before or after the convergent region - PGO
    // would help with this decision We could either also set the DR after the
    // end of the CR for cases where the DR is especially infrequent. Alos,
    // maybe it should depend on whether we use dynamic convergence
    Function::iterator InsertBefore = std::next(DR.To->getIterator());
    std::string Suffix = ".drs." + std::to_string(It++);
    for (BasicBlock *BB : BBsToCoarsen) {
      BasicBlock *NewBB = CloneBasicBlock(BB, *DR.DRVMap, Suffix, F);
      NewBB->moveBefore(InsertBefore);
      (*DR.DRVMap)[BB] = NewBB;
      DR.ClonedBlocks.push_back(NewBB);
    }

    remapInstructionsInBlocks(DR.ClonedBlocks, *DR.DRVMap);
  }

  // Interleave instructions
  SmallVector<SmallVector<Instruction *>> ClonedInsts;
  ClonedInsts.reserve(Factor);
  for (unsigned I = 0; I < Factor; I++)
    ClonedInsts.push_back({});

  for (BasicBlock *BB : BBsToCoarsen) {
    SmallVector<Instruction *> ToClone;
    for (Instruction &I : *BB) {
      ToClone.push_back(&I);
    }
    for (Instruction *I : ToClone) {
      Instruction *LastI = I;
      (*VMaps[0])[I] = I;
      (*ReverseVMaps[0])[I] = I;

      if (InterProceduralInterleavingLevel) {
        if (KmpcParallel51Interleave(ORE, Factor, UseDynamicConvergence,
                                     InterProceduralInterleavingLevel - 1)
                .tryToInterleave(I))
          continue;

        Function *InterleavableCallee = getInterleavableCallee(I);
        Function *UnrolledF = nullptr;
        if (InterleavableCallee)
          UnrolledF = FunctionInterleave(ORE, Factor, UseDynamicConvergence,
                                         InterProceduralInterleavingLevel - 1)
                          .tryToInterleaveFunction(InterleavableCallee);
        if (UnrolledF) {
          auto *CB = cast<CallBase>(I);
          SmallVector<Value *> Args;
          FunctionType *FTy = InterleavableCallee->getFunctionType();
          for (unsigned ArgN = 0; ArgN < FTy->getNumParams(); ArgN++) {
            Value *Arg = CB->getArgOperand(ArgN);
            for (unsigned It = 0; It < Factor; It++) {
              Value *CoarsenedArg = (*VMaps[It])[Arg];
              if (!CoarsenedArg) {
                assert(!isa<Instruction>(Arg) ||
                       find(BBsToCoarsen,
                            (dyn_cast<Instruction>(Arg)->getParent())) ==
                           BBsToCoarsen.end());
                CoarsenedArg = Arg;
              }
              Args.push_back(CoarsenedArg);
            }
          }
          CallInst::Create(UnrolledF, Args, "", I);
          I->eraseFromParent();
          continue;
        }
      }

      bool IsTerminator = I->isTerminator();
      for (unsigned It = 1; It < Factor; It++) {
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
        Cloned->cloneDebugInfoFrom(I);
        ClonedInsts[It].push_back(Cloned);
        (*VMaps[It])[I] = Cloned;
        (*ReverseVMaps[It])[Cloned] = I;
        LastI = Cloned;
      }
    }
  }
  if (UseDynamicConvergence) {
    for (auto *BI : ConvergentToDivergentEdges) {
      Instruction *DivergentCond = nullptr;
      assert(BI && !BI->isConditional());
      auto *DivBr = BI->getSuccessor(0)->getTerminator();
      if (auto *Sw = dyn_cast<SwitchInst>(DivBr))
        DivergentCond = cast<Instruction>(Sw->getCondition());
      else if (auto *BI = dyn_cast<BranchInst>(DivBr))
        DivergentCond = cast<Instruction>(BI->getCondition());
      else
        llvm_unreachable(
            "Divergent branches must be either branch or switch insts");

      // The current state:
      //
      // ConvergentBlock:
      //   ...
      //   br %DivergentEntry ( = BI)
      // DivergentEntry:
      //   ...
      //   br %cond %... ( = DivBr)
      //
      // We change `BI` to be a conditional branch with the condition of whether
      // all the coarsened iterations agreed on the `DivBr` condition
      Value *AllSame = nullptr;
      Value *Cond = DivergentCond;
      IRBuilder<> Builder(BI);
      for (unsigned It = 1; It < Factor; It++) {
        // The condition has to be an instruction otherwise it cannot be
        // divergent
        auto *CoarsenedCond = cast<Instruction>((*VMaps[It])[Cond]);
        auto *Same =
            Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, CoarsenedCond, Cond);
        if (AllSame)
          AllSame = Builder.CreateAnd(Same, AllSame);
        else
          AllSame = Same;
      }
      // If the branches agree on the condition, branch to the convergent
      // region, else, we branch to the divergent region (we will insert that
      // successor later)
      Builder.CreateCondBr(AllSame, BI->getSuccessor(0), BI->getSuccessor(0));
      BI->eraseFromParent();
    }
  }

  for (unsigned It = 1; It < Factor; It++)
    for (Instruction *I : ClonedInsts[It])
      RemapInstruction(I, *VMaps[It],
                       RemapFlags::RF_IgnoreMissingLocals |
                           RemapFlags::RF_NoModuleLevelChanges);

  DT.recalculate(*F); // TODO another recalculation...
  // Find all values used in the divergent region but defined outside and demote
  // them to memory - now we can change the CFG easier.
  demoteDRRegs();

  for (auto &DR : DivergentRegions) {
    ValueToValueMapTy &ReverseDRLVMap = *DR.ReverseDRVMap;
    for (auto M : *DR.DRVMap) {
      ReverseDRLVMap[cast<Value>(M.second)] = const_cast<Value *>(M.first);
    }
  }

  IRBuilder<> PreheaderBuilder(DominatingBlock->getTerminator());

  // Hook into the the blocks from the DR loop to do the divergent part of the
  // coarsened computation. Generates Intro and Outro blocks which bring in the
  // appropriate coarsened values into the DR in the uncoarsened part, and then
  // bring out those values for use in the subsequent coarsened computation.
  Type *CoarsenedIdentifierTy = IntegerType::getInt32Ty(Ctx);
  for (auto &DR : DivergentRegions) {
    auto &DivergentRegionsVMap = *DR.DRVMap;
    auto &ReverseDRLVMap = *DR.ReverseDRVMap;
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
    for (unsigned It = 0; It < Factor; It++) {
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
        if (auto *AI = cast_or_null<AllocaInst>(
                DemotedRegsVMap.lookup(CoarsenedValue))) {
          CoarsenedValue = IntroBuilder.CreateLoad(
              CoarsenedValue->getType(), AI, AI->getName() + ".demoted.reload");
        }
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
        // TODO we can "forward" these using phi nodes through the DR iterations
        // and only store them in the final outro. This will significantly
        // simplify the re-promoted values (currently we get a mess of phis with
        // some undef's, see defined-in-used-outside-dr.ll)
      }
      if (It == Factor - 1) {
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

  return LoopUnrollResult::PartiallyUnrolled;
}

void BBInterleave::cleanup() {
  // Now that we have done the plumbing around the divergent regions loop, erase
  // the remainders
  for (auto &DR : DivergentRegions) {
    BBSet UnusedDRBlocks(DR.ClonedBlocks.begin(), DR.ClonedBlocks.end());
    for (auto &DR : DivergentRegions) {
      auto MappedDRBlocks =
          mapContainer<BBSet, BasicBlock>(DR.Blocks, *DR.DRVMap);
      set_subtract(UnusedDRBlocks, MappedDRBlocks);
    }
    SmallVector<BasicBlock *> Tmp(UnusedDRBlocks.begin(), UnusedDRBlocks.end());
    DeleteDeadBlocks(Tmp);
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
  for (auto &DR : DivergentRegions) {
    DRTmpStorage.push_back(DR.IdentPtr);
  }
  DT->recalculate(*F); // TODO another recalculation...
  PromoteMemToReg(DRTmpStorage, *DT);
}

LoopUnrollResult LoopUnrollAndInterleave::tryToUnrollAndInterleaveLoop(
    Loop *L, DominatorTree &DT, LoopInfo &LI, ScalarEvolution &SE,
    PostDominatorTree &PDT) {
  this->LI = &LI;
  this->TheLoop = L;

  Function *F = L->getHeader()->getParent();
  auto &Ctx = F->getContext();

  LLVM_DEBUG(DBGS << "F[" << F->getName() << "] Loop %"
                  << L->getHeader()->getName() << " "
                  << "Parallel=" << L->isAnnotatedParallel() << "\n");

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "Before unroll and interleave:\n" << *F);
  }

  if (!collectDivergentBranches(L, &LI))
    return LoopUnrollResult::Unmodified;

  auto LoopBounds = L->getBounds(SE);
  if (LoopBounds == std::nullopt) {
    LLVM_DEBUG(DBGS_FAIL << "Unable to find loop bounds of the omp workshare "
                            "loop, not coarsening\n");
    return LoopUnrollResult::Unmodified;
  }

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

  // Clone the loop once more to use as an epilogue, the original one will be
  // coarsened in-place
  ValueToValueMapTy EpilogueVMap;
  SmallVector<BasicBlock *> EpilogueLoopBlocks;
  Loop *EpilogueLoop =
      cloneLoopWithPreheader(ExitBlock, &F->getEntryBlock(), L, EpilogueVMap,
                             ".epilogue", &LI, &DT, EpilogueLoopBlocks);
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
  VMaps.reserve(Factor);
  ReverseVMaps.reserve(Factor);
  for (unsigned I = 0; I < Factor; I++) {
    VMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
    ReverseVMaps.emplace_back(std::make_unique<ValueToValueMapTy>());
  }

  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());

  // The new Step is UnrollFactor * OriginalStep
  Value *IVStepVal = LoopBounds->getStepValue();
  if (Instruction *IVStepInst = dyn_cast_or_null<Instruction>(IVStepVal)) {
    Value *NewStep = PreheaderBuilder.CreateMul(
        IVStepVal, ConstantInt::get(IVStepVal->getType(), Factor),
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
        Factor * IVStepConst->getValue());
    Instruction *StepInst = &LoopBounds->getStepInst();
    for (unsigned It = 0; It < StepInst->getNumOperands(); It++)
      if (StepInst->getOperand(It) == IVStepVal)
        StepInst->setOperand(It, NewStep);
  }

  // Set up new initial IV values, for now we do initial + stride, initial + 2 *
  // stride, ..., initial + (UnrollFactor - 1) * stride
  (*VMaps[0])[InitialIVVal] = InitialIVVal;
  (*ReverseVMaps[0])[InitialIVVal] = InitialIVVal;
  for (unsigned I = 1; I < Factor; I++) {
    Value *CoarsenedInitialIV;
    Value *MultipliedStep = PreheaderBuilder.CreateMul(
        IVStepVal, ConstantInt::get(IVStepVal->getType(), I));
    CoarsenedInitialIV =
        PreheaderBuilder.CreateAdd(InitialIVVal, MultipliedStep,
                                   "initial.iv.coarsened." + std::to_string(I));
    (*VMaps[I])[InitialIVVal] = CoarsenedInitialIV;
    (*ReverseVMaps[I])[CoarsenedInitialIV] = InitialIVVal;
  }

  BasicBlock *EpiloguePH = cast<BasicBlock>(EpilogueVMap[Preheader]);
  EpilogueLoopBlocks.erase(std::find(EpilogueLoopBlocks.begin(),
                                     EpilogueLoopBlocks.end(), EpiloguePH));
  for (Instruction &I : *Preheader) {
    EpilogueVMap.erase(&I);
  }
  EpilogueVMap.erase(Preheader);
  remapInstructionsInBlocks(EpilogueLoopBlocks, EpilogueVMap);
  if (LI.getLoopFor(EpiloguePH))
    LI.removeBlock(EpiloguePH);
  EpiloguePH->eraseFromParent();

  // Plumbing around the coarsened and epilogue loops

  for (auto &PN : ExitBlock->phis()) {
    Value *IncomingVal = PN.getIncomingValueForBlock(CombinedLatchExiting);
    Value *MappedIncoming = EpilogueVMap[IncomingVal];
    // If the value is defined outside the loop.
    if (!MappedIncoming)
      continue;

    PN.addIncoming(MappedIncoming,
                   cast<BasicBlock>(EpilogueVMap[CombinedLatchExiting]));
  }

  // Note this calculation works for both increasing and decreasing loops.
  //
  // Calculate the start value for the epilogue loop, should be:
  // Start + (ceil((End - Start) / Stride) / UnrollFactor) * UnrollFactor *
  // Stride.
  // I.e. when we are done with our iterations of the coarsened loop.
  Value *EpilogueStart;
  Value *Start = &LoopBounds->getInitialIVValue();
  // Value *End = GetEnd(&LoopBounds->getFinalIVValue()); // Defined above.
  Value *Stride = LoopBounds->getStepValue();
  Value *One = ConstantInt::get(Start->getType(), 1);
  Value *UnrollFactorCst = ConstantInt::get(Start->getType(), Factor);
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
    IRBuilder<> LatchBuilder(BackEdge);
    Value *IsAtEpilogueStart = LatchBuilder.CreateCmp(
        CmpInst::Predicate::ICMP_EQ, &LoopBounds->getStepInst(), EpilogueStart,
        "is.epilogue.start");
    BasicBlock *EndCheckBB = EpilogueFrom = BasicBlock::Create(
        Ctx, "coarsened.end.check", F, EpilogueLoop->getHeader());
    LatchBuilder.CreateCondBr(IsAtEpilogueStart, EndCheckBB,
                              TheLoop->getHeader());
    Instruction *Cloned = BackEdge->clone();
    Cloned->insertInto(EndCheckBB, EndCheckBB->begin());
    Cloned->cloneDebugInfoFrom(BackEdge);
    BackEdge->eraseFromParent();
    ExitBlock->replacePhiUsesWith(CombinedLatchExiting, EndCheckBB);
    Cloned->replaceSuccessorWith(TheLoop->getHeader(),
                                 EpilogueLoop->getHeader());
    EpilogueLoop->getHeader()->replacePhiUsesWith(Preheader, EndCheckBB);
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

  tryToUnrollBBs(L, &LI, Preheader, CombinedLatchExiting, TheLoop->getBlocks(),
                 DT, PDT, VMaps, ReverseVMaps, DivergentBranches);
  cleanup();

  if (getenv("UNROLL_AND_INTERLEAVE_DUMP")) {
    LLVM_DEBUG(DBGS << "After unroll and interleave:\n" << *F);
  }

  setLoopAlreadyCoarsened(L);
  setLoopAlreadyCoarsened(EpilogueLoop);

  LLVM_DEBUG(DBGS << "SUCCESS\n");

  return LoopUnrollResult::PartiallyUnrolled;
}

PreservedAnalyses LoopUnrollAndInterleavePass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  bool Changed = false;

  for (Function &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    auto &LI = FAM.getResult<LoopAnalysis>(F);
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

    SmallVector<Loop *> ToUAI;
    for (auto &L : LI.getLoopsInPreorder()) {
      if (getLoopAlreadyCoarsened(L)) {
        LLVM_DEBUG(DBGS_DISABLED << "Coarsening disabled\n");
        continue;
      }
      auto Factor = getLoopCoarseningFactor(L);
      if (Factor == 0) {
        LLVM_DEBUG(DBGS_DISABLED << "Coarsening metadata missing - ignoring\n");
        continue;
      }
      ToUAI.push_back(L);
    }
    // TODO No nested UAI'ing allowed - either support it or check
    for (auto *L : ToUAI) {
      auto Factor = getLoopCoarseningFactor(L);
      if (Factor == 0) {
        LLVM_DEBUG(DBGS_DISABLED << "Coarsening metadata missing - ignoring\n");
        continue;
      }
      if (FactorOpt.getNumOccurrences() > 0) {
        LLVM_DEBUG(DBGS << "Setting factor from cl opt\n");
        Factor = FactorOpt;
      }
      if (char *Env = getenv("UNROLL_AND_INTERLEAVE_FACTOR")) {
        LLVM_DEBUG(DBGS << "Setting factor from env var\n");
        StringRef(Env).getAsInteger(10, Factor);
      }
      if (Factor <= 1) {
        LLVM_DEBUG(DBGS_DISABLED << "Coarsening factor of 1 or 0 - ignoring\n");
        continue;
      }
      bool UseDynamicConvergence = UseDynamicConvergenceOpt;
      if (char *Env = getenv("UNROLL_AND_INTERLEAVE_DYNAMIC_CONVERGENCE")) {
        unsigned Int = 0;
        StringRef(Env).getAsInteger(10, Int);
        UseDynamicConvergence = Int;
      }

      int Level = getLoopCoarseningLevel(L);

      OptimizationRemarkEmitter ORE(&F);

      Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
      Changed |= simplifyLoop(L, &DT, &LI, &SE, /*AC=*/nullptr,
                              /*MSSAU=*/nullptr, /*PreserveLCSSA=*/true);
      Changed |=
          LoopUnrollAndInterleave(ORE, Factor, UseDynamicConvergence, Level)
              .tryToUnrollAndInterleaveLoop(L, DT, LI, SE, PDT) !=
          LoopUnrollResult::Unmodified;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

namespace llvm {
bool loopUnrollAndInterleave(OptimizationRemarkEmitter &ORE,
                             unsigned UnrollFactor, bool UseDynamicConvergence,
                             int InterProceduralInterleavingLevel, Loop *L,
                             DominatorTree &DT, LoopInfo &LI,
                             ScalarEvolution &SE, PostDominatorTree &PDT) {
  if (UnrollFactor <= 1) {
    LLVM_DEBUG(DBGS << "Unroll factor of 1 or 0 - ignoring\n");
    return false;
  }
  return LoopUnrollAndInterleave(ORE, UnrollFactor, UseDynamicConvergenceOpt,
                                 InterProceduralInterleavingLevel)
             .tryToUnrollAndInterleaveLoop(L, DT, LI, SE, PDT) !=
         LoopUnrollResult::Unmodified;
}
} // namespace llvm
