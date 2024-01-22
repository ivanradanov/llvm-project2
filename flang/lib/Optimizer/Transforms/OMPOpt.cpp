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
/// TODO there is some in-place operation updating which we should notify the
/// rewriters about

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_HLFIROMPOPT
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir
namespace fir {
#define GEN_PASS_DEF_FIROMPOPT
#define GEN_PASS_DEF_LLVMOMPOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-omp-opt"
#define TAG ("[" DEBUG_TYPE "] ")

using llvm::dbgs;
using namespace mlir;

static inline mlir::Type getLlvmPtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

/// Return the LLVMFuncOp corresponding to omp_target_alloc
///
/// void* omp_target_alloc(size_t size, int device_num);
///
/// TODO is the abi correct for all targets?
static mlir::LLVM::LLVMFuncOp getOmpTargetAlloc(ModuleOp module) {
  if (mlir::LLVM::LLVMFuncOp mallocFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("omp_target_alloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto i64Ty = mlir::IntegerType::get(module->getContext(), 64);
  auto i32Ty = mlir::IntegerType::get(module->getContext(), 32);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      moduleBuilder.getUnknownLoc(), "omp_target_alloc",
      mlir::LLVM::LLVMFunctionType::get(
          LLVM::LLVMPointerType::get(module->getContext()), {i64Ty, i32Ty},
          /*isVarArg=*/false));
}

/// Return the LLVMFuncOp corresponding to omp_target_free
///
/// void omp_target_free(void *device_ptr, int device_num);
///
/// TODO is the abi correct for all targets?
static mlir::LLVM::LLVMFuncOp getOmpTargetFree(ModuleOp module) {
  if (mlir::LLVM::LLVMFuncOp freeFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("omp_target_free"))
    return freeFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto i32Ty = mlir::IntegerType::get(module->getContext(), 32);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      moduleBuilder.getUnknownLoc(), "omp_target_free",
      mlir::LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(module->getContext()),
          {getLlvmPtrType(module->getContext()), i32Ty},
          /*isVarArg=*/false));
}

static bool isRuntimeCall(Operation *op) {
  if (auto callOp = dyn_cast<fir::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    auto *func = op->getParentOfType<ModuleOp>().lookupSymbol(*callee);
    // TODO need to insert a check here whether it is a call we can actually
    // parallelize currently
    if (func->getAttr(fir::FIROpsDialect::getFirRuntimeAttrName()))
      return true;
  }
  return false;
}

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
  if (isRuntimeCall(op)) {
    return true;
  }
  // We cannot parallise anything else
  return false;
}

// TODO we dont need to isolate random operations before a runtime call. We
// still need to isolate things before parallel regions because we cannot be
// sure whether they are sinkable into the parallel region, whereas the reloads
// from temporaries are always sinkable.
static std::optional<std::tuple<Operation *, bool, bool>>
getNestedOpToIsolate(omp::TargetOp targetOp) {
  auto *targetBlock = &targetOp.getRegion().front();
  for (auto &op : *targetBlock) {
    bool first = &op == &*targetBlock->begin();
    bool last = op.getNextNode() == targetBlock->getTerminator();
    if (first && last)
      return std::nullopt;

    if (isa<omp::TeamsOp, omp::ParallelOp>(&op))
      return {{&op, first, last}};

    if (isRuntimeCall(&op)) {
      return {{&op, first, last}};
    }
  }
  return std::nullopt;
}

mlir::LLVM::ConstantOp genI32Constant(mlir::Location loc,
                                      mlir::RewriterBase &rewriter, int value) {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

mlir::LLVM::ConstantOp genI64Constant(mlir::Location loc,
                                      mlir::RewriterBase &rewriter, int value) {
  mlir::Type i64Ty = rewriter.getI64Type();
  mlir::IntegerAttr attr = rewriter.getI64IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, attr);
}

/// If multiple coexecutes are nested in a target regions, we will need to split
/// the target region, but we want to preserve the data semantics of the
/// original data region - we split the target region into a target_data{target}
/// nest
///
/// TODO need to handle special case where the target regions had a always copy
/// or always free map types (or something similar, I forgot how they are
/// called); I think these just need to be removed from the inner data region
/// map
mlir::LogicalResult splitTargetData(omp::TargetOp targetOp,
                                    RewriterBase &rewriter) {
  auto loc = targetOp->getLoc();
  if (targetOp.getMapOperands().empty()) {
    LLVM_DEBUG(dbgs() << TAG << "target region has no data maps\n");
    return mlir::failure();
  }

  // This is from before we started running this on the llvm level
  //
  // unsigned coexecuteNum = 0;
  // targetOp->walk([&](omp::CoexecuteOp) { coexecuteNum++; });
  // if (coexecuteNum < 2) {
  //   LLVM_DEBUG(
  //       dbgs() << TAG
  //              << "target region has fewer than two nested coexecutes\n");
  //   return mlir::failure();
  // }

  rewriter.setInsertionPoint(targetOp);
  auto dataOp = rewriter.create<omp::DataOp>(
      loc, targetOp.getIfExpr(), targetOp.getDevice(),
      /*use_device_ptr=*/mlir::ValueRange(),
      /*use_device_addr=*/mlir::ValueRange(), targetOp.getMapOperands());
  rewriter.createBlock(&dataOp.getRegion(), dataOp.getRegion().begin(), {}, {});
  auto newTargetOp = rewriter.create<omp::TargetOp>(
      loc, targetOp.getIfExpr(), targetOp.getDevice(),
      targetOp.getThreadLimit(), targetOp.getTripCount(),
      targetOp.getNowaitAttr(), targetOp.getMapOperands(),
      targetOp.getNumTeamsLower(), targetOp.getNumTeamsUpper(),
      targetOp.getTeamsThreadLimit(), targetOp.getNumThreads());
  rewriter.create<omp::TerminatorOp>(loc);

  rewriter.inlineRegionBefore(targetOp.getRegion(), newTargetOp.getRegion(),
                              newTargetOp.getRegion().begin());

  rewriter.replaceOp(targetOp, newTargetOp);

  return mlir::success();
}

/// Borrowed from CodeGen.cpp
///
/// Helper function for generating the LLVM IR that computes the distance
/// in bytes between adjacent elements pointed to by a pointer
/// of type \p ptrTy. The result is returned as a value of \p idxTy integer
/// type.
static mlir::Value computeElementDistance(mlir::Location loc,
                                          mlir::Type llvmObjectType,
                                          mlir::Type idxTy,
                                          mlir::RewriterBase &rewriter) {
  // Note that we cannot use something like
  // mlir::LLVM::getPrimitiveTypeSizeInBits() for the element type here. For
  // example, it returns 10 bytes for mlir::Float80Type for targets where it
  // occupies 16 bytes. Proper solution is probably to use
  // mlir::DataLayout::getTypeABIAlignment(), but DataLayout is not being set
  // yet (see llvm-project#57230). For the time being use the '(intptr_t)((type
  // *)0 + 1)' trick for all types. The generated instructions are optimized
  // into constant by the first pass of InstCombine, so it should not be a
  // performance issue.
  auto llvmPtrTy = ::getLlvmPtrType(llvmObjectType.getContext());
  auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPtrTy);
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      loc, llvmPtrTy, llvmObjectType, nullPtr,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{1});
  return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, gep);
}

/// Removes unused operands/args from the omp.target op
///
/// TODO is this ok? are there map types that need to be preserved even though
/// we do not use them in the target region?
static void minimizeArgs(omp::TargetOp targetOp) {
  auto *targetBlock = &targetOp.getRegion().front();
  for (unsigned i = 0; i < targetBlock->getNumArguments();) {
    if (targetBlock->getArgument(i).use_empty()) {
      targetBlock->eraseArgument(i);
      targetOp.getMapOperandsMutable().erase(i);
    } else {
      i++;
    }
  }
}

struct TempOmpVar {
  omp::MapInfoOp from, to;
};

static bool isPtr(Type ty) {
  return isa<fir::ReferenceType>(ty) || isa<LLVM::LLVMPointerType>(ty);
}

static Type getPtrTypeForOmp(Type ty) {
  if (isPtr(ty))
    return LLVM::LLVMPointerType::get(ty.getContext());
  else
    return fir::LLVMPointerType::get(ty);
}

static bool isRecomputableAfterFission(Operation *op, Operation *splitBefore) {
  // TODO do we need hlfir.declare?
  if (isa<fir::DeclareOp>(op))
    return true;

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    return false;
  }
  interface.getEffects(effects);
  if (effects.empty())
    return true;

  return false;
}

/// Isolates the first target{parallel|teams{}} nest in its own omp.target op
///
/// TODO lifetime analysis to lower amount of memory required for temporaries
/// TODO flow mincut analysis to figure out the lowest amount of memory we need
/// to allocate for the crossing
/// TODO try to sink top level operations in the parallel regions in order to
/// avoid having to split them off in a separate omp.target
/// TODO when we generate loads for the temporaries these should be in the
/// parallel region
omp::TargetOp fissionTarget(omp::TargetOp targetOp, RewriterBase &rewriter) {
  auto tuple = getNestedOpToIsolate(targetOp);
  if (!tuple) {
    LLVM_DEBUG(dbgs() << TAG << "No op to isolate\n");
    return nullptr;
  }

  Operation *toIsolate = std::get<0>(*tuple);
  bool splitBefore = !std::get<1>(*tuple);
  bool splitAfter = !std::get<2>(*tuple);

  LLVM_DEBUG(dbgs() << TAG << "Will isolate " << *toIsolate << " from "
                    << targetOp << "\n");

  auto loc = targetOp->getLoc();
  auto llvmPtrTy = LLVM::LLVMPointerType::get(targetOp.getContext());
  auto allocTemp = [&](Type ty) -> TempOmpVar {
    Value alloc;
    Type allocType;
    if (isPtr(ty)) {
      auto one =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
      allocType = llvmPtrTy;
      alloc = rewriter.create<LLVM::AllocaOp>(loc, llvmPtrTy, allocType, one);
    } else {
      allocType = ty;
      alloc = rewriter.create<fir::AllocaOp>(loc, allocType);
    }
    SmallVector<Value> bounds = {rewriter.create<omp::DataBoundsOp>(
        loc, rewriter.getType<mlir::omp::DataBoundsType>(),
        genI64Constant(loc, rewriter, 0), genI64Constant(loc, rewriter, 1),
        nullptr, nullptr, false, nullptr)};
    auto getMapInfo = [&](auto mappingFlags, const char *name) {
      return rewriter.create<omp::MapInfoOp>(
          loc, alloc.getType(), alloc, allocType, /*var_ptr_ptr=*/nullptr,
          /*members*/ ValueRange(), bounds,
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, false),
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  mappingFlags)),
          rewriter.getAttr<mlir::omp::VariableCaptureKindAttr>(
              mlir::omp::VariableCaptureKind::ByCopy),
          rewriter.getStringAttr(name));
    };
    auto mapInfoFrom =
        getMapInfo(llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM,
                   "__flang_coexecute_from");
    auto mapInfoTo =
        getMapInfo(llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO,
                   "__flang_coexecute_to");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(targetOp);
    return {mapInfoFrom, mapInfoTo};
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
    SmallVector<fir::LoadOp> toClone;
    auto preMapOperands = SmallVector<Value>(targetOp.getMapOperands());
    auto postMapOperands = SmallVector<Value>(targetOp.getMapOperands());
    SmallVector<Value> requiredVals;
    SmallPtrSet<Operation *, 8> nonRecomputable;
    for (auto it = targetBlock->begin(); it != splitBefore->getIterator();
         it++) {
      // TODO this can be made more generic, e.g. fir.declare is also used on te
      // args
      // Skip if it is already a load from a mapped argument to the target
      // region
      //
      // Disabled for now because I am not sure whether we may not have memory
      // that aliases memory that is written to in a parallel region. We would
      // like to read and stash that in a new temporary in that case.
      //
      // if (auto loadOp = dyn_cast<fir::LoadOp>(it))
      //   if (auto blockArg = dyn_cast<BlockArgument>(loadOp.getMemref()))
      //     if (blockArg.getOwner() == targetBlock) {
      //       toClone.push_back(loadOp);
      //       continue;
      //     }
      for (auto res : it->getResults())
        if (usedOutsideSplit(res, splitBefore))
          requiredVals.push_back(res);
      if (!isRecomputableAfterFission(&*it, splitBefore))
        nonRecomputable.insert(&*it);
    }

    SmallPtrSet<Operation *, 8> toCache;
    SmallPtrSet<Operation *, 8> toRecompute;
    std::function<void(Value)> collectNonRecomputableDeps = [&](Value v) {
      Operation *op = v.getDefiningOp();

      if (!op) {
        assert(v.cast<BlockArgument>().getOwner()->getParentOp() == targetOp);
        return;
      }

      if (nonRecomputable.contains(op)) {
        toCache.insert(op);
        return;
      }

      toRecompute.insert(op);
      for (auto opr : op->getOperands())
        collectNonRecomputableDeps(opr);
    };
    for (auto requiredVal : requiredVals)
      collectNonRecomputableDeps(requiredVal);

    for (Operation *op : toCache) {
      for (auto res : op->getResults()) {
        auto alloc = allocTemp(res.getType());
        allocs.push_back({res, preMapOperands.size()});
        preMapOperands.push_back(alloc.from);
        postMapOperands.push_back(alloc.to);
      }
    }

    // Split into two blocks with additional mapppings for the values to be
    // passed in memory across the regions

    rewriter.setInsertionPoint(targetOp);
    auto preTargetOp = rewriter.create<omp::TargetOp>(
        loc, targetOp.getIfExpr(), targetOp.getDevice(),
        targetOp.getThreadLimit(), targetOp.getTripCount(),
        targetOp.getNowait(), preMapOperands, targetOp.getNumTeamsLower(),
        targetOp.getNumTeamsUpper(), targetOp.getTeamsThreadLimit(),
        targetOp.getNumThreads());
    auto *preTargetBlock = rewriter.createBlock(
        &preTargetOp.getRegion(), preTargetOp.getRegion().begin(), {}, {});
    IRMapping preMapping;
    for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
      auto originalArg = targetBlock->getArgument(i);
      auto newArg = preTargetBlock->addArgument(originalArg.getType(),
                                                originalArg.getLoc());
      preMapping.map(originalArg, newArg);
    }
    for (auto it = targetBlock->begin(); it != splitBefore->getIterator(); it++)
      rewriter.clone(*it, preMapping);
    for (auto tup : allocs) {
      auto original = std::get<0>(tup);
      Value toStore = preMapping.lookup(original);
      auto newArg = preTargetBlock->addArgument(
          getPtrTypeForOmp(original.getType()), original.getLoc());
      if (isPtr(original.getType())) {
        // TODO maybe we should use fir convertop here, but a LLVM::LLVMPointer
        // is currently not considered convertible, and there is no allocaop for
        // fir::LLVMPointer, we can change `isPointerCompatible` to change the
        // ConvertOp behaviour
        if (!isa<LLVM::LLVMPointerType>(toStore.getType()))
          toStore = rewriter
                        .create<UnrealizedConversionCastOp>(loc, llvmPtrTy,
                                                            ValueRange(toStore))
                        .getResult(0);
        rewriter.create<LLVM::StoreOp>(loc, toStore, newArg);
      } else {
        rewriter.create<fir::StoreOp>(loc, toStore, newArg);
      }
    }
    rewriter.create<omp::TerminatorOp>(loc);

    rewriter.setInsertionPoint(targetOp);
    auto postTargetOp = rewriter.create<omp::TargetOp>(
        loc, targetOp.getIfExpr(), targetOp.getDevice(),
        targetOp.getThreadLimit(), targetOp.getTripCount(),
        targetOp.getNowait(), postMapOperands, targetOp.getNumTeamsLower(),
        targetOp.getNumTeamsUpper(), targetOp.getTeamsThreadLimit(),
        targetOp.getNumThreads());
    auto *postTargetBlock = rewriter.createBlock(
        &postTargetOp.getRegion(), postTargetOp.getRegion().begin(), {}, {});
    IRMapping postMapping;
    for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
      auto originalArg = targetBlock->getArgument(i);
      auto newArg = postTargetBlock->addArgument(originalArg.getType(),
                                                 originalArg.getLoc());
      postMapping.map(originalArg, newArg);
    }
    // See above
    assert(toClone.empty());
    // for (auto loadOp : toClone)
    //   rewriter.clone(*loadOp, postMapping);
    for (auto tup : allocs) {
      auto original = std::get<0>(tup);
      Value newArg = postTargetBlock->addArgument(
          getPtrTypeForOmp(original.getType()), original.getLoc());
      Value restored;
      if (isPtr(original.getType())) {
        restored = rewriter.create<LLVM::LoadOp>(loc, llvmPtrTy, newArg);
        if (!isa<LLVM::LLVMPointerType>(original.getType()))
          restored = rewriter
                         .create<UnrealizedConversionCastOp>(
                             loc, original.getType(), ValueRange(restored))
                         .getResult(0);
      } else {
        restored = rewriter.create<fir::LoadOp>(loc, newArg);
      }
      postMapping.map(original, restored);
    }
    for (auto it = targetBlock->begin(); it != splitBefore->getIterator(); it++)
      if (toRecompute.contains(&*it))
        rewriter.clone(*it, postMapping);
    for (auto it = splitBefore->getIterator(); it != targetBlock->end(); it++)
      rewriter.clone(*it, postMapping);

    rewriter.eraseOp(targetOp);

    minimizeArgs(preTargetOp);
    minimizeArgs(postTargetOp);

    return postMapping.lookup(splitBefore);
  };

  if (splitBefore && splitAfter) {
    auto *newSplitBefore = splitBeforeOp(toIsolate);
    newSplitBefore = splitBeforeOp(newSplitBefore->getNextNode());
    return cast<omp::TargetOp>(newSplitBefore->getParentOp());
  }
  if (splitBefore) {
    splitBeforeOp(toIsolate);
    return nullptr;
  }
  if (splitAfter) {
    auto *newSplitBefore = splitBeforeOp(toIsolate->getNextNode());
    return cast<omp::TargetOp>(newSplitBefore->getParentOp());
  }
  llvm_unreachable("we should not have had an op to isolate");
}

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
      rewriter.setInsertionPoint(coexecuteOp);
      auto distributeOp = rewriter.create<omp::DistributeOp>(teamsLoc);
      rewriter.createBlock(&distributeOp.getRegion(),
                           distributeOp.getRegion().begin(), {}, {});
      auto parallelOp = rewriter.create<omp::ParallelOp>(
          teamsLoc, teamsOp.getIfExpr(), /*num_threads_var=*/nullptr,
          teamsOp.getAllocateVars(), teamsOp.getAllocatorsVars(),
          /*reduction_vars=*/ValueRange(), /*reductions=*/nullptr,
          /*proc_bind_val=*/nullptr);
      rewriter.create<omp::TerminatorOp>(coexecuteLoc);
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
      rewriter.replaceOp(coexecuteOp, parallelOp);
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
LogicalResult fissionCoexecute(omp::CoexecuteOp coexecute,
                               PatternRewriter &rewriter) {
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
struct FissionCoexecute : public OpRewritePattern<omp::CoexecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(omp::CoexecuteOp coexecute,
                                PatternRewriter &rewriter) const override {
    return fissionCoexecute(coexecute, rewriter);
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

static Value getHostValue(Value v, RewriterBase &rewriter, IRMapping &mapping) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return nullptr;
  if (auto loadOp = dyn_cast_or_null<LLVM::LoadOp>(op)) {
    // TODO need to check aliasing and if there arent any stores
    Value mem = loadOp.getAddr();
    auto arg = mem.dyn_cast<BlockArgument>();
    if (!arg)
      return nullptr;
    auto targetOp = dyn_cast<omp::TargetOp>(arg.getOwner()->getParentOp());
    if (!targetOp)
      return nullptr;
    auto argNum = arg.getArgNumber();
    auto mapInfoOp =
        cast<omp::MapInfoOp>(targetOp.getMapOperands()[argNum].getDefiningOp());
    auto hostValue = rewriter.create<LLVM::LoadOp>(
        loadOp->getLoc(), v.getType(), mapInfoOp.getVarPtr());
    mapping.map(v, hostValue);
    return hostValue;
  }
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return nullptr;

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  if (!effects.empty())
    return nullptr;

  unsigned resNum = 0;
  unsigned i = 0;
  for (auto opr : op->getOperands()) {
    if (opr == v)
      resNum = i;
    if (!getHostValue(opr, rewriter, mapping))
      return nullptr;
    i++;
  }

  return rewriter.clone(*op, mapping)->getResult(resNum);
}

/// Hoists out temporary allocations from the target region to the host
///
/// We hoist malloc's which are allocated for temporary storage of arrays
///
/// We need to hoist out alloca's as well because these appear when using the
/// fortran runtime functionss, e.g. @_FortranAAssign which take a pointer to a
/// struct which describes the array being operated on
///
/// TODO should we only do this if we will be splitting this target region, or
/// always?
/// TODO we should probably check that the allocation is freed inside the target
/// region in question (i.e. the pointer does not escape)
/// TODO lifetime analysis to lower amount of memory required for the mem
template <typename AllocOpTy>
struct HoistAllocs : public OpRewritePattern<AllocOpTy> {
  using OpRewritePattern<AllocOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(AllocOpTy allocMemOp,
                                PatternRewriter &rewriter) const override {
    // Only these two types supported currently
    auto callOp = dyn_cast<LLVM::CallOp>(allocMemOp.getOperation());
    auto allocaOp = dyn_cast<LLVM::AllocaOp>(allocMemOp.getOperation());
    assert(callOp || allocaOp);

    if (callOp) {
      if (auto callee = callOp.getCallee()) {
        if (*callee != "malloc")
          return failure();
      } else {
        return failure();
      }
    }

    auto targetOp = dyn_cast<omp::TargetOp>(allocMemOp->getParentOp());
    if (!targetOp)
      return failure();

    auto loc = allocMemOp->getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(targetOp.getContext());

    mlir::LLVM::LLVMFuncOp ompTargetAllocFunc =
        getOmpTargetAlloc(allocMemOp->template getParentOfType<ModuleOp>());
    mlir::LLVM::LLVMFuncOp ompTargetFreeFunc =
        getOmpTargetFree(allocMemOp->template getParentOfType<ModuleOp>());

    rewriter.setInsertionPoint(targetOp);
    Value allocationSize = nullptr;
    if (callOp) {
      allocationSize = callOp.getArgOperands()[0];
    } else if (allocaOp) {
      allocationSize = computeElementDistance(loc, allocaOp.getType(),
                                              rewriter.getI64Type(), rewriter);
      allocationSize = rewriter.create<arith::MulIOp>(loc, allocationSize,
                                                      allocaOp.getArraySize());
    } else {
      llvm_unreachable("Wrong template type");
    }

    IRMapping mapping;
    mlir::Value size = getHostValue(allocationSize, rewriter, mapping);
    if (!size) {
      LLVM_DEBUG(dbgs() << TAG << "Could not get host value of "
                        << allocationSize << "\n");
      return failure();
    }

    Value device = targetOp.getDevice();
    if (!device) {
      // TODO is this the correct way to get the default device?
      device = genI32Constant(loc, rewriter, 0);
    }
    auto newAlloc =
        rewriter
            .create<mlir::LLVM::CallOp>(loc, ompTargetAllocFunc,
                                        SmallVector<Value>({size, device}))
            ->getResult(0);
    auto ompHostAlloca = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, ptrTy, genI64Constant(loc, rewriter, 1));
    rewriter.create<LLVM::StoreOp>(loc, newAlloc, ompHostAlloca);

    SmallVector<Value> bounds = {rewriter.create<omp::DataBoundsOp>(
        loc, rewriter.getType<mlir::omp::DataBoundsType>(),
        genI64Constant(loc, rewriter, 0), genI64Constant(loc, rewriter, 1),
        nullptr, nullptr, false, nullptr)};
    auto mapInfo = rewriter.create<omp::MapInfoOp>(
        loc, ptrTy, ompHostAlloca, ptrTy, /*var_ptr_ptr=*/nullptr,
        /*members*/ ValueRange(), bounds,
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(64, false),
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO)),
        rewriter.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByCopy),
        rewriter.getStringAttr("coexecute_hoisted_malloc"));

    auto *targetBlock = &targetOp.getRegion().front();
    targetOp.getMapOperandsMutable().append({mapInfo});
    auto newArg = targetBlock->addArgument(mapInfo.getType(), loc);
    rewriter.setInsertionPointToStart(targetBlock);
    auto ompDeviceAlloc = rewriter.create<LLVM::LoadOp>(loc, ptrTy, newArg);

    rewriter.setInsertionPointAfter(targetOp);
    rewriter.create<mlir::LLVM::CallOp>(loc, ompTargetFreeFunc,
                                        SmallVector<Value>({newAlloc, device}));

    if (callOp) {
      SmallVector<Operation *> frees;
      for (Operation *user : allocMemOp.getResult().getUsers())
        if (auto callOp = dyn_cast<LLVM::CallOp>(user))
          if (auto callee = callOp.getCallee())
            if (*callee == "free")
              frees.push_back(user);
      for (Operation *user : frees)
        rewriter.eraseOp(user);
    }

    rewriter.replaceAllUsesWith(allocMemOp.getResult(), ompDeviceAlloc);
    rewriter.eraseOp(allocMemOp);

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

class HLFIROMPOptPass : public ::hlfir::impl::HLFIROMPOptBase<HLFIROMPOptPass> {
public:
  void runOnOperation() override;
};

class FIROMPOptPass : public ::fir::impl::FIROMPOptBase<FIROMPOptPass> {
public:
  void runOnOperation() override;
};

class LLVMOMPOptPass : public ::fir::impl::LLVMOMPOptBase<LLVMOMPOptPass> {
public:
  void runOnOperation() override;
};

void HLFIROMPOptPass::runOnOperation() {
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE "-hlfir ===\n");

  Operation *op = getOperation();

  LLVM_DEBUG({
    dbgs() << "Dumping memory effects\n";
    op->walk([](omp::CoexecuteOp coexecute) { dumpMemoryEffects(coexecute); });
  });
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

  // We need to split the coexecute when we have the parallel regions
  // represented as unordered do loops
  {
    RewritePatternSet patterns(&context);
    patterns.insert<FissionCoexecute>(&context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc(), "error in OpenMP optimizations\n");
      signalPassFailure();
    }
  }
  LLVM_DEBUG(dbgs() << TAG << "After coexecute fission:\n" << *op << "\n");
  // We need to convert the unordered do loops to omp.distribute while we still
  // have that representation (later in the pipeline they will be converted to
  // cf and will be unrecoverable)
  {
    RewritePatternSet patterns(&context);
    patterns.insert<TeamsCoexecuteLowering, TeamsCoexecuteToSingle>(&context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc(), "error in OpenMP optimizations\n");
      signalPassFailure();
    }
  }
  LLVM_DEBUG(dbgs() << TAG << "After coexecute lower:\n" << *op << "\n");
  // We must split out the target data before we fission the target regions in
  // order to preserve the memory movement semantics
  //
  // TODO Let us ignore trying to preserve any semantics for now and assume we
  // can break the state of the program after the coexecute...
  // {
  //   SmallVector<omp::TargetOp> targetOps;
  //   op->walk([&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
  //   IRRewriter rewriter(&context);
  //   for (auto targetOp : targetOps)
  //     (void)splitTargetData(targetOp, rewriter);
  // }

  SmallVector<omp::TargetOp> targetOps;
  op->walk([&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
  IRRewriter rewriter(&context);
  for (auto targetOp : targetOps)
    while (targetOp)
      targetOp = fissionTarget(targetOp, rewriter);
}

void LLVMOMPOptPass::runOnOperation() {
  LLVM_DEBUG(dbgs() << "=== Begin " DEBUG_TYPE "-llvm ===\n");
}

/// OpenMP optimizations
std::unique_ptr<Pass> hlfir::createHLFIROMPOptPass() {
  return std::make_unique<HLFIROMPOptPass>();
}
/// OpenMP optimizations
std::unique_ptr<Pass> fir::createFIROMPOptPass() {
  return std::make_unique<FIROMPOptPass>();
}
/// OpenMP optimizations
std::unique_ptr<Pass> fir::createLLVMOMPOptPass() {
  return std::make_unique<LLVMOMPOptPass>();
}
