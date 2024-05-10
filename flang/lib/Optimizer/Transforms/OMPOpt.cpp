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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
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

[[maybe_unused]] static inline mlir::Type
getLlvmPtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
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
  }
  return std::nullopt;
}

static mlir::LLVM::ConstantOp
genI32Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i32Ty = rewriter.getI32Type();
  mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
}

[[maybe_unused]] static mlir::LLVM::ConstantOp
genI64Constant(mlir::Location loc, mlir::RewriterBase &rewriter, int value) {
  mlir::Type i64Ty = rewriter.getI64Type();
  mlir::IntegerAttr attr = rewriter.getI64IntegerAttr(value);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, attr);
}

struct SplitTargetResult {
  omp::TargetOp targetOp;
  omp::DataOp dataOp;
};

/// If multiple coexecutes are nested in a target regions, we will need to split
/// the target region, but we want to preserve the data semantics of the
/// original data region and avoid unnecessary data movement at each of the
/// subkernels - we split the target region into a target_data{target}
/// nest where only the outer one moves the data
///
/// TODO need to handle special case where the target regions had a always copy
/// or always free map types (or something similar, I forgot how they are
/// called); I think these just need to be removed from the inner data region
/// map
std::optional<SplitTargetResult> splitTargetData(omp::TargetOp targetOp,
                                                 RewriterBase &rewriter) {

  // We should be doing these checks at the callsite

  auto loc = targetOp->getLoc();
  if (targetOp.getMapOperands().empty()) {
    LLVM_DEBUG(dbgs() << TAG << "target region has no data maps\n");
    return std::nullopt;
  }

  SmallVector<omp::MapInfoOp> mapInfos;
  for (auto opr : targetOp.getMapOperands())
    mapInfos.push_back(cast<omp::MapInfoOp>(opr.getDefiningOp()));

  LLVM_DEBUG(dbgs() << "Generating target data wrap\n";);

  // Generate maps that do not move any memory which will be used for the inner,
  // and the device pointers that we will use.
  // TODO Not sure - do we need one more level of indirection for the
  // use_device_ptr?
  rewriter.setInsertionPoint(targetOp);
  SmallVector<Value> innerMapInfos;
  SmallVector<Value> outerMapInfos;
  // SmallVector<Value> toMapInfos;
  // SmallVector<Value> fromMapInfos;
  SmallVector<Value> useDevicePtr;
  for (auto mapInfo : mapInfos) {
    // useDevicePtr.push_back(mapInfo.getVarPtr());
    assert(!mapInfo.getVarPtrPtr() && "TODO");
    // TODO are these ever not present?
    auto originalMapType =
        (llvm::omp::OpenMPOffloadMappingFlags)*mapInfo.getMapType();
    auto originalCaptureType = *mapInfo.getMapCaptureType();
    LLVM_DEBUG(dbgs() << mapInfo << " with map type "
                      << (uint64_t)originalMapType << " and capture type "
                      << originalCaptureType << "\n");

    llvm::omp::OpenMPOffloadMappingFlags newMapType;
    mlir::omp::VariableCaptureKind newCaptureType;
    // It looks like arrays get passed ByRef, and scalar variables ByCopy so we
    // remove the to/from mapping for ByRef maps to avoid the copies at each
    // coexecute subkernel.
    if (originalCaptureType == mlir::omp::VariableCaptureKind::ByCopy) {
      newMapType = originalMapType;
      newCaptureType = originalCaptureType;
    } else if (originalCaptureType == mlir::omp::VariableCaptureKind::ByRef) {
      newMapType = llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_NONE;
      newCaptureType = originalCaptureType;
      outerMapInfos.push_back(mapInfo);
    } else {
      llvm_unreachable("Unhandled case");
    }

    LLVM_DEBUG(dbgs() << "New: map type " << (uint64_t)newMapType
                      << " and capture type " << newCaptureType << "\n");

    auto innerMapInfo = cast<omp::MapInfoOp>(rewriter.clone(*mapInfo));
    innerMapInfo.setMapTypeAttr(rewriter.getIntegerAttr(
        rewriter.getIntegerType(64, false),
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            newMapType)));
    innerMapInfo.setMapCaptureType(newCaptureType);

    innerMapInfos.push_back(innerMapInfo.getResult());
  }

  rewriter.setInsertionPoint(targetOp);
  // auto dataEnterOp = rewriter.create<omp::EnterDataOp>(loc,
  // targetOp.getIfExpr(), targetOp.getDevice(), targetOp.getNowaitAttr(),
  // targetOp.getMapOperands());
  // TODO I still dont understand the use_device_addr thing...
  auto dataOp = rewriter.create<omp::DataOp>(
      loc, targetOp.getIfExpr(), targetOp.getDevice(), useDevicePtr,
      /*use_device_addr=*/mlir::ValueRange(), outerMapInfos);
  Block *dataOpBlock = rewriter.createBlock(&dataOp.getRegion(),
                                            dataOp.getRegion().begin(), {}, {});
  for (auto ptr : useDevicePtr)
    dataOpBlock->addArgument(ptr.getType(), ptr.getLoc());
  auto newTargetOp = rewriter.create<omp::TargetOp>(
      loc, targetOp.getIfExpr(), targetOp.getDevice(),
      targetOp.getThreadLimit(), targetOp.getTripCount(),
      targetOp.getNowaitAttr(), innerMapInfos, targetOp.getNumTeamsLower(),
      targetOp.getNumTeamsUpper(), targetOp.getTeamsThreadLimit(),
      targetOp.getNumThreads());
  rewriter.create<omp::TerminatorOp>(loc);
  // auto dataExitOp = rewriter.create<omp::ExitDataOp>(loc,
  // targetOp.getIfExpr(), targetOp.getDevice(), targetOp.getNowaitAttr(),
  // targetOp.getMapOperands());

  rewriter.inlineRegionBefore(targetOp.getRegion(), newTargetOp.getRegion(),
                              newTargetOp.getRegion().begin());

  rewriter.replaceOp(targetOp, newTargetOp);

  return SplitTargetResult{newTargetOp, nullptr};
}

/// Removes unused operands/args from the omp.target op
///
/// TODO is this ok? are there map types that need to be preserved even though
/// we do not use them in the target region?
[[maybe_unused]] static void minimizeArgs(omp::TargetOp targetOp) {
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

static Type getOmpDeviceType(MLIRContext *c) { return IntegerType::get(c, 32); }

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

void moveToHost(omp::TargetOp targetOp, RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  Block *targetBlock = &targetOp.getRegion().front();
  assert(targetBlock == &targetOp.getRegion().back());
  IRMapping mapping;
  for (auto map :
       zip_equal(targetOp.getMapOperands(), targetBlock->getArguments())) {
    Value mapInfo = std::get<0>(map);
    BlockArgument arg = std::get<1>(map);
    Operation *op = mapInfo.getDefiningOp();
    assert(op);
    auto mapInfoOp = cast<omp::MapInfoOp>(op);
    mapping.map(arg, mapInfoOp.getVarPtr());
  }

  rewriter.setInsertionPoint(targetOp);
  // rewriter.inlineBlockBefore(&targetOp.getRegion().front(), targetOp, args);

  for (auto it = targetBlock->begin(), end = std::prev(targetBlock->end());
       it != end; ++it) {
    auto allocOp = dyn_cast<fir::AllocMemOp>(&*it);
    auto freeOp = dyn_cast<fir::FreeMemOp>(&*it);
    fir::CallOp runtimeCall = nullptr;
    if (isRuntimeCall(&*it))
      runtimeCall = cast<fir::CallOp>(&*it);

    Value device;
    if (allocOp || freeOp || runtimeCall) {
      device = targetOp.getDevice();
      if (!device) {
        // TODO is this the correct way to get the default device?
        device = genI32Constant(it->getLoc(), rewriter, 0);
      }
    }
    if (allocOp) {
      auto tmpAllocOp = rewriter.create<fir::OmpTargetAllocMemOp>(
          allocOp.getLoc(), allocOp.getType(), device, allocOp.getInTypeAttr(),
          allocOp.getUniqNameAttr(), allocOp.getBindcNameAttr(),
          allocOp.getTypeparams(), allocOp.getShape());
      auto newAllocOp = cast<fir::OmpTargetAllocMemOp>(
          rewriter.clone(*tmpAllocOp.getOperation(), mapping));
      mapping.map(allocOp.getResult(), newAllocOp.getResult());
      rewriter.eraseOp(tmpAllocOp);
    } else if (freeOp) {
      auto tmpFreeOp = rewriter.create<fir::OmpTargetFreeMemOp>(
          freeOp.getLoc(), device, freeOp.getHeapref());
      rewriter.clone(*tmpFreeOp.getOperation(), mapping);
      rewriter.eraseOp(tmpFreeOp);
    } else if (runtimeCall) {
      auto module = runtimeCall->getParentOfType<ModuleOp>();
      auto callee =
          cast<func::FuncOp>(module.lookupSymbol(runtimeCall.getCalleeAttr()));
      std::string newCalleeName = (callee.getName() + "_omp").str();
      mlir::OpBuilder moduleBuilder(module.getBodyRegion());
      func::FuncOp newCallee =
          cast_or_null<func::FuncOp>(module.lookupSymbol(newCalleeName));
      if (!newCallee) {
        SmallVector<Type> argTypes(callee.getFunctionType().getInputs());
        argTypes.push_back(getOmpDeviceType(rewriter.getContext()));
        newCallee = moduleBuilder.create<func::FuncOp>(
            callee->getLoc(), newCalleeName,
            FunctionType::get(rewriter.getContext(), argTypes,
                              callee.getFunctionType().getResults()));
        if (callee.getArgAttrs())
          newCallee.setArgAttrsAttr(*callee.getArgAttrs());
        if (callee.getResAttrs())
          newCallee.setResAttrsAttr(*callee.getResAttrs());
        newCallee.setSymVisibility(callee.getSymVisibility());
        newCallee->setDiscardableAttrs(callee->getDiscardableAttrDictionary());
      }
      SmallVector<Value> operands = runtimeCall.getOperands();
      operands.push_back(device);
      auto tmpCall = rewriter.create<fir::CallOp>(
          runtimeCall.getLoc(), runtimeCall.getResultTypes(),
          SymbolRefAttr::get(newCallee), operands,
          runtimeCall.getFastmathAttr());
      Operation *newCall = rewriter.clone(*tmpCall, mapping);
      mapping.map(&*it, newCall);
      rewriter.eraseOp(tmpCall);
    } else {
      rewriter.clone(*it, mapping);
    }
  }

  rewriter.eraseOp(targetOp);
}

struct SplitResult {
  omp::TargetOp preTargetOp;
  omp::TargetOp isolatedTargetOp;
  omp::TargetOp postTargetOp;
};

/// Isolates the first target{parallel|teams{}} nest in its own omp.target op
///
/// TODO lifetime analysis to lower amount of memory required for temporaries
/// TODO flow mincut analysis to figure out the lowest amount of memory we need
/// to allocate for the crossing
/// TODO try to sink top level operations in the parallel regions in order to
/// avoid having to split them off in a separate omp.target
/// TODO when we generate loads for the temporaries these should be in the
/// parallel region
void fissionTarget(omp::TargetOp targetOp, RewriterBase &rewriter) {
  auto tuple = getNestedOpToIsolate(targetOp);
  if (!tuple) {
    LLVM_DEBUG(dbgs() << TAG << "No op to isolate\n");
    moveToHost(targetOp, rewriter);
    return;
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
    auto getMapInfo = [&](auto mappingFlags, const char *name) {
      return rewriter.create<omp::MapInfoOp>(
          loc, alloc.getType(), alloc, allocType, /*var_ptr_ptr=*/nullptr,
          /*members=*/ValueRange(), /*bounds=*/ValueRange(),
          rewriter.getIntegerAttr(
              rewriter.getIntegerType(64, false),
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  mappingFlags)),
          rewriter.getAttr<mlir::omp::VariableCaptureKindAttr>(
              mlir::omp::VariableCaptureKind::ByRef),
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

  auto isolateOp = [&](Operation *splitBefore) -> SplitResult {
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

    // Iterating these sets must be deterministic w.r.t. the order we insert
    // insert into them because we need to end up with the same argument order
    // in the host and all the device modules.
    // TODO In the future we would want a better pipeline where we can embed the
    // target code into modules in the host file - then we can properly deal
    // with issues that can arise from having (slightly) different code on the
    // device and host. Currently we depend on them having the same (or close
    // enough) representation at this level that the transformations we do in
    // this pass will result in the same transformation in all modules of
    // different targets of the same compilation.
    SetVector<Operation *> toCache;
    SetVector<Operation *> toRecompute;
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

    FileLineColLoc preLoc, isolatedLoc, postLoc;
    if (auto flc = loc.dyn_cast<FileLineColLoc>()) {
      int offset = 0;
      int increment = 1;
      preLoc = FileLineColLoc::get(flc.getFilename(), flc.getLine() + offset,
                                   flc.getColumn());
      offset += increment;
      isolatedLoc = FileLineColLoc::get(
          flc.getFilename(), flc.getLine() + offset, flc.getColumn());
      offset += increment;
      postLoc = FileLineColLoc::get(flc.getFilename(), flc.getLine() + offset,
                                    flc.getColumn());
    } else {
      llvm_unreachable("target op must have file line col loc");
    }

    rewriter.setInsertionPoint(targetOp);
    auto preTargetOp = rewriter.create<omp::TargetOp>(
        preLoc, targetOp.getIfExpr(), targetOp.getDevice(),
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

    auto reloadCache = [&](IRMapping &mapping, Block *newTargetBlock) {
      for (unsigned i = 0; i < targetBlock->getNumArguments(); i++) {
        auto originalArg = targetBlock->getArgument(i);
        auto newArg = newTargetBlock->addArgument(originalArg.getType(),
                                                  originalArg.getLoc());
        mapping.map(originalArg, newArg);
      }

      // See above
      assert(toClone.empty());
      // for (auto loadOp : toClone)
      //   rewriter.clone(*loadOp, postMapping);

      for (auto tup : allocs) {
        auto original = std::get<0>(tup);
        Value newArg = newTargetBlock->addArgument(
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
        mapping.map(original, restored);
      }
      for (auto it = targetBlock->begin(); it != splitBefore->getIterator();
           it++)
        if (toRecompute.contains(&*it))
          rewriter.clone(*it, mapping);
    };

    rewriter.setInsertionPoint(targetOp);
    auto isolatedTargetOp = rewriter.create<omp::TargetOp>(
        isolatedLoc, targetOp.getIfExpr(), targetOp.getDevice(),
        targetOp.getThreadLimit(), targetOp.getTripCount(),
        targetOp.getNowait(), postMapOperands, targetOp.getNumTeamsLower(),
        targetOp.getNumTeamsUpper(), targetOp.getTeamsThreadLimit(),
        targetOp.getNumThreads());
    auto *isolatedTargetBlock =
        rewriter.createBlock(&isolatedTargetOp.getRegion(),
                             isolatedTargetOp.getRegion().begin(), {}, {});
    IRMapping isolatedMapping;
    reloadCache(isolatedMapping, isolatedTargetBlock);

    rewriter.clone(*splitBefore, isolatedMapping);
    rewriter.create<omp::TerminatorOp>(loc);

    omp::TargetOp postTargetOp = nullptr;
    if (splitAfter) {
      rewriter.setInsertionPoint(targetOp);
      postTargetOp = rewriter.create<omp::TargetOp>(
          postLoc, targetOp.getIfExpr(), targetOp.getDevice(),
          targetOp.getThreadLimit(), targetOp.getTripCount(),
          targetOp.getNowait(), postMapOperands, targetOp.getNumTeamsLower(),
          targetOp.getNumTeamsUpper(), targetOp.getTeamsThreadLimit(),
          targetOp.getNumThreads());
      auto *postTargetBlock = rewriter.createBlock(
          &postTargetOp.getRegion(), postTargetOp.getRegion().begin(), {}, {});
      IRMapping postMapping;
      reloadCache(postMapping, postTargetBlock);

      assert(splitBefore->getNumResults() == 0 ||
             llvm::all_of(splitBefore->getResults(),
                          [](Value result) { return result.use_empty(); }));

      for (auto it = std::next(splitBefore->getIterator());
           it != targetBlock->end(); it++)
        rewriter.clone(*it, postMapping);
    }

    rewriter.eraseOp(targetOp);

    // minimizeArgs(preTargetOp);
    // minimizeArgs(postTargetOp);

    return {preTargetOp, isolatedTargetOp, postTargetOp};
  };

  if (splitBefore && splitAfter) {
    auto res = isolateOp(toIsolate);
    moveToHost(res.preTargetOp, rewriter);
    fissionTarget(res.postTargetOp, rewriter);
    return;
  }
  if (splitBefore) {
    auto res = isolateOp(toIsolate);
    moveToHost(res.preTargetOp, rewriter);
    return;
  }
  if (splitAfter) {
    assert(false && "TODO");
    auto res = isolateOp(toIsolate->getNextNode());
    fissionTarget(res.postTargetOp, rewriter);
    return;
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

static bool canCollapse(fir::DoLoopOp nested, fir::DoLoopOp outermost) {
  return llvm::all_of(nested->getOperands(), [&](Value v) {
    if (auto *op = v.getDefiningOp())
      return op->getParentOp()->isProperAncestor(outermost);
    return true;
  });
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

    SmallVector<fir::DoLoopOp> nestToParallelize;
    auto outermostLoopOp = getPerfectlyNested<fir::DoLoopOp>(coexecuteOp);
    if (outermostLoopOp && shouldParallelize(outermostLoopOp)) {
      nestToParallelize.push_back(outermostLoopOp);
      while (auto nestedLoopOp =
                 getPerfectlyNested<fir::DoLoopOp>(nestToParallelize.back())) {
        if (canCollapse(nestedLoopOp, outermostLoopOp))
          nestToParallelize.push_back(nestedLoopOp);
        else
          break;
      }
      llvm::for_each(nestToParallelize, [&](fir::DoLoopOp loopOp) {
        auto *loopTerminator = loopOp.getBody()->getTerminator();
        assert(loopTerminator->getNumResults() == 0);
      });
      auto innermostLoopOp = nestToParallelize.back();

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
      auto lbs = llvm::map_to_vector(nestToParallelize,
                                     [&](fir::DoLoopOp doLoop) -> Value {
                                       return doLoop.getLowerBound();
                                     });
      auto ubs = llvm::map_to_vector(nestToParallelize,
                                     [&](fir::DoLoopOp doLoop) -> Value {
                                       return doLoop.getUpperBound();
                                     });
      auto steps = llvm::map_to_vector(
          nestToParallelize,
          [&](fir::DoLoopOp doLoop) -> Value { return doLoop.getStep(); });
      auto wsLoop =
          rewriter.create<omp::WsLoopOp>(coexecuteLoc, lbs, ubs, steps);
      rewriter.create<omp::TerminatorOp>(coexecuteLoc);
      wsLoop.setInclusive(true);
      rewriter.createBlock(&wsLoop.getRegion(), wsLoop.getRegion().begin(), {},
                           {});

      IRMapping mapping;
      unsigned numCollapse = nestToParallelize.size();
      for (unsigned j = 0; j < numCollapse; j++) {
        Value originalIV = nestToParallelize[j].getInductionVar();
        mapping.map(originalIV, wsLoop.getRegion().front().addArgument(
                                    originalIV.getType(), originalIV.getLoc()));
      }
      for (auto op = innermostLoopOp.getBody()->begin();
           op != innermostLoopOp.getBody()->end(); op++)
        rewriter.clone(*op, mapping);
      rewriter.replaceOpWithNewOp<omp::YieldOp>(
          rewriter.getInsertionBlock()->getTerminator(), ValueRange());

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

    // TODO we should assign the location of the coexecute op to be the location
    // of the statement it is for so that we get a meaningful message here. Also
    // the message is wrong.
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

  // TODO We should grab all TargetOps that we need to handle and run our
  // patterns and transformations on them and not recollect anything between
  // transformations

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
  {
    SmallVector<omp::TargetOp> targetOps;
    op->walk([&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
    IRRewriter rewriter(&context);
    for (auto targetOp : targetOps) {

      auto loc = targetOp.getLoc();

      if (auto flc = loc.dyn_cast<FileLineColLoc>()) {
        int offset = 1'000'000'000;
        int multiplier = 100;
        auto newLoc = FileLineColLoc::get(flc.getFilename(),
                                          multiplier * flc.getLine() + offset,
                                          flc.getColumn() + offset);
        targetOp->setLoc(newLoc);
      } else {
        llvm_unreachable("target op must have file line col loc");
      }

      auto res = splitTargetData(targetOp, rewriter);
      if (res)
        fissionTarget(res->targetOp, rewriter);
    }
  }
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
