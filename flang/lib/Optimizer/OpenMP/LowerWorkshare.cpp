//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lower omp workshare construct.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/ADT/STLExtras.h"
#include <llvm/ADT/iterator_range.h>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/IR/IRMapping.h"

#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKSHARE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workshare"

using namespace mlir;

namespace {

struct SingleRegion {
  Block::iterator begin, end;
};

static bool isPtr(Type ty) {
  return isa<fir::ReferenceType>(ty) || isa<LLVM::LLVMPointerType>(ty);
}

static bool isSafeToParallelize(Operation *op) {
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

void lowerWorkshare(mlir::omp::WorkshareOp wsOp) {
  assert(wsOp.getRegion().getBlocks().size() == 1);

  Location loc = wsOp->getLoc();

  auto parallelOp = mlir::cast<mlir::omp::ParallelOp>(wsOp->getParentOp());
  OpBuilder allocBuilder(parallelOp);
  OpBuilder rootBuilder(wsOp);
  IRMapping rootMapping;

  omp::SingleOp singleOp = nullptr;

  auto mapReloadedValue = [&](Value v) {
    if (auto reloaded = rootMapping.lookupOrNull(v))
      return;
    Type llvmPtrTy = LLVM::LLVMPointerType::get(allocBuilder.getContext());
    Type ty = v.getType();
    Value alloc;
    if (isPtr(ty)) {
      auto one =
          allocBuilder.create<LLVM::ConstantOp>(loc, allocBuilder.getI32Type(), 1);
      alloc = allocBuilder.create<LLVM::AllocaOp>(loc, llvmPtrTy, llvmPtrTy, one);
    } else {
      alloc = allocBuilder.create<fir::AllocaOp>(loc, ty);
    }
    auto reloaded = rootBuilder.create<fir::LoadOp>(loc, ty, alloc);
    rootMapping.map(v, reloaded);
  };

  omp::SingleOperands singleOperands;

  auto moveToSingle = [&](SingleRegion sr, OpBuilder singleBuilder) {
    IRMapping singleMapping;

    // Prepare reloaded values for results of operations that cannot be safely
    // parallelized and which are used after the region `sr`
    for (Operation &op : llvm::make_range(sr.begin, sr.end)) {
      if (isSafeToParallelize(&op))
        continue;
      for (auto res : op.getResults()) {
        for (auto &use : res.getUses()) {
          Operation *user = use.getOwner();
          while (user->getParentOp() != wsOp)
            user = user->getParentOp();
          if (!user->isBeforeInBlock(&*sr.end)) {
            // We need to reload
            mapReloadedValue(use.get());
          }
        }
      }
    }

    for (Operation &op : llvm::make_range(sr.begin, sr.end)) {
      if (isSafeToParallelize(&op))
        rootBuilder.clone(op, rootMapping);
      singleBuilder.clone(op, singleMapping);
    }

  };

  Block *wsBlock = &wsOp.getRegion().front();
  assert(wsBlock->getTerminator()->getNumOperands() == 0);
  Operation *terminator = wsBlock->getTerminator();

  SmallVector<std::variant<SingleRegion, omp::ParallelOp>> regions;

  auto it = wsBlock->begin();
  auto getSingleRegion = [&]() {
    if (&*it == terminator)
      return false;
    if (auto pop = dyn_cast<omp::ParallelOp>(&*it)) {
      regions.push_back(pop);
      it++;
      return true;
    }
    SingleRegion sr;
    sr.begin = it;
    while (&*it != terminator && !isa<omp::ParallelOp>(&*it))
      it++;
    sr.end = it;
    assert(sr.begin != sr.end);
    regions.push_back(sr);
    return true;
  };
  while(getSingleRegion());

  for (auto loopOrSingle : regions) {
    if (std::holds_alternative<SingleRegion>(loopOrSingle)) {
      singleOp = rootBuilder.create<omp::SingleOp>(loc, singleOperands);
      OpBuilder singleBuilder(singleOp);
      singleBuilder.createBlock(&singleOp.getRegion());
      moveToSingle(std::get<SingleRegion>(loopOrSingle), singleBuilder);
    } else {
      Region &popRegion = std::get<omp::ParallelOp>(loopOrSingle).getRegion();
      assert(popRegion.hasOneBlock());
      Block *popBlock = &popRegion.front();
      assert(popBlock->getTerminator()->getNumOperands() == 0);
      for (auto &op : popBlock->without_terminator())
        rootBuilder.clone(op, rootMapping);
    }
  }

  if (!wsOp.getNowait())
    rootBuilder.create<omp::BarrierOp>(loc);

  wsOp->erase();

  return;
}

class LowerWorksharePass
    : public flangomp::impl::LowerWorkshareBase<LowerWorksharePass> {
public:
  void runOnOperation() override {
    getOperation()->walk([&](mlir::omp::WorkshareOp wsOp) {
      lowerWorkshare(wsOp);
    });
  }
};
} // namespace
