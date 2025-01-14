#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "polly/Support/GICHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"

#include "LoopUndistribute.h"

using namespace mlir;

namespace {

bool areEquiv(affine::AffineParallelOp a, affine::AffineParallelOp b) {
  return a.getLowerBoundsValueMap() == b.getLowerBoundsValueMap() &&
         a.getLowerBoundsValueMap() == b.getLowerBoundsValueMap() &&
         a.getSteps() == b.getSteps();
}

LogicalResult parallelizePreceeding(affine::AffineParallelOp parOp,
                                    RewriterBase &rewriter) {
  assert(gpu::affine_opt::isAffineBlockPar(parOp) && "op is not block par");
  Block *block = parOp->getBlock();

  bool changed = false;
  IRMapping mapping;
  rewriter.setInsertionPointToStart(parOp.getBody());
  for (auto &op : llvm::make_range(block->begin(), parOp->getIterator())) {
    if (isMemoryEffectFree(&op) && !gpu::affine_opt::isBlockPar(&op)) {
      Operation *cloned = rewriter.clone(op, mapping);
      rewriter.replaceOpUsesWithinBlock(&op, cloned->getResults(),
                                        parOp.getBody());
    }
  }
  for (Operation *op = parOp->getPrevNode(); op != nullptr;) {
    Operation *prev = op->getPrevNode();
    if (isOpTriviallyDead(op)) {
      op->erase();
      changed |= true;
    }
    op = prev;
  }

  return success(changed);
}

bool regionNeedsBarrier(Region &region) {
  Operation *op = region.getParentOp();
  if (isa<affine::AffineForOp, scf::ForOp>(op)) {
    return true;
  } else if (isa<affine::AffineIfOp, scf::IfOp>(op)) {
    return false;
  }
  op->dump();
  llvm_unreachable("unknown op");
}

// interchange is only supported for operations with no results for now
LogicalResult interchange(Operation *parent, RewriterBase &rewriter) {
  if (gpu::affine_opt::isGridPar(parent))
    return rewriter.notifyMatchFailure(parent, "Parent op is grid par");

  SmallVector<affine::AffineParallelOp> pars;
  affine::AffineParallelOp par = nullptr;
  if (!llvm::all_of(parent->getRegions(), [&](Region &region) {
        if (region.getBlocks().size() == 0) {
          pars.push_back(nullptr);
          return true;
        }
        if (region.getBlocks().size() == 1) {
          assert(region.front().getTerminator()->getNumResults() == 0);
          assert(region.front().getTerminator()->getNumOperands() == 0);
          if (region.front().getOperations().size() == 1) {
            pars.push_back(nullptr);
            return true;
          } else if (region.front().getOperations().size() == 2) {
            auto thisPar =
                gpu::affine_opt::isAffineBlockPar(&region.front().front());
            if (thisPar) {
              pars.push_back(par);
              par = thisPar;
              return true;
            } else {
              return false;
            }
          } else {
            return false;
          }
        } else {
          return false;
        }
      }))
    return rewriter.notifyMatchFailure(
        parent, "Parent op not suitable for interchange");

  assert(par);
  assert(pars.size() > 0);
  assert(llvm::all_of(pars, [&](auto thisPar) {
    return thisPar == nullptr || areEquiv(par, thisPar);
  }));

  auto loc = parent->getLoc();

  // affine.for
  //   affine.parallel
  //
  // to
  //
  // affine.parallel
  //   affine.for

  rewriter.setInsertionPoint(parent);
  auto newPar =
      cast<affine::AffineParallelOp>(rewriter.cloneWithoutRegions(*par));
  rewriter.createBlock(&newPar.getRegion(), newPar.getRegion().begin(),
                       par.getBody()->getArgumentTypes(),
                       par.getBody()->getArgumentLocs());
  Operation *child = rewriter.cloneWithoutRegions(*parent);
  rewriter.create<affine::AffineYieldOp>(loc);

  for (auto [childRegion, parentRegion] :
       llvm::zip(child->getRegions(), parent->getRegions())) {
    if (parentRegion.getBlocks().size() == 0)
      continue;

    bool needsBarrier = regionNeedsBarrier(childRegion);

    Block *parentBody = &parentRegion.front();
    rewriter.createBlock(&childRegion, childRegion.begin(),
                         parentBody->getArgumentTypes(),
                         parentBody->getArgumentLocs());
    Block *childBody = &childRegion.front();
    rewriter.inlineBlockBefore(&parentBody->front().getRegion(0).front(),
                               childBody, childBody->begin(),
                               newPar.getBody()->getArguments());
    assert(childBody->getTerminator()->getNumResults() == 0);
    rewriter.eraseOp(childBody->getTerminator());
    rewriter.setInsertionPointToEnd(childBody);

    if (needsBarrier)
      rewriter.create<affine::AffineBarrierOp>(
          newPar.getLoc(), newPar.getBody()->getArguments());

    assert(parentBody->getTerminator()->getNumResults() == 0);
    assert(parentBody->getTerminator()->getNumOperands() == 0);
    rewriter.clone(*parentBody->getTerminator());

    for (auto [oldArg, newArg] :
         llvm::zip(parentBody->getArguments(), childBody->getArguments()))
      rewriter.replaceAllUsesWith(oldArg, newArg);
  }

  rewriter.eraseOp(parent);

  return success();
}

LogicalResult fuse(affine::AffineParallelOp par, RewriterBase &rewriter) {
  if (!gpu::affine_opt::isAffineBlockPar(par))
    return rewriter.notifyMatchFailure(par, "op is not block par");
  auto nextPar = gpu::affine_opt::isAffineBlockPar(par->getNextNode());
  if (!nextPar)
    return rewriter.notifyMatchFailure(par->getNextNode(),
                                       "Next op is not block par");
  assert(areEquiv(par, nextPar));
  if (!areEquiv(par, nextPar))
    return rewriter.notifyMatchFailure(nextPar, "Non equiv pars");

  assert(nextPar.getBody()->getTerminator()->getNumResults() == 0);
  rewriter.eraseOp(nextPar.getBody()->getTerminator());
  rewriter.setInsertionPoint(par.getBody()->getTerminator());
  rewriter.create<affine::AffineBarrierOp>(nextPar.getLoc(),
                                           par.getBody()->getArguments());
  rewriter.inlineBlockBefore(nextPar.getBody(), par.getBody()->getTerminator(),
                             par.getBody()->getArguments());
  rewriter.eraseOp(nextPar);

  return success();
}

SmallVector<affine::AffineParallelOp> findBlockPars(Operation *parent) {
  SmallVector<affine::AffineParallelOp> blockPars;
  parent->walk([&](Operation *op) {
    if (auto par = gpu::affine_opt::isAffineBlockPar(op)) {
      blockPars.push_back(par);
      return WalkResult::skip();
    }
    assert(!gpu::affine_opt::isBlockPar(op));
    return WalkResult::advance();
  });
  return blockPars;
}

LogicalResult didSucceed(Operation *girdPar) {
  // TODO check for the GPU structure
  return success();
}

} // namespace

LogicalResult mlir::undistributeLoops(Operation *func) {
  Operation *gridPar = nullptr;
  func->walk([&](Operation *op) {
    if (gpu::affine_opt::isGridPar(op)) {
      assert(!gridPar);
      gridPar = op;
    }
  });
  assert(gridPar && gridPar->getNumRegions() == 1 &&
         gridPar->getRegion(0).getBlocks().size() == 1);

  while (true) {
    auto blockPars = findBlockPars(gridPar);
    assert(blockPars.size() > 0);

    IRRewriter rewriter(func->getContext());
    bool changed = false;

    for (auto blockPar : blockPars) {
      changed |= interchange(blockPar->getParentOp(), rewriter).succeeded();
      if (changed)
        break;
    }
    if (changed)
      continue;

    for (auto blockPar : blockPars) {
      changed |= fuse(blockPar, rewriter).succeeded();
      if (changed)
        break;
    }
    if (changed)
      continue;

    for (auto blockPar : blockPars) {
      changed |= parallelizePreceeding(blockPar, rewriter).succeeded();
    }

    llvm::SmallSetVector<Block *, 8> blocks;
    for (auto par : blockPars)
      blocks.insert(par->getBlock());
    if (changed)
      continue;

    break;
  }

  return didSucceed(gridPar);
}
