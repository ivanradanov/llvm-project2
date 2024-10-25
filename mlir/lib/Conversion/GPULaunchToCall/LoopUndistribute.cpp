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

#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/constraint.h"
#include "isl/id.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

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
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(
           llvm::make_range(block->begin(), parOp->getIterator())))) {
    if (isOpTriviallyDead(&op)) {
      op.erase();
      changed |= true;
    }
  }

  return success(changed);
}

LogicalResult interchange(affine::AffineParallelOp parOp,
                          RewriterBase &rewriter) {
  if (!gpu::affine_opt::isAffineBlockPar(parOp))
    return rewriter.notifyMatchFailure(parOp, "op is not block par");
  Operation *parent = parOp->getParentOp();
  if (gpu::affine_opt::isGridPar(parent))
    return rewriter.notifyMatchFailure(parent, "Parent op is grid par");
  if (parOp->getBlock()->getOperations().size() != 2)
    return rewriter.notifyMatchFailure(parOp, "imperfectly nested");
  auto forOp = dyn_cast<affine::AffineForOp>(parent);
  if (!forOp)
    return rewriter.notifyMatchFailure(parent,
                                       "Parent op is not affine for op");

  auto loc = parOp->getLoc();

  // affine.for
  //   affine.parallel
  //
  // to
  //
  // affine.parallel
  //   affine.for
  rewriter.setInsertionPoint(forOp);
  auto newPar =
      cast<affine::AffineParallelOp>(rewriter.cloneWithoutRegions(*parOp));
  rewriter.createBlock(&newPar.getRegion(), newPar.getRegion().begin(),
                       parOp.getBody()->getArgumentTypes(),
                       parOp.getBody()->getArgumentLocs());
  auto newFor = cast<affine::AffineForOp>(rewriter.cloneWithoutRegions(*forOp));
  rewriter.createBlock(&newFor.getRegion(), newFor.getRegion().begin(),
                       forOp.getBody()->getArgumentTypes(),
                       forOp.getBody()->getArgumentLocs());
  rewriter.inlineBlockBefore(parOp.getBody(), newFor.getBody(),
                             newFor.getBody()->begin(),
                             newPar.getBody()->getArguments());
  assert(newFor.getBody()->getTerminator()->getNumResults() == 0);
  rewriter.eraseOp(newFor.getBody()->getTerminator());
  rewriter.setInsertionPointToEnd(newFor.getBody());
  rewriter.create<affine::AffineYieldOp>(loc);
  rewriter.setInsertionPointToEnd(newPar.getBody());
  rewriter.create<affine::AffineYieldOp>(loc);

  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newFor.getBody()->getArguments()))
    rewriter.replaceAllUsesWith(oldArg, newArg);

  rewriter.eraseOp(forOp);

  return success();
}
struct Interchange : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp parOp,
                                PatternRewriter &rewriter) const override {
    return interchange(parOp, rewriter);
  }
};

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

struct FuseBlockPars : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp par,
                                PatternRewriter &rewriter) const override {
    return fuse(par, rewriter);
  }
};

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
      changed |= interchange(blockPar, rewriter).succeeded();
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

// void mlir::populateLoopUndistributePatterns(RewritePatternSet &patterns) {
//   patterns.insert<FuseBlockPars, Interchange, ParallelizeSequential>(
//       patterns.getContext());
// }
