
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"

namespace mlir {
namespace gpu {
namespace affine_opt {

struct Copy {
  affine::AffineReadOpInterface load;
  affine::AffineWriteOpInterface store;
};

struct AccessInfo {
  SmallVector<Value, 8> regionSymbols;
  SmallVector<Value, 4> lbs, ubs;
  SmallVector<AffineMap, 4> lbMaps, ubMaps;
};

void optGlobalSharedMemCopies(Operation *root) {
  SmallVector<memref::AllocaOp> shmemAllocas;
  llvm::SmallSetVector<MemrefValue, 4> shmemMemrefs;
  root->walk([&](memref::AllocaOp alloca) {
    if (nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
            alloca.getType().getMemorySpace())) {
      shmemAllocas.push_back(alloca);
      shmemMemrefs.insert(cast<MemrefValue>(alloca.getMemref()));
    }
  });

  auto isGlobalMemref = [&](MemrefValue m) {
    return !nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
               m.getType().getMemorySpace()) &&
           m.getParentRegion()->getParentOp() == root;
  };

  affine::AffineParallelOp outermostBlockPar = nullptr;
  root->walk([&](affine::AffineParallelOp par) {
    if (par->hasAttr("gpu.par.block.z") || par->hasAttr("gpu.par.block.y") ||
        par->hasAttr("gpu.par.block.x")) {
      if (outermostBlockPar) {
        if (par->isProperAncestor(outermostBlockPar)) {
          outermostBlockPar = par;
        }
      } else {
        outermostBlockPar = par;
      }
    }
  });

  SmallVector<Value> gridIVs;
  affine::getAffineIVs(*outermostBlockPar, gridIVs);
  unsigned numGridLoops = gridIVs.size();

  for (auto alloca : shmemAllocas) {
    LLVM_DEBUG(llvm::dbgs() << "Handling " << *alloca << ":\n");
    SmallVector<Copy> copies;
    bool allAffineAccesses = true;
    for (auto opInst : alloca.getMemref().getUsers()) {
      if (auto load = dyn_cast<affine::AffineReadOpInterface>(opInst)) {
        // These can't be optimized, copy_async only supports global->shared
        continue;
        if (!load.getValue().hasOneUse())
          continue;
        auto store = dyn_cast_or_null<affine::AffineWriteOpInterface>(
            *load.getValue().getUsers().begin());
        if (!store)
          continue;
        if (isGlobalMemref(cast<MemrefValue>(store.getMemRef())))
          copies.push_back({load, store});
      } else if (auto store =
                     dyn_cast<affine::AffineWriteOpInterface>(opInst)) {
        if (!store.getValueToStore().hasOneUse())
          continue;
        auto load = dyn_cast_or_null<affine::AffineReadOpInterface>(
            store.getValueToStore().getDefiningOp());
        if (!load)
          continue;
        if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
          copies.push_back({load, store});
      } else {
        allAffineAccesses = false;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "all affine: " << allAffineAccesses << "\n");

    IRRewriter rewriter(alloca->getNextNode());

    auto handleAccess = [&](Operation *access) -> std::optional<AccessInfo> {
      affine::MemRefRegion region(access->getLoc());
      SmallVector<Value> accessIVs;
      affine::getAffineIVs(*access, accessIVs);
      if (failed(region.compute(access, accessIVs.size() - numGridLoops))) {
        LLVM_DEBUG(llvm::dbgs() << "Could not compute memref region for "
                                << *access << "\n");
        return {};
      }
      LLVM_DEBUG(llvm::dbgs() << "Got memref region for " << *access
                              << " with rank " << region.getRank() << "\n");
      LLVM_DEBUG(region.getConstraints()->dump());
      unsigned rank = region.getRank();
      const affine::FlatAffineValueConstraints *cst = region.getConstraints();
      AccessInfo info;
      info.lbMaps.append(rank, {});
      info.ubMaps.append(rank, {});
      cst->getValues(rank, cst->getNumVars(), &info.regionSymbols);
      for (unsigned i = 0; i < rank; ++i) {
        region.getLowerAndUpperBound(i, info.lbMaps[i], info.ubMaps[i]);
        LLVM_DEBUG(llvm::dbgs() << "lb " << info.lbMaps[i] << "\n");
        LLVM_DEBUG(llvm::dbgs() << "ub " << info.ubMaps[i] << "\n");
        auto lb = rewriter.create<affine::AffineMaxOp>(
            access->getLoc(), info.lbMaps[i], info.regionSymbols);
        auto ub = rewriter.create<affine::AffineMinOp>(
            access->getLoc(), info.lbMaps[i], info.regionSymbols);
        info.lbs.push_back(lb);
        info.ubs.push_back(ub);
      }
      return info;
    };

    for (auto copy : copies) {
      LLVM_DEBUG(llvm::dbgs() << "Found copy\n");
      auto loadBounds = handleAccess(copy.load);
      if (!loadBounds)
        continue;
      auto storeBounds = handleAccess(copy.store);
      if (!storeBounds)
        continue;

      // TODO assert ranges of the load = ranges of the store

      // TODO we support only memrefs of i8's for now, check that.

      // TODO need to align stuff to 4, 8, or 16-byte chunks

      // TODO currently we also have copies in nested constrol flow... need to
      // analyse those

      if (loadBounds->lbMaps.size() != 1)
        continue;
      if (storeBounds->lbMaps.size() != 1)
        continue;

      auto tempFor = rewriter.create<affine::AffineForOp>(
          copy.load->getLoc(), storeBounds->regionSymbols,
          storeBounds->lbMaps[0], storeBounds->regionSymbols,
          storeBounds->ubMaps[0], 1, ValueRange(), nullptr);
      auto largestDivisor = affine::getLargestDivisorOfTripCount(tempFor);

      int64_t copySize = 0;
      if (largestDivisor % 4 == 0)
        copySize = 4;
      if (largestDivisor % 8 == 0)
        copySize = 8;
      if (largestDivisor % 16 == 0)
        copySize = 16;

      rewriter.eraseOp(tempFor);

      if (!copySize) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Region span was not divisible by 4, 8, or 16\n");
        continue;
      }

      rewriter.create<affine::AffineForOp>(
          copy.load->getLoc(), storeBounds->regionSymbols,
          storeBounds->lbMaps[0], storeBounds->regionSymbols,
          storeBounds->ubMaps[0], copySize, ValueRange(),
          [&](OpBuilder &b, Location loc, Value iv, ValueRange ivs) {
            Value storeIdx = iv;
            // loadIdx = storeIdx - storeLB + loadLB
            Value loadIdx =
                b.create<arith::SubIOp>(loc, storeIdx, storeBounds->lbs[0]);
            loadIdx = b.create<arith::AddIOp>(loc, loadIdx, loadBounds->lbs[0]);
            rewriter.create<nvgpu::DeviceAsyncCopyOp>(
                copy.load->getLoc(), copy.store.getMemRef(), storeIdx,
                copy.load.getMemRef(), loadIdx, rewriter.getIndexAttr(copySize),
                nullptr, nullptr);
            rewriter.create<affine::AffineYieldOp>(loc);
          });
    }
  }

  root->dump();
}

} // namespace affine_opt
} // namespace gpu
} // namespace mlir
