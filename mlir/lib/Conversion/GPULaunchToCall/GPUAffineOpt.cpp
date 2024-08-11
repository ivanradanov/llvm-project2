#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"

namespace mlir {
#define GEN_PASS_DEF_GPUAFFINEOPTPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

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
  unsigned rank;
};

static Value isVectorStore(affine::AffineWriteOpInterface store) {
  if (!store->getParentOp()->hasAttr("affine.vector.store"))
    return nullptr;
  return cast<vector::ExtractOp>(store.getValueToStore().getDefiningOp())
      .getVector();
}

static affine::AffineLoadOp isVectorLoad(affine::AffineParallelOp par) {
  if (!par->hasAttr("affine.vector.load"))
    return nullptr;
  return cast<affine::AffineLoadOp>(
      cast<vector::BroadcastOp>(
          par.getBody()->getTerminator()->getOperand(0).getDefiningOp())
          .getSource()
          .getDefiningOp());
}

static Value isVectorLoad(affine::AffineReadOpInterface load) {
  if (!load->getParentOp()->hasAttr("affine.vector.load"))
    return nullptr;
  return load->getParentOp()->getResult(0);
}

template <typename T>
static Value computeMap(RewriterBase &rewriter, Location loc, AffineMap map,
                        ValueRange operands) {
  if (map.getNumResults() == 1)
    return rewriter.create<T>(loc, map, operands);
  else if (map.getNumResults() > 1)
    return rewriter.create<affine::AffineApplyOp>(loc, map, operands);
  else
    llvm_unreachable("map with 0 results");
}

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
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(opInst)) {
        if (Value storedVal = isVectorStore(store)) {
          if (!storedVal.hasOneUse())
            continue;
          auto par = dyn_cast_or_null<affine::AffineParallelOp>(
              storedVal.getDefiningOp());
          if (!par)
            continue;
          auto load = isVectorLoad(par);
          if (!load)
            continue;
          if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
            copies.push_back({load, store});
        } else {
          if (!store.getValueToStore().hasOneUse())
            continue;
          auto load = dyn_cast_or_null<affine::AffineReadOpInterface>(
              store.getValueToStore().getDefiningOp());
          if (!load)
            continue;
          if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
            copies.push_back({load, store});
        }
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
      AccessInfo info;
      unsigned loopDepth = numGridLoops;
      if (failed(region.compute(access, loopDepth,
                                /*sliceState=*/nullptr,
                                /*addMemRefDimBounds=*/false))) {
        LLVM_DEBUG(llvm::dbgs() << "Could not compute memref region for "
                                << *access << "\n");
        return {};
      }
      LLVM_DEBUG(llvm::dbgs() << "Got memref region for " << *access
                              << " with rank " << region.getRank() << "\n");
      LLVM_DEBUG(region.getConstraints()->dump());
      unsigned rank = region.getRank();
      const affine::FlatAffineValueConstraints *cst = region.getConstraints();
      info.lbMaps.append(rank, {});
      info.ubMaps.append(rank, {});
      cst->getValues(rank, cst->getNumVars(), &info.regionSymbols);
      for (unsigned i = 0; i < rank; ++i) {
        region.getLowerAndUpperBound(i, info.lbMaps[i], info.ubMaps[i]);
        LLVM_DEBUG(llvm::dbgs() << "lb " << info.lbMaps[i] << "\n");
        LLVM_DEBUG(llvm::dbgs() << "ub " << info.ubMaps[i] << "\n");
        auto lb = computeMap<affine::AffineMaxOp>(
            rewriter, access->getLoc(), info.lbMaps[i], info.regionSymbols);
        auto ub = computeMap<affine::AffineMinOp>(
            rewriter, access->getLoc(), info.ubMaps[i], info.regionSymbols);
        info.lbs.push_back(lb);
        info.ubs.push_back(ub);
      }
      info.rank = rank;
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

      // TODO currently we may also have copies in nested (non-affine) constrol
      // flow... need to analyse those

      int64_t copySize = 16;
      for (const AccessInfo &info : {*loadBounds, *storeBounds}) {
        unsigned lastDim = info.rank - 1;
        // TODO is step=1 correct here?
        auto tempFor = rewriter.create<affine::AffineForOp>(
            copy.load->getLoc(), info.regionSymbols, info.lbMaps[lastDim],
            info.regionSymbols, info.ubMaps[lastDim], 1, ValueRange(), nullptr);

        auto largestDivisor = affine::getLargestDivisorOfTripCount(tempFor);

        int64_t thisCopySize = 0;
        if (largestDivisor % 4 == 0)
          thisCopySize = 4;
        if (largestDivisor % 8 == 0)
          thisCopySize = 8;
        if (largestDivisor % 16 == 0)
          thisCopySize = 16;

        copySize = thisCopySize < copySize ? thisCopySize : copySize;

        rewriter.eraseOp(tempFor);

        if (!thisCopySize) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Region span was not divisible by 4, 8, or 16\n");
        }
      }
      if (copySize == 0)
        continue;

      if (loadBounds->rank != storeBounds->rank) {
        LLVM_DEBUG(llvm::dbgs() << "store and load rank not equal\n");
        continue;
      }

      // TODO need to check if the iteration num over each dim is the same

      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> ivs;
      for (unsigned dim = 0; dim < storeBounds->rank; dim++) {
        int64_t step = dim == storeBounds->rank - 1 ? copySize : 1;
        auto forOp = rewriter.create<affine::AffineForOp>(
            copy.load->getLoc(), storeBounds->regionSymbols,
            storeBounds->lbMaps[dim], storeBounds->regionSymbols,
            storeBounds->ubMaps[dim], step, ValueRange());
        rewriter.setInsertionPointToStart(forOp.getBody());
        ivs.push_back(forOp.getInductionVar());
      }
      Location loc = copy.load.getLoc();
      // loadIdx = storeIdx - storeLB + loadLB
      // Value loadIdx =
      //     rewriter.create<arith::SubIOp>(loc, storeIdx, storeBounds->lbs[0]);
      // loadIdx =
      //     rewriter.create<arith::AddIOp>(loc, loadIdx, loadBounds->lbs[0]);
      rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          copy.load->getLoc(), copy.store.getMemRef(), ivs,
          copy.load.getMemRef(), /* TODO TEMP THIS IS WRONG */ ivs,
          rewriter.getIndexAttr(copySize), nullptr, nullptr);
    }
  }

  root->dump();
}

} // namespace affine_opt
} // namespace gpu
} // namespace mlir

struct GPUAffineOptPass : public impl::GPUAffineOptPassBase<GPUAffineOptPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](mlir::gpu::GPUModuleOp gpuModule) {
      const mlir::DataLayoutAnalysis dl(gpuModule);
      gpuModule->walk([&](mlir::LLVM::LLVMFuncOp func) {
        if (func->getAttr("gpu.par.kernel")) {
          (void)mlir::convertLLVMToAffineAccess(func, dl, false);
          mlir::gpu::affine_opt::optGlobalSharedMemCopies(func);
        }
      });
    });
  }
};

std::unique_ptr<Pass> mlir::createGPUAffineOptPass() {
  return std::make_unique<GPUAffineOptPass>();
}
