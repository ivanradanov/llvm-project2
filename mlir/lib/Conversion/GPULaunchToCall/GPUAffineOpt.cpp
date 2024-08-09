
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Value.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"

namespace mlir {
namespace gpu {
namespace affine_opt {

struct Copy {
  MemrefValue src, dst;
  SmallVector<Value, 5> srcOperands;
  SmallVector<Value, 5> dstOperands;
};

void optGlobalSharedMemCopies(Operation *root) {
  SmallVector<memref::AllocaOp> sharedMemAllocas;
  root->walk([&](memref::AllocaOp alloca) {
    if (nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
            alloca.getType().getMemorySpace()))
      sharedMemAllocas.push_back(alloca);
  });

  affine::AffineParallelOp outermostBlockPar = nullptr;
  root->walk([&](affine::AffineParallelOp par) {
    if (par->hasAttr("gpu.par.block.z") || par->hasAttr("gpu.par.block.y") ||
        par->hasAttr("gpu.par.block.x"))
      if (outermostBlockPar) {
        if (par->isProperAncestor(outermostBlockPar)) {
          outermostBlockPar = par;
        }
      } else {
        outermostBlockPar = par;
      }
  });

  Block *block = outermostBlockPar->getBlock();
  SmallVector<Value> gridIVs;
  affine::getAffineIVs(*outermostBlockPar, gridIVs);
  unsigned numGridLoops = gridIVs.size();

  for (auto alloca : sharedMemAllocas) {
    LLVM_DEBUG(llvm::dbgs() << "Handling " << *alloca << ":\n");
    SmallVector<Operation *> loads;
    SmallVector<Operation *> stores;
    bool allAffineAccesses = true;
    for (auto opInst : alloca.getMemref().getUsers()) {
      if (isa<affine::AffineReadOpInterface>(opInst)) {
        loads.push_back(opInst);
      } else if (isa<affine::AffineWriteOpInterface>(opInst)) {
        stores.push_back(opInst);
      } else {
        allAffineAccesses = false;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "all affine: " << allAffineAccesses << ":\n");

    auto handleAccess = [&](Operation *access) {
      auto region = std::make_unique<affine::MemRefRegion>(access->getLoc());
      SmallVector<Value> accessIVs;
      affine::getAffineIVs(*access, accessIVs);
      if (failed(region->compute(access, accessIVs.size() - numGridLoops)))
        return;
      LLVM_DEBUG(llvm::dbgs() << "Got memref region for " << *access << ":\n");
      LLVM_DEBUG(region->getConstraints()->dump());
    };

    for (auto access : loads)
      handleAccess(access);
    for (auto access : stores)
      handleAccess(access);
  }

}

} // namespace affine_opt
} // namespace gpu
} // namespace mlir
