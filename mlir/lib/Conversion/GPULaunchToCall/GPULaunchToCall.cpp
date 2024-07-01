#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPULAUNCHTOCALLPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-gpu-launch-to-call"

struct ConvertGPULaunchToCall
    : public impl::ConvertGPULaunchToCallPassBase<ConvertGPULaunchToCall> {
  using Base::Base;
  void runOnOperation() override {
    auto context = getOperation()->getContext();
    auto res = getOperation()->walk([&](gpu::LaunchFuncOp launchOp) {
      auto st = SymbolTable::getNearestSymbolTable(launchOp);
      if (!st) {
        return WalkResult::interrupt();
      }
      auto gpuKernelFunc =
          SymbolTable::lookupSymbolIn(st, launchOp.getKernel());
      auto callLoc = launchOp->getLoc();
      auto funcLoc = gpuKernelFunc->getLoc();
      OpBuilder builder(gpuKernelFunc);

      // TODO assert all are the same
      Type boundType = launchOp.getGridSizeX().getType();
      Type shmemSizeType = launchOp.getDynamicSharedMemorySize().getType();

      Block *entryBlock;
      if (isa<gpu::GPUFuncOp>(gpuKernelFunc)) {
        return WalkResult::interrupt();
      } else if (auto llvmKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
        auto fty = llvmKernel.getFunctionType();
        assert(!fty.isVarArg());
        SmallVector<Type> paramTypes;
        paramTypes.insert(paramTypes.begin(), 6, boundType);
        paramTypes.push_back(shmemSizeType);
        paramTypes.insert(paramTypes.end(), fty.getParams().begin(),
                          fty.getParams().end());
        auto newFty = LLVM::LLVMFunctionType::get(fty.getReturnType(),
                                                  paramTypes, fty.isVarArg());

        LLVM::LLVMFuncOp llvmFuncOp = builder.create<LLVM::LLVMFuncOp>(
            funcLoc, llvmKernel.getSymName(), newFty, llvmKernel.getLinkage(),
            llvmKernel.getDsoLocal(), llvmKernel.getCConv(),
            llvmKernel.getComdatAttr(), llvmKernel->getAttrs(),
            /* TODO arg attrs */ ArrayRef<DictionaryAttr>(),
            llvmKernel.getFunctionEntryCount());
        entryBlock = builder.createBlock(&llvmFuncOp.getBody());
      } else if (auto funcKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
        // TODO
        return WalkResult::interrupt();
      }

      auto createPar = [&](unsigned argPos) {
        SmallVector<AffineMap> idMaps, zeroMaps;
        SmallVector<Value> lbVs, ubVs;
        auto idMap =
            AffineMap::getMultiDimIdentityMap(1, launchOp.getContext());
        auto zeroMap = AffineMap::getConstantMap(0, launchOp.getContext());
        idMaps.insert(idMaps.begin(), 3, idMap);
        zeroMaps.insert(zeroMaps.begin(), 3, zeroMap);

        ubVs.push_back(entryBlock->getArgument(argPos++));
        ubVs.push_back(entryBlock->getArgument(argPos++));
        ubVs.push_back(entryBlock->getArgument(argPos++));

        SmallVector<int64_t> steps = {1, 1, 1};
        auto par = builder.create<affine::AffineParallelOp>(
            funcLoc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), idMaps,
            ValueRange(), idMaps, ubVs, steps);
        return par;
      };
      auto gridPar = createPar(0);
      builder.setInsertionPointToStart(gridPar.getBody());
      auto blockPar = createPar(3);
      builder.setInsertionPointToStart(blockPar.getBody());
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createConvertGPULaunchToCallPass() {
  std::make_unique<ConvertGPULaunchToCall>();
}
