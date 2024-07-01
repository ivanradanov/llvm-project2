#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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
      if (!st)
        return WalkResult::interrupt();

      std::string funcName = launchOp.getKernelModuleName().str() + "." +
                             launchOp.getKernelName().str() + ".mlir.outlined";
      auto symOp = SymbolTable::lookupSymbolIn(st, funcName);

      assert(launchOp.getAsyncDependencies().size() == 0);
      IRRewriter rewriter(launchOp);
      func::FuncOp func = cast_or_null<func::FuncOp>(symOp);
      auto loc = launchOp->getLoc();
      if (!func) {
        assert(!symOp);
        OpBuilder builder = OpBuilder::atBlockBegin(&st->getRegion(0).front());
        func = builder.create<func::FuncOp>(
            loc, funcName,
            FunctionType::get(context, launchOp->getOperandTypes(),
                              TypeRange()));
        Block *entryBlock = builder.createBlock(
            &func.getBody(), {}, launchOp->getOperandTypes(),
            SmallVector<Location>(launchOp->getNumOperands(),
                                  builder.getUnknownLoc()));
        IRMapping mapping;
        mapping.map(launchOp.getOperands(), entryBlock->getArguments());
        auto newLaunchOp = builder.clone(*launchOp, mapping);
        builder.create<func::ReturnOp>(loc, ValueRange());
      }
      rewriter.replaceOpWithNewOp<func::CallOp>(launchOp, funcName, TypeRange(),
                                                launchOp.getOperands());
      return WalkResult::advance();
    });
    // TODO needs stream support
    res = getOperation()->walk([&](gpu::LaunchFuncOp launchOp) {
      auto st = SymbolTable::getNearestSymbolTable(launchOp);
      if (!st)
        return WalkResult::interrupt();

      // TODO Only kernels with no dynamic mem for now
      Value shMemSize = launchOp.getDynamicSharedMemorySize();
      auto cst =
          dyn_cast_or_null<arith::ConstantIntOp>(shMemSize.getDefiningOp());
      if (!(cst && cst.value() == 0)) {
        // TODO put this back once we get rid of cudaPush/Pop calls which hide
        // it

        // return WalkResult::interrupt();
      }

      auto kernelSymbol = launchOp.getKernel();
      auto gpuKernelFunc = SymbolTable::lookupSymbolIn(st, kernelSymbol);
      auto callLoc = launchOp->getLoc();
      auto funcLoc = gpuKernelFunc->getLoc();
      OpBuilder builder(gpuKernelFunc);

      // TODO assert all are the same
      Type boundType = builder.getIndexType();
      Type shmemSizeType = launchOp.getDynamicSharedMemorySize().getType();

      constexpr unsigned additionalArgs = 7;

      Block *newEntryBlock = nullptr;
      Region *kernelRegion = nullptr;
      if (isa<gpu::GPUFuncOp>(gpuKernelFunc)) {
        return WalkResult::interrupt();
      } else if (auto llvmKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
        auto fty = llvmKernel.getFunctionType();
        assert(!fty.isVarArg());
        SmallVector<Type> paramTypes;
        SmallVector<Location> paramLocs;
        SmallVector<DictionaryAttr> paramAttrs;
        paramTypes.insert(paramTypes.begin(), 6, boundType);
        paramAttrs.insert(paramAttrs.begin(), 6,
                          DictionaryAttr::getWithSorted(context, {}));
        paramLocs.insert(paramLocs.begin(), 6, callLoc);
        paramTypes.push_back(shmemSizeType);
        paramAttrs.push_back(DictionaryAttr::getWithSorted(context, {}));
        paramLocs.push_back(callLoc);
        paramTypes.insert(paramTypes.end(), fty.getParams().begin(),
                          fty.getParams().end());
        llvmKernel.getAllArgAttrs(paramAttrs);
        for (unsigned i = 0; i < llvmKernel.getNumArguments(); i++)
          paramLocs.push_back(
              llvmKernel.getBody().front().getArgument(i).getLoc());

        [[maybe_unused]] unsigned newArgs =
            additionalArgs + llvmKernel.getNumArguments();
        assert(paramLocs.size() == newArgs && paramTypes.size() == newArgs &&
               paramAttrs.size() == newArgs);

        auto newFty = FunctionType::get(context, paramTypes, TypeRange());
        func::FuncOp funcOp = builder.create<func::FuncOp>(
            funcLoc, llvmKernel.getSymName(), newFty,
            /* TODO attrs if we pass in llvmKernel->getAttrs() here we get an
               error in the op builder */
            ArrayRef<NamedAttribute>(), paramAttrs);
        // TODO temp until we get attrs working
        funcOp->setAttr("gpu.kernel", builder.getUnitAttr());
        newEntryBlock =
            builder.createBlock(&funcOp.getBody(), {}, paramTypes, paramLocs);
        kernelRegion = &llvmKernel.getFunctionBody();
      } else if (auto funcKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
        // TODO
        return WalkResult::interrupt();
      }

      auto createPar = [&](unsigned argPos, unsigned argNum) {
        SmallVector<AffineMap> idMaps, zeroMaps;
        SmallVector<Value> lbVs, ubVs;
        auto idMap =
            AffineMap::getMultiDimIdentityMap(1, launchOp.getContext());
        auto zeroMap = AffineMap::getConstantMap(0, launchOp.getContext());
        idMaps.insert(idMaps.begin(), argNum, idMap);
        zeroMaps.insert(zeroMaps.begin(), argNum, zeroMap);

        for (int i = 0; i < argNum; i++)
          ubVs.push_back(newEntryBlock->getArgument(argPos + i));

        SmallVector<int64_t> steps(argNum, 1);
        auto par = builder.create<affine::AffineParallelOp>(
            funcLoc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), zeroMaps,
            ValueRange(), idMaps, ubVs, steps);
        builder.setInsertionPointToStart(par.getBody());
        return par;
      };
      // TODO combine them
      unsigned argPos = 0;
      unsigned argNum = 1;
      auto gridParX = createPar(argPos++, argNum);
      auto gridParY = createPar(argPos++, argNum);
      auto gridParZ = createPar(argPos++, argNum);
      // TODO shared mem alloc
      auto blockParX = createPar(argPos++, argNum);
      auto blockParY = createPar(argPos++, argNum);
      auto blockParZ = createPar(argPos++, argNum);

      // TODO handle multi block regions would be better as I think there may be
      // cases where removing CFG may be impossible before having the parallel
      // wrap. need to wrap things in scf execute
      if (!kernelRegion->hasOneBlock())
        return WalkResult::interrupt();

      IRMapping mapping;

      mapping.map(kernelRegion->getArguments(),
                  SmallVector<Value>(newEntryBlock->getArguments().drop_front(
                      additionalArgs)));

      for (auto it = kernelRegion->front().begin();
           it != std::next(kernelRegion->front().end(), -1); it++)
        builder.clone(*it, mapping);

      // is return void
      assert(kernelRegion->front().back().getNumResults() == 0);
      builder.setInsertionPointToEnd(newEntryBlock);
      builder.create<func::ReturnOp>(funcLoc, ValueRange());

      gpuKernelFunc->erase();

      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createConvertGPULaunchToCallPass() {
  return std::make_unique<ConvertGPULaunchToCall>();
}
