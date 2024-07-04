#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>

namespace mlir {
// TODO needs stream support
LogicalResult convertGPULaunchFuncToParallel(gpu::LaunchFuncOp launchOp,
                                             RewriterBase &rewriter) {
  MLIRContext *context = launchOp->getContext();
  auto st = SymbolTable::getNearestSymbolTable(launchOp);
  if (!st)
    return failure();

  // TODO Only kernels with no dynamic mem for now
  Value shMemSize = launchOp.getDynamicSharedMemorySize();
  auto cst = dyn_cast_or_null<arith::ConstantIntOp>(shMemSize.getDefiningOp());
  if (!(cst && cst.value() == 0)) {
    // TODO put this back once we get rid of cudaPush/Pop calls which hide
    // it

    // return WalkResult::interrupt();
  }

  auto kernelSymbol = launchOp.getKernel();
  auto gpuKernelFunc = SymbolTable::lookupSymbolIn(st, kernelSymbol);
  auto callLoc = launchOp->getLoc();
  auto funcLoc = gpuKernelFunc->getLoc();
  rewriter.setInsertionPoint(gpuKernelFunc);

  // TODO assert all are the same
  Type boundType = rewriter.getIndexType();
  Type shmemSizeType = launchOp.getDynamicSharedMemorySize().getType();

  constexpr unsigned additionalArgs = 7;

  Block *newEntryBlock = nullptr;
  Region *kernelRegion = nullptr;
  if (isa<gpu::GPUFuncOp>(gpuKernelFunc)) {
    return failure();
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
      paramLocs.push_back(llvmKernel.getBody().front().getArgument(i).getLoc());

    [[maybe_unused]] unsigned newArgs =
        additionalArgs + llvmKernel.getNumArguments();
    assert(paramLocs.size() == newArgs && paramTypes.size() == newArgs &&
           paramAttrs.size() == newArgs);

    auto newFty = FunctionType::get(context, paramTypes, TypeRange());
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        funcLoc, llvmKernel.getSymName(), newFty,
        /* TODO attrs if we pass in llvmKernel->getAttrs() here we get an
           error in the op builder */
        ArrayRef<NamedAttribute>(), paramAttrs);
    // TODO temp until we get attrs working
    funcOp->setAttr("gpu.kernel", rewriter.getUnitAttr());
    newEntryBlock =
        rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, paramLocs);
    kernelRegion = &llvmKernel.getFunctionBody();
  } else if (auto funcKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
    // TODO
    return failure();
  }

  auto createPar = [&](unsigned argPos, unsigned argNum) {
    SmallVector<AffineMap> idMaps, zeroMaps;
    SmallVector<Value> lbVs, ubVs;
    auto idMap = AffineMap::getMultiDimIdentityMap(1, launchOp.getContext());
    auto zeroMap = AffineMap::getConstantMap(0, launchOp.getContext());
    idMaps.insert(idMaps.begin(), argNum, idMap);
    zeroMaps.insert(zeroMaps.begin(), argNum, zeroMap);

    for (unsigned i = 0; i < argNum; i++)
      ubVs.push_back(newEntryBlock->getArgument(argPos + i));

    SmallVector<int64_t> steps(argNum, 1);
    auto par = rewriter.create<affine::AffineParallelOp>(
        funcLoc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), zeroMaps,
        ValueRange(), idMaps, ubVs, steps);
    rewriter.setInsertionPointToStart(par.getBody());
    return par;
  };
  // TODO combine them
  unsigned argPos = 0;
  unsigned argNum = 1;
  [[maybe_unused]] auto gridParX = createPar(argPos++, argNum);
  [[maybe_unused]] auto gridParY = createPar(argPos++, argNum);
  [[maybe_unused]] auto gridParZ = createPar(argPos++, argNum);
  // TODO shared mem alloc
  [[maybe_unused]] auto blockParX = createPar(argPos++, argNum);
  [[maybe_unused]] auto blockParY = createPar(argPos++, argNum);
  [[maybe_unused]] auto blockParZ = createPar(argPos++, argNum);

  // TODO handle multi block regions would be better as I think there may be
  // cases where removing CFG may be impossible before having the parallel
  // wrap. need to wrap things in scf execute
  if (!kernelRegion->hasOneBlock())
    return failure();

  IRMapping mapping;

  mapping.map(kernelRegion->getArguments(),
              SmallVector<Value>(
                  newEntryBlock->getArguments().drop_front(additionalArgs)));

  for (auto it = kernelRegion->front().begin();
       it != std::next(kernelRegion->front().end(), -1); it++)
    rewriter.clone(*it, mapping);

  // is return void
  assert(kernelRegion->front().back().getNumResults() == 0);
  rewriter.setInsertionPointToEnd(newEntryBlock);
  rewriter.create<func::ReturnOp>(funcLoc, ValueRange());

  rewriter.eraseOp(gpuKernelFunc);

  return success();
}
} // namespace mlir
