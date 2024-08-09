#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_GPULAUNCHTOPARALLELPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "gpu-launch-to-parallel"
#define DEBUG_TYPE PASS_NAME

static LogicalResult propagateConstants(Operation *call,
                                        RewriterBase &rewriter) {
  auto callInterface = dyn_cast<CallOpInterface>(call);
  if (!callInterface) {
    LLVM_DEBUG(llvm::dbgs() << "Not a CallOpInterface: " << *call << "\n");
    return failure();
  }
  SymbolRefAttr callee = llvm::dyn_cast_if_present<SymbolRefAttr>(
      callInterface.getCallableForCallee());
  if (!callee) {
    return failure();
  }
  auto func = SymbolTable::lookupNearestSymbolFrom(call, callee);
  auto funcInterface = dyn_cast<FunctionOpInterface>(func);
  Region *body = &funcInterface.getFunctionBody();

  if (!funcInterface.isPrivate()) {
    return failure();
  }
  assert(!funcInterface.isDeclaration());

  std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(
      func, SymbolTable::getNearestSymbolTable(call));
  // All uses visible
  if (!uses) {
    return failure();
  }
  // Has only one use
  if (std::next(uses->begin()) != uses->end()) {
    return failure();
  }

  auto operands = callInterface.getArgOperands();
  auto args = body->getArguments();
  if (operands.size() != args.size()) {
    return failure();
  }

  unsigned numPropagated = 0;
  rewriter.setInsertionPointToStart(&body->front());
  for (auto [operand, arg] : llvm::zip(operands, args)) {
    if (arg.use_empty())
      continue;
    auto op = operand.getDefiningOp();
    if (!op)
      continue;
    if (!op->hasTrait<OpTrait::ConstantLike>())
      continue;
    assert(op->getNumOperands() == 0);
    unsigned resNo = 0;
    while (op->getResult(resNo) != operand) {
      resNo++;
      assert(resNo < op->getNumResults());
    }
    arg.replaceAllUsesWith(rewriter.clone(*op)->getResult(resNo));
    numPropagated++;
  }

  return success(numPropagated > 0);
}

template <typename NVVMOp>
static void replaceIdDim(RewriterBase &rewriter, Region *region, Value v) {
  OpBuilder::InsertionGuard guard(rewriter);
  region->walk([&](NVVMOp op) {
    Value newV, res = op.getResult();
    if (res.getType() == v.getType()) {
      newV = v;
    } else {
      rewriter.setInsertionPoint(op);
      if (res.getType().isIndex() || v.getType().isIndex())
        newV =
            rewriter.create<arith::IndexCastOp>(res.getLoc(), res.getType(), v);
      else if (res.getType().getIntOrFloatBitWidth() <
               v.getType().getIntOrFloatBitWidth())
        newV = rewriter.create<arith::TruncIOp>(res.getLoc(), res.getType(), v);
      else
        newV = rewriter.create<arith::ExtUIOp>(res.getLoc(), res.getType(), v);
    }
    rewriter.replaceAllUsesWith(res, newV);
    rewriter.eraseOp(op);
  });
}

static constexpr int32_t getSharedMemAddrSpace() { return 3; }

namespace mlir {
struct ConvertedKernel {
  SmallString<128> name;
  Operation *kernel;
};
FailureOr<ConvertedKernel> convertGPUKernelToParallel(Operation *gpuKernelFunc,
                                                      Type shmemSizeType,
                                                      RewriterBase &rewriter,
                                                      bool generateNewKernel) {
  MLIRContext *context = gpuKernelFunc->getContext();
  auto funcLoc = gpuKernelFunc->getLoc();
  rewriter.setInsertionPoint(gpuKernelFunc);

  // TODO assert all are the same
  // TODO get this from datalayout, should be same as index
  Type boundType = rewriter.getI64Type();

  constexpr unsigned additionalArgs = 7;

  Block *newEntryBlock = nullptr;
  Operation *newKernel;
  Region *kernelRegion = nullptr;
  SmallString<128> newSymName;
  if (isa<gpu::GPUFuncOp>(gpuKernelFunc)) {
    return failure();
  } else if (auto llvmKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
    auto st = SymbolTable::getNearestSymbolTable(gpuKernelFunc);
    if (!st)
      return failure();
    if (!generateNewKernel) {
      newSymName = "__mlir.par.kernel." + llvmKernel.getSymName().str();
      unsigned counter;
      newSymName = SymbolTable::generateSymbolName<128>(
          newSymName,
          [&st](StringRef newName) {
            return SymbolTable::lookupSymbolIn(st, newName);
          },
          counter);
    } else {
      newSymName = "__mlir.par.kernel." + llvmKernel.getSymName().str();
      auto existing = SymbolTable::lookupSymbolIn(st, newSymName);
      if (existing)
        return ConvertedKernel{newSymName, existing};
    }

    auto fty = llvmKernel.getFunctionType();
    assert(!fty.isVarArg());
    SmallVector<Type> paramTypes;
    SmallVector<Location> paramLocs;
    SmallVector<DictionaryAttr> paramAttrs;
    paramTypes.insert(paramTypes.begin(), 6, boundType);
    paramAttrs.insert(paramAttrs.begin(), 6,
                      DictionaryAttr::getWithSorted(context, {}));
    paramLocs.insert(paramLocs.begin(), 6, funcLoc);
    paramTypes.push_back(shmemSizeType);
    paramAttrs.push_back(DictionaryAttr::getWithSorted(context, {}));
    paramLocs.push_back(funcLoc);
    paramTypes.insert(paramTypes.end(), fty.getParams().begin(),
                      fty.getParams().end());
    llvmKernel.getAllArgAttrs(paramAttrs);
    for (unsigned i = 0; i < llvmKernel.getNumArguments(); i++)
      paramLocs.push_back(llvmKernel.getBody().front().getArgument(i).getLoc());

    [[maybe_unused]] unsigned newArgs =
        additionalArgs + llvmKernel.getNumArguments();
    assert(paramLocs.size() == newArgs && paramTypes.size() == newArgs &&
           paramAttrs.size() == newArgs);

    // TODO we can use affine::AffineScopeOp in an llvm func
    auto newFty = LLVM::LLVMFunctionType::get(fty.getReturnType(), paramTypes,
                                              fty.isVarArg());
    auto oldAttrs = llvmKernel->getAttrs();
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : oldAttrs)
      if (attr.getName() != "function_type" && attr.getName() != "arg_attrs" &&
          attr.getName() != "CConv" && attr.getName() != "linkage" &&
          attr.getName() != "sym_name")
        newAttrs.push_back(attr);
    LLVM::LLVMFuncOp funcOp;
    newKernel = funcOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcLoc, newSymName.c_str(), newFty, llvmKernel.getLinkage(),
        llvmKernel.getDsoLocal(), llvmKernel.getCConv(),
        llvmKernel.getComdatAttr(), newAttrs, paramAttrs,
        llvmKernel.getFunctionEntryCount());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
    funcOp->setAttr("gpu.par.kernel", rewriter.getUnitAttr());
    funcOp.setLinkage(LLVM::Linkage::Private);

    newEntryBlock =
        rewriter.createBlock(&funcOp.getBody(), {}, paramTypes, paramLocs);
    kernelRegion = &llvmKernel.getFunctionBody();
  } else {
    // TODO
    return failure();
  }

  auto getArg = [&](unsigned i) -> Value {
    Value v = newEntryBlock->getArgument(i);
    if (v.getType() == rewriter.getIndexType())
      return v;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newEntryBlock);
    return rewriter
        .create<arith::IndexCastOp>(v.getLoc(), rewriter.getIndexType(), v)
        .getResult();
  };

  SmallVector<Value, 6> ivs;
  auto createPar = [&](unsigned argPos, unsigned argNum, StringRef attrName) {
    SmallVector<AffineMap> idMaps, zeroMaps;
    SmallVector<Value> lbVs, ubVs;
    auto idMap = AffineMap::getMultiDimIdentityMap(1, context);
    auto zeroMap = AffineMap::getConstantMap(0, context);
    idMaps.insert(idMaps.begin(), argNum, idMap);
    zeroMaps.insert(zeroMaps.begin(), argNum, zeroMap);

    for (unsigned i = 0; i < argNum; i++)
      ubVs.push_back(getArg(argPos + i));

    SmallVector<int64_t> steps(argNum, 1);
    auto par = rewriter.create<affine::AffineParallelOp>(
        funcLoc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), zeroMaps,
        ValueRange(), idMaps, ubVs, steps);
    par->setAttr(attrName, rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(par.getBody());
    ivs.insert(ivs.end(), par.getIVs().begin(), par.getIVs().end());
    return par;
  };
  // TODO combine them
  StringRef attrName = "gpu.par.grid";
  unsigned argPos = 2;
  unsigned argNum = 1;
  [[maybe_unused]] auto gridParZ = createPar(argPos--, argNum, attrName);
  [[maybe_unused]] auto gridParY = createPar(argPos--, argNum, attrName);
  [[maybe_unused]] auto gridParX = createPar(argPos--, argNum, attrName);
  attrName = "gpu.par.block";
  argPos = 5;
  [[maybe_unused]] auto blockParZ = createPar(argPos--, argNum, attrName);
  [[maybe_unused]] auto blockParY = createPar(argPos--, argNum, attrName);
  [[maybe_unused]] auto blockParX = createPar(argPos--, argNum, attrName);

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
  Operation *term = &kernelRegion->front().back();
  assert(term->getNumResults() == 0);
  rewriter.setInsertionPointToEnd(newEntryBlock);
  rewriter.clone(*term, mapping);

  // clang-format off
  unsigned dim = 0;
  replaceIdDim<NVVM::GridDimXOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));
  replaceIdDim<NVVM::GridDimYOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));
  replaceIdDim<NVVM::GridDimZOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));
  replaceIdDim<NVVM::BlockDimXOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));
  replaceIdDim<NVVM::BlockDimYOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));
  replaceIdDim<NVVM::BlockDimZOp>(rewriter, newEntryBlock->getParent(), newEntryBlock->getArgument(dim++));

  dim = 0;
  replaceIdDim<NVVM::BlockIdXOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  replaceIdDim<NVVM::BlockIdYOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  replaceIdDim<NVVM::BlockIdZOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  replaceIdDim<NVVM::ThreadIdXOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  replaceIdDim<NVVM::ThreadIdYOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  replaceIdDim<NVVM::ThreadIdZOp>(rewriter, newEntryBlock->getParent(), ivs[dim++]);
  // clang-format on

  // TODO unused currently but left here just in case
  newEntryBlock->getParent()->walk([&](gpu::GridDimOp gridDim) {
    auto dim = (unsigned)gridDim.getDimension();
    assert(0 <= dim && dim < 3);
    rewriter.replaceAllUsesWith(gridDim, newEntryBlock->getArgument(dim));
    rewriter.eraseOp(gridDim);
  });
  newEntryBlock->getParent()->walk([&](gpu::BlockDimOp blockDim) {
    auto dim = (unsigned)blockDim.getDimension();
    assert(0 <= dim && dim < 3);
    rewriter.replaceAllUsesWith(blockDim, newEntryBlock->getArgument(3 + dim));
    rewriter.eraseOp(blockDim);
  });
  newEntryBlock->getParent()->walk([&](gpu::BlockIdOp blockId) {
    auto dim = (unsigned)blockId.getDimension();
    assert(0 <= dim && dim < 3);
    rewriter.replaceAllUsesWith(blockId, ivs[dim]);
    rewriter.eraseOp(blockId);
  });
  newEntryBlock->getParent()->walk([&](gpu::ThreadIdOp threadId) {
    auto dim = (unsigned)threadId.getDimension();
    assert(0 <= dim && dim < 3);
    rewriter.replaceAllUsesWith(threadId, ivs[3 + dim]);
    rewriter.eraseOp(threadId);
  });

  DenseMap<LLVM::GlobalOp, Value> globalToAlloca;
  newKernel->walk([&](LLVM::AddressOfOp addrOf) {
    auto global = cast<LLVM::GlobalOp>(SymbolTable::lookupNearestSymbolFrom(
        newKernel, addrOf.getGlobalNameAttr()));
    if (global.getAddrSpace() != getSharedMemAddrSpace())
      return;
    Value alloca;
    if (globalToAlloca.count(global)) {
      alloca = globalToAlloca.lookup(global);
    } else {
      OpBuilder blockBuilder = OpBuilder::atBlockBegin(gridParX.getBody());
      auto arrayType = cast<LLVM::LLVMArrayType>(global.getGlobalType());
      alloca = blockBuilder
                   .create<LLVM::AllocaOp>(
                       global.getLoc(), addrOf.getRes().getType(), arrayType,
                       blockBuilder.create<arith::ConstantIntOp>(
                           global->getLoc(), 1, blockBuilder.getI32Type()))
                   .getResult();
      globalToAlloca[global] = alloca;
    }
    addrOf.getRes().replaceAllUsesWith(alloca);
    addrOf->erase();
  });

  return ConvertedKernel{newSymName, newKernel};
}

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
    return failure();
  }

  auto kernelSymbol = launchOp.getKernel();
  auto gpuKernelFunc = SymbolTable::lookupSymbolIn(st, kernelSymbol);
  Type shmemSizeType = launchOp.getDynamicSharedMemorySize().getType();
  auto converted =
      convertGPUKernelToParallel(gpuKernelFunc, shmemSizeType, rewriter, true);
  if (failed(converted))
    return failure();

  Operation *newKernelFunc = converted->kernel;
  SmallString<128> newKernelName = converted->name;

  auto kernelModule = newKernelFunc->getParentOfType<gpu::GPUModuleOp>();
  auto newKernelSymbol = SymbolRefAttr::get(
      kernelModule.getNameAttr(),
      {SymbolRefAttr::get(StringAttr::get(context, newKernelName.c_str()))});

  SmallVector<Value, 20> operands;
  operands.push_back(launchOp.getGridSizeX());
  operands.push_back(launchOp.getGridSizeY());
  operands.push_back(launchOp.getGridSizeZ());
  operands.push_back(launchOp.getBlockSizeX());
  operands.push_back(launchOp.getBlockSizeY());
  operands.push_back(launchOp.getBlockSizeZ());
  operands.push_back(launchOp.getDynamicSharedMemorySize());
  operands.insert(operands.end(), launchOp.getKernelOperands().begin(),
                  launchOp.getKernelOperands().end());

  Type tokenType = nullptr;
  if (launchOp.getAsyncToken())
    tokenType = launchOp.getAsyncToken().getType();
  rewriter.setInsertionPoint(launchOp);
  auto callOp = rewriter.create<gpu::CallOp>(
      launchOp.getLoc(), newKernelSymbol, operands, tokenType,
      launchOp.getAsyncObject(), launchOp.getAsyncDependencies());
  rewriter.replaceOp(launchOp, callOp);

  return success();
}
} // namespace mlir

namespace {
template <typename T>
struct CallConstantPropagation : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T call,
                                PatternRewriter &rewriter) const override {
    return propagateConstants(call, rewriter);
  }
};
} // namespace

struct GPULaunchToParallelPass
    : public impl::GPULaunchToParallelPassBase<GPULaunchToParallelPass> {
  using Base::Base;
  void runOnOperation() override {
    auto context = &getContext();
    IRRewriter rewriter(context);
    getOperation()->walk([&](gpu::LaunchFuncOp launchOp) {
      (void)convertGPULaunchFuncToParallel(launchOp, rewriter);
    });

    RewritePatternSet patterns(context);
    patterns.insert<CallConstantPropagation<gpu::CallOp>>(context);
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  }
};
std::unique_ptr<Pass> mlir::createGPULaunchToParallelPass() {
  return std::make_unique<GPULaunchToParallelPass>();
}
