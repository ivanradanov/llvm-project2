#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "LoopUndistribute.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>

namespace mlir {
#define GEN_PASS_DEF_GPULAUNCHTOPARALLELPASS
#define GEN_PASS_DEF_GPUPARALLELTOLAUNCHPASS
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

template <typename T>
struct FindSingleResult {
  SmallVector<Operation *> before, after;
  T theOp;
};
template <typename T>
static FindSingleResult<T>
findSingleNestedOp(Operation *parent, std::function<bool(Operation *)> isIt) {
  assert(parent->getNumRegions() == 1);
  Region &region = parent->getRegion(0);
  assert(region.getBlocks().size() == 1);
  Block *block = &region.front();

  FindSingleResult<T> fsr;

  Operation *theOp = nullptr;
  for (auto &op : *block) {
    if (isIt(&op)) {
      assert(!theOp);
      theOp = &op;
    } else if (theOp) {
      fsr.after.push_back(&op);
    } else {
      fsr.before.push_back(&op);
    }
  }
  assert(theOp);
  fsr.theOp = cast<T>(theOp);
  return fsr;
}

namespace mlir {

struct ConvertedKernel {
  SmallString<128> name;
  Operation *kernel;
};
FailureOr<ConvertedKernel> convertGPUKernelToParallel(Operation *gpuKernelFunc,
                                                      Type shmemSizeType,
                                                      RewriterBase &rewriter,
                                                      bool generateNewKernel) {
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *context = gpuKernelFunc->getContext();
  auto funcLoc = gpuKernelFunc->getLoc();
  rewriter.setInsertionPoint(gpuKernelFunc);

  // TODO assert all are the same
  // TODO get this from datalayout, should be same as index
  // it should be the host index size
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
    newSymName = "__mlir_par_kernel_" + llvmKernel.getSymName().str();
    if (generateNewKernel) {
      unsigned counter = 0;
      newSymName = SymbolTable::generateSymbolName<128>(
          newSymName,
          [&st](StringRef newName) {
            return SymbolTable::lookupSymbolIn(st, newName);
          },
          counter);
    } else {
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

    auto newFty = LLVM::LLVMFunctionType::get(fty.getReturnType(), paramTypes,
                                              fty.isVarArg());
    auto oldAttrs = llvmKernel->getAttrs();
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : oldAttrs)
      if (attr.getName() != "function_type" && attr.getName() != "arg_attrs" &&
          attr.getName() != "CConv" && attr.getName() != "linkage" &&
          attr.getName() != "sym_name" && attr.getName() != "comdat")
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
    auto zeroMap = AffineMap::getConstantMap(0, context);
    zeroMaps.insert(zeroMaps.begin(), argNum, zeroMap);
    for (unsigned i = 0; i < argNum; i++) {
      auto idMap = AffineMap::get(0, argNum, getAffineSymbolExpr(i, context));
      idMaps.push_back(idMap);
    }

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
  unsigned argPos = 0;
  unsigned argNum = 3;
  // clang-format off
  [[maybe_unused]] auto gridPar = createPar(argPos--, argNum, "gpu.par.grid");
  argPos = 3;
  [[maybe_unused]] auto blockPar = createPar(argPos--, argNum, "gpu.par.block");
  // clang-format on

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
    assert(dim < 3);
    rewriter.replaceAllUsesWith(gridDim, newEntryBlock->getArgument(dim));
    rewriter.eraseOp(gridDim);
  });
  newEntryBlock->getParent()->walk([&](gpu::BlockDimOp blockDim) {
    auto dim = (unsigned)blockDim.getDimension();
    assert(dim < 3);
    rewriter.replaceAllUsesWith(blockDim, newEntryBlock->getArgument(3 + dim));
    rewriter.eraseOp(blockDim);
  });
  newEntryBlock->getParent()->walk([&](gpu::BlockIdOp blockId) {
    auto dim = (unsigned)blockId.getDimension();
    assert(dim < 3);
    rewriter.replaceAllUsesWith(blockId, ivs[dim]);
    rewriter.eraseOp(blockId);
  });
  newEntryBlock->getParent()->walk([&](gpu::ThreadIdOp threadId) {
    auto dim = (unsigned)threadId.getDimension();
    assert(dim < 3);
    rewriter.replaceAllUsesWith(threadId, ivs[3 + dim]);
    rewriter.eraseOp(threadId);
  });

  // Barriers
  SmallVector<Value, 3> blockIndices(blockPar.getIVs());
  newEntryBlock->getParent()->walk([&](NVVM::Barrier0Op barrier) {
    rewriter.setInsertionPoint(barrier);
    rewriter.replaceOpWithNewOp<affine::AffineBarrierOp>(barrier, blockIndices);
    return WalkResult::skip();
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
      OpBuilder blockBuilder = OpBuilder::atBlockBegin(gridPar.getBody());
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
  Value shmemSize = launchOp.getDynamicSharedMemorySize();
  Type shmemSizeType;
  if (shmemSize) {
    shmemSizeType = shmemSize.getType();
    auto cst =
        dyn_cast_or_null<arith::ConstantIntOp>(shmemSize.getDefiningOp());
    if (!(cst && cst.value() == 0)) {
      return failure();
    }
  } else {
    shmemSizeType = rewriter.getI32Type();
  }

  auto kernelSymbol = launchOp.getKernel();
  auto gpuKernelFunc = SymbolTable::lookupSymbolIn(st, kernelSymbol);
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
  if (shmemSize)
    operands.push_back(shmemSize);
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

static mlir::Value createConstantInt(RewriterBase &rewriter, Location loc,
                                     Type ty, int64_t v) {
  if (ty.isIndex())
    return rewriter.create<arith::ConstantIndexOp>(loc, v);
  else
    return rewriter.create<arith::ConstantIntOp>(loc, v, ty);
}

struct ConvertedForLaunch {
  SymbolRefAttr symbol;
  Value shmem;
  Value stream;
  gpu::KernelDim3 blocks, threads;
};

LogicalResult convertGPUCallToLaunch(gpu::CallOp callOp,
                                     RewriterBase &rewriter) {
  auto st = SymbolTable::getNearestSymbolTable(callOp);
  if (!st)
    return failure();

  auto kernelSymbol = callOp.getKernel();
  Operation *gpuKernelFunc = SymbolTable::lookupSymbolIn(st, kernelSymbol);
  if (!gpuKernelFunc)
    return rewriter.notifyMatchFailure(callOp->getLoc(), "Kernel not found");
  auto loc = gpuKernelFunc->getLoc();

  auto gridPar = findSingleNestedOp<scf::ParallelOp>(
      gpuKernelFunc, gpu::affine_opt::isGridPar);
  auto blockPar = findSingleNestedOp<scf::ParallelOp>(
      gridPar.theOp, gpu::affine_opt::isBlockPar);

  st = SymbolTable::getNearestSymbolTable(gpuKernelFunc);
  if (!st)
    return failure();

  IRMapping toHost;
  ConvertedForLaunch cfl;
  Block *newEntryBlock = nullptr;
  [[maybe_unused]]
  Operation *newKernel;
  [[maybe_unused]]
  Region *kernelRegion = nullptr;
  SmallString<128> newSymName;
  if (isa<gpu::GPUFuncOp>(gpuKernelFunc)) {
    return failure();
  } else if (auto llvmKernel = dyn_cast<LLVM::LLVMFuncOp>(gpuKernelFunc)) {
    newSymName = "__mlir_launch_kernel_" + llvmKernel.getSymName().str();
    unsigned counter = 0;
    newSymName = SymbolTable::generateSymbolName<128>(
        newSymName,
        [&st](StringRef newName) {
          return SymbolTable::lookupSymbolIn(st, newName);
        },
        counter);

    auto fty = llvmKernel.getFunctionType();
    assert(!fty.isVarArg());

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(gpuKernelFunc);
    LLVM::LLVMFuncOp funcOp;
    newKernel = funcOp =
        cast<LLVM::LLVMFuncOp>(rewriter.cloneWithoutRegions(*gpuKernelFunc));
    funcOp.setSymName(newSymName);
    funcOp.setVisibility(SymbolTable::Visibility::Public);
    funcOp->removeAttr("gpu.par.kernel");
    funcOp.setLinkage(LLVM::Linkage::External);

    newEntryBlock = funcOp.addEntryBlock(rewriter);
    kernelRegion = &llvmKernel.getFunctionBody();

    auto kernelModule = gpuKernelFunc->getParentOfType<gpu::GPUModuleOp>();
    auto kernelSymbol = SymbolRefAttr::get(
        kernelModule.getNameAttr(), {SymbolRefAttr::get(funcOp.getNameAttr())});
    cfl.symbol = kernelSymbol;
    toHost.map(llvmKernel.getArguments(), callOp.getKernelOperands());
  } else {
    return failure();
  }

  auto assertIsMemEffectFree = [&](Operation *op) {
    if (!isMemoryEffectFree(op)) {
      op->dump();
      llvm_unreachable("Operation not memory effect free");
    }
  };
  rewriter.setInsertionPoint(callOp);
  SmallVector<Operation *> clonedToHost;
  auto clone = [&](Operation *op) {
    // This is a workaround. The device may use index of a different width than
    // the host and it may be in the middle of lowering so some unrealized casts
    // (e.g. i32 to index) which don't match the lowering on the host. This
    // happens because we do some lowering in the gpu-affine-opt but not
    // entirely (we leave the scf.parallel's which still take index instead of
    // i32/i64)
    if (auto ucc = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (ucc.getInputs().size() == 1 && ucc.getOutputs().size() == 1) {
        auto in = ucc.getInputs()[0];
        auto out = ucc.getOutputs()[0];
        if (in.getType() == rewriter.getIndexType() ||
            out.getType() == rewriter.getIndexType()) {
          auto newOut = rewriter.create<arith::IndexCastOp>(
              op->getLoc(), out.getType(), toHost.lookupOrDefault(in));
          toHost.map(out, newOut);
          return;
        }
      }
    }
    Operation *cloned = rewriter.clone(*op, toHost);
    clonedToHost.push_back(cloned);
  };
  for (auto *op : gridPar.before) {
    assertIsMemEffectFree(op);
    clone(op);
  }
  for (auto *op : blockPar.before) {
    assertIsMemEffectFree(op);
    if (llvm::all_of(op->getOperands(),
                     [&](Value v) { return toHost.lookupOrNull(v); }))
      clone(op);
  }
  for (auto *op : gridPar.after)
    assertIsMemEffectFree(op);
  for (auto *op : blockPar.after)
    assertIsMemEffectFree(op);

  // FIXME assert normalized parallel loops [0; n)

  auto hoistDims = [&](gpu::KernelDim3 &dims, scf::ParallelOp par) {
    SmallVector<Value *, 3> dimsVec = {&dims.x, &dims.y, &dims.z};
    for (auto [dim, ub] : llvm::zip_longest(dimsVec, par.getUpperBound())) {
      assert(dim);
      if (ub) {
        **dim = toHost.lookup(*ub);
      } else {
        assert(*dimsVec[0]);
        **dim = createConstantInt(rewriter, loc, dimsVec[0]->getType(), 1);
      }
    }
  };
  hoistDims(cfl.blocks, gridPar.theOp);
  hoistDims(cfl.threads, blockPar.theOp);

  // FIXME 0 for now
  cfl.shmem = nullptr;
  // FIXME default for now
  cfl.stream = nullptr;

  IRMapping toNew;
  toNew.map(gpuKernelFunc->getRegion(0).getArguments(),
            newEntryBlock->getArguments());
  rewriter.setInsertionPointToStart(newEntryBlock);

  for (auto *op : gridPar.before)
    clonedToHost.push_back(rewriter.clone(*op, toNew));

  assert(gridPar.theOp.getNumLoops() <= 3);
  for (unsigned i = 0; i < gridPar.theOp.getNumLoops(); ++i)
    toNew.map(gridPar.theOp.getInductionVars()[i],
              rewriter.create<gpu::BlockIdOp>(loc, (gpu::Dimension)i));

  for (auto *op : blockPar.before)
    clonedToHost.push_back(rewriter.clone(*op, toNew));

  assert(blockPar.theOp.getNumLoops() <= 3);
  for (unsigned i = 0; i < blockPar.theOp.getNumLoops(); ++i)
    toNew.map(blockPar.theOp.getInductionVars()[i],
              rewriter.create<gpu::ThreadIdOp>(loc, (gpu::Dimension)i));

  assert(blockPar.theOp->getNumRegions() == 1);
  Region &region = blockPar.theOp->getRegion(0);
  assert(region.getBlocks().size() == 1);
  Block *block = &region.front();

  assert(block->back().getNumOperands() == 0);
  for (auto &op : llvm::drop_end(*block))
    rewriter.clone(op, toNew);
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

  LLVM_DEBUG(assert(verify(newKernel, true).succeeded()));

  // We should be using streams currently.
  assert(!callOp.getAsyncToken());
  rewriter.setInsertionPoint(callOp);
  auto launch = rewriter.create<gpu::LaunchFuncOp>(
      callOp.getLoc(), cfl.symbol, cfl.blocks, cfl.threads, cfl.shmem,
      callOp.getKernelOperands(), callOp.getAsyncObject());
  rewriter.replaceOp(callOp, launch.getResults());

  // We need this to get rid of the invalid addressof ops that got cloned to the
  // host.
  for (auto *op : llvm::reverse(clonedToHost))
    if (isOpTriviallyDead(op))
      rewriter.eraseOp(op);

  return success();
}

} // namespace mlir

namespace {
static Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!values.empty() && "empty min/max chain");
  assert(predicate == arith::CmpIPredicate::sgt ||
         predicate == arith::CmpIPredicate::slt);

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    if (predicate == arith::CmpIPredicate::sgt)
      value = builder.create<arith::MaxSIOp>(loc, value, *valueIt);
    else
      value = builder.create<arith::MinSIOp>(loc, value, *valueIt);
  }

  return value;
}
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = affine::expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}
struct PromoteIfSingleIteration
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    auto ranges = op.getConstantRanges();
    if (!ranges)
      return failure();
    unsigned iterations = 1;
    for (auto range : *ranges)
      iterations *= range;
    if (iterations != 1)
      return failure();
    SmallVector<Value, 3> argReplacements;
    for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
      Value lower =
          lowerAffineMapMax(rewriter, op.getLoc(), op.getLowerBoundMap(i),
                            op.getLowerBoundsOperands());
      if (!lower)
        return rewriter.notifyMatchFailure(op, "couldn't convert lower bounds");
      argReplacements.push_back(lower);
    }
    auto term = op.getBody()->getTerminator();
    rewriter.inlineBlockBefore(op.getBody(), op, argReplacements);
    rewriter.replaceOp(op, term->getResults());
    rewriter.eraseOp(term);

    return success();
  }
};
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

struct GPUParallelToLaunchPass
    : public impl::GPUParallelToLaunchPassBase<GPUParallelToLaunchPass> {
  using Base::Base;
  void runOnOperation() override {
    auto context = &getContext();
    IRRewriter rewriter(context);
    getOperation()->walk([&](gpu::CallOp callOp) {
      (void)convertGPUCallToLaunch(callOp, rewriter);
    });
    getOperation()->walk([&](gpu::GPUModuleOp gpum) {
      for (auto &op :
           llvm::make_early_inc_range(gpum.getBodyRegion().getOps())) {
        if (op.getAttr("gpu.par.kernel")) {
          assert(cast<mlir::SymbolOpInterface>(op).symbolKnownUseEmpty(gpum));
          op.erase();
        }
      }
    });
  }
};
std::unique_ptr<Pass> mlir::createGPULaunchToParallelPass() {
  return std::make_unique<GPULaunchToParallelPass>();
}

std::unique_ptr<Pass> mlir::createGPUParallelToLaunchPass() {
  return std::make_unique<GPUParallelToLaunchPass>();
}
