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
#define GEN_PASS_DEF_OUTLINEGPUJITREGIONSPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "outline-gpu-jit-regions"
#define DEBUG_TYPE PASS_NAME

static FailureOr<std::unique_ptr<ModuleOp>>
outlineFunctionForJit(func::FuncOp f) {
  Operation *oldModule = f->getParentOp();
  std::unique_ptr<ModuleOp> newModule =
      std::make_unique<ModuleOp>(ModuleOp::create(f.getLoc()));
  (*newModule)->setAttrs(oldModule->getAttrs());

  auto st = SymbolTable::getNearestSymbolTable(f);
  if (!st)
    return failure();
  assert(f->getParentOp() == st);

  std::function<LogicalResult(Operation *, Operation *, RewriterBase &)>
      cloneCG;
  cloneCG = [&cloneCG](Operation *f, Operation *st, RewriterBase &rewriter) {
    CallGraph cg(f);
    IRMapping mapping;
    for (auto *node : cg) {
      if (node->isExternal())
        return failure();
      Region *callableRegion = node->getCallableRegion();
      Operation *callableOp = callableRegion->getParentOp();
      if (callableOp->getParentOp() != st)
        return failure();
      rewriter.clone(*callableOp, mapping);

      SmallPtrSet<Operation *, 1> nestedModules;
      SmallPtrSet<Operation *, 5> nestedFuncs;
      callableOp->walk([&](gpu::LaunchFuncOp launchOp) {
        auto nm =
            SymbolTable::lookupSymbolIn(st, launchOp.getKernelModuleName());
        auto nf = SymbolTable::lookupSymbolIn(st, launchOp.getKernel());
        nestedModules.insert(nm);
        nestedFuncs.insert(nf);
      });
      for (auto nm : nestedModules) {
        auto newNM = rewriter.cloneWithoutRegions(*nm, mapping);
        OpBuilder::InsertionGuard g(rewriter);
        auto newBlock = rewriter.createBlock(&newNM->getRegion(0));
        rewriter.create<gpu::ModuleEndOp>(newNM->getLoc());
        mapping.map(&nm->getRegion(0).front(), newBlock);
      }
      for (auto nf : nestedFuncs) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(mapping.lookup(nf->getBlock()));
        if (cloneCG(nf, nf->getParentOp(), rewriter).failed())
          return failure();
      }
    }
    return success();
  };
  IRRewriter rewriter(f.getContext());
  rewriter.setInsertionPointToStart(newModule->getBody());
  if (cloneCG(f, st, rewriter).failed())
    return failure();

  return newModule;
}

struct OutlineGPUJitRegions
    : public impl::OutlineGPUJitRegionsPassBase<OutlineGPUJitRegions> {
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
      auto loc = launchOp->getLoc();
      func::FuncOp func;
      if (!symOp) {
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
        builder.clone(*launchOp, mapping);
        builder.create<func::ReturnOp>(loc, ValueRange());
      } else {
        func = cast<func::FuncOp>(symOp);
      }
      auto outlinedModule = outlineFunctionForJit(func);
      LLVM_DEBUG({
        if (succeeded(outlinedModule))
          outlinedModule->get()->dump();
      });

      if (!succeeded(outlinedModule))
        return WalkResult::interrupt();

      std::string out;
      llvm::raw_string_ostream os(out);
      os << **outlinedModule;
      StringAttr ir = rewriter.getStringAttr(os.str().c_str());
      rewriter.replaceOpWithNewOp<func::CallJitOp>(
          launchOp, funcName, ir, TypeRange(), launchOp.getOperands());

      assert(
          mlir::verify((*(*outlinedModule)).getOperation(), true).succeeded());
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createOutlineGPUJitRegionsPass() {
  return std::make_unique<OutlineGPUJitRegions>();
}
