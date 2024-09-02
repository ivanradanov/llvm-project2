#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "LoopDistribute.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"
#define DBGS (llvm::dbgs() << "gpu-affine-opt: ")

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
  Operation *loadOp;
  Operation *storeOp;
};

struct AccessInfo {
  SmallVector<Value, 8> regionSymbols;
  SmallVector<Value, 4> lbs, ubs;
  SmallVector<AffineMap, 4> lbMaps, ubMaps;
  unsigned rank;
};

struct VectorStore {
  affine::AffineParallelOp par;
  affine::AffineStoreOp store;
  Value val;
};
struct VectorLoad {
  affine::AffineParallelOp par;
  affine::AffineLoadOp load;
  Value val;
};

static affine::AffineParallelOp isBlockPar(Operation *op) {
  auto blockPar = dyn_cast_or_null<affine::AffineParallelOp>(op);
  if (!blockPar)
    return nullptr;
  if (blockPar->getAttr("gpu.par.block"))
    return blockPar;
  return nullptr;
}

static std::optional<VectorStore> isVectorStore(affine::AffineStoreOp store) {
  if (!store->getParentOp()->hasAttr("affine.vector.store"))
    return {};
  return VectorStore{
      cast<affine::AffineParallelOp>(store->getParentOp()), store,
      cast<vector::ExtractOp>(store.getValueToStore().getDefiningOp())
          .getVector()};
}

static std::optional<VectorLoad> isVectorLoad(affine::AffineParallelOp par) {
  if (!par->hasAttr("affine.vector.load"))
    return {};
  return VectorLoad{
      par,
      cast<affine::AffineLoadOp>(
          cast<vector::BroadcastOp>(
              par.getBody()->getTerminator()->getOperand(0).getDefiningOp())
              .getSource()
              .getDefiningOp()),
      par->getResult(0)};
}

template <typename T>
static Value computeMap(RewriterBase &rewriter, Location loc, AffineMap map,
                        ValueRange operands) {
  if (map.getNumResults() > 1)
    return rewriter.create<T>(loc, map, operands);
  else if (map.getNumResults() == 1)
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

  affine::AffineParallelOp gridPar = nullptr;
  SmallVector<affine::AffineParallelOp> gridPars;
  root->walk([&](affine::AffineParallelOp par) {
    if (par->hasAttr("gpu.par.grid")) {
      assert(!gridPar);
      gridPar = par;
    }
  });

  llvm::SmallSetVector<Operation *, 4> syncsInserted;

  for (auto alloca : shmemAllocas) {
    LLVM_DEBUG(llvm::dbgs() << "Handling " << *alloca << ":\n");
    SmallVector<Copy> copies;
    bool allAffineAccesses = true;
    for (auto opInst : alloca.getMemref().getUsers()) {
      if (auto load = dyn_cast<affine::AffineReadOpInterface>(opInst)) {
        // These can't be optimized, copy_async only supports global->shared
        continue;
      }
      if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(opInst)) {
        if (auto store = dyn_cast<affine::AffineStoreOp>(opInst)) {
          if (auto vectorStore = isVectorStore(store)) {
            if (!vectorStore->val.hasOneUse())
              continue;
            auto parLoad = dyn_cast_or_null<affine::AffineParallelOp>(
                vectorStore->val.getDefiningOp());
            if (!parLoad)
              continue;
            auto vectorLoad = isVectorLoad(parLoad);
            if (!vectorLoad)
              continue;
            if (isGlobalMemref(cast<MemrefValue>(vectorLoad->load.getMemRef())))
              copies.push_back({vectorLoad->load, vectorStore->store,
                                vectorLoad->par, vectorStore->par});
          } else {
            if (!store.getValueToStore().hasOneUse())
              continue;
            auto load = dyn_cast_or_null<affine::AffineReadOpInterface>(
                store.getValueToStore().getDefiningOp());
            if (!load)
              continue;
            if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
              copies.push_back({load, store, load, store});
          }
        } else if (auto store = dyn_cast<affine::AffineVectorStoreOp>(opInst)) {
          if (!store.getValueToStore().hasOneUse())
            continue;
          auto load = dyn_cast_or_null<affine::AffineReadOpInterface>(
              store.getValueToStore().getDefiningOp());
          if (!load)
            continue;
          if (isGlobalMemref(cast<MemrefValue>(load.getMemRef())))
            copies.push_back({load, store, load, store});
        }
      } else {
        allAffineAccesses = false;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "all affine: " << allAffineAccesses << "\n");

    for (auto copy : copies) {
      Location loc = copy.load.getLoc();
      MLIRContext *ctx = copy.load.getContext();

      auto blockPar = copy.store->getParentOfType<affine::AffineParallelOp>();
      assert(blockPar->getAttr("gpu.par.block"));

      IRRewriter rewriter(copy.storeOp);

      LLVM_DEBUG(llvm::dbgs() << "Found copy\n");

      auto vty = dyn_cast<VectorType>(copy.load.getValue().getType());
      if (!vty) {
        LLVM_DEBUG(llvm::dbgs() << "Need a vector load/store\n");
        continue;
      }
      if (vty.getShape().size() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "Need 1d vector load/store\n");
        continue;
      }
      if (vty.getElementType() != rewriter.getI8Type()) {
        LLVM_DEBUG(llvm::dbgs() << "Need i8 vector\n");
        continue;
      }
      unsigned copySize = vty.getShape()[0];
      if (!(copySize == 4 || copySize == 8 || copySize == 16)) {
        LLVM_DEBUG(llvm::dbgs() << "Need 4/8/16 size\n");
        continue;
      }

      // TODO need to align stuff to 4, 8, or 16-byte chunks
      rewriter.setInsertionPoint(copy.store);
      SmallVector<Value> storeIdxs = {computeMap<affine::AffineApplyOp>(
          rewriter, copy.store.getLoc(), copy.store.getAffineMap(),
          copy.store.getMapOperands())};
      SmallVector<Value> loadIdxs = {computeMap<affine::AffineApplyOp>(
          rewriter, copy.load.getLoc(), copy.load.getAffineMap(),
          copy.load.getMapOperands())};
      rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          loc, copy.store.getMemRef(), storeIdxs, copy.load.getMemRef(),
          loadIdxs, rewriter.getIndexAttr(copySize), nullptr, nullptr);
      rewriter.eraseOp(copy.store);
      rewriter.eraseOp(copy.load);

      // TODO Needs revisiting, will break very easily, need to insert the
      // synchronisation before the next use of the shared mem
      Operation *synchronisationPt = blockPar->getNextNode();
      while (!(synchronisationPt == blockPar->getBlock()->getTerminator() ||
               synchronisationPt->getAttr("gpu.par.block")))
        synchronisationPt = synchronisationPt->getNextNode();

      if (!syncsInserted.contains(synchronisationPt)) {
        rewriter.setInsertionPoint(synchronisationPt);
        auto token = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
            loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange());
        rewriter.create<nvgpu::DeviceAsyncWaitOp>(loc, token, nullptr);
        syncsInserted.insert(synchronisationPt);
      }
    }
  }
}

void transform(LLVM::LLVMFuncOp f) {
  std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(f);
}

} // namespace affine_opt
} // namespace gpu
} // namespace mlir

static bool areEquiv(affine::AffineParallelOp a, affine::AffineParallelOp b) {
  return a.getLowerBoundsValueMap() == b.getLowerBoundsValueMap() &&
         a.getLowerBoundsValueMap() == b.getLowerBoundsValueMap() &&
         a.getSteps() == b.getSteps();
}

namespace {

struct FuseBlockPars : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp par,
                                PatternRewriter &rewriter) const override {
    if (gpu::affine_opt::isBlockPar(par))
      return rewriter.notifyMatchFailure(par, "op is not block par");
    auto nextPar = gpu::affine_opt::isBlockPar(par->getNextNode());
    if (!nextPar)
      return rewriter.notifyMatchFailure(par->getNextNode(),
                                         "Next op is not block par");
    if (!areEquiv(par, nextPar))
      return rewriter.notifyMatchFailure(nextPar, "Non equiv pars");

    assert(nextPar.getBody()->getTerminator()->getNumResults() == 0);
    rewriter.eraseOp(nextPar.getBody()->getTerminator());
    rewriter.setInsertionPoint(par.getBody()->getTerminator());
    rewriter.create<affine::AffineBarrierOp>(nextPar.getLoc(),
                                             par.getBody()->getArguments());
    rewriter.inlineBlockBefore(nextPar.getBody(),
                               par.getBody()->getTerminator(),
                               par.getBody()->getArguments());
    rewriter.eraseOp(nextPar);

    return success();
  }
};

struct ParallelizeSequential
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp par,
                                PatternRewriter &rewriter) const override {
    if (gpu::affine_opt::isBlockPar(par))
      return rewriter.notifyMatchFailure(par, "op is not block par");
    Block *block = par->getBlock();
    if (block->getOperations().size() == 2)
      return rewriter.notifyMatchFailure(par, "no ops around par");
  }
};

} // namespace

namespace mlir {
void populateRemoveIVPatterns(RewritePatternSet &patterns);
}

// TODO we shuold not loop distribute non-affine ops, as we cannot reason about
// them.
//
// e.g.
//
// affine.parallel {
//   scf.for {
//     A
//     barrier
//     B
//   }
// } {gpu.par.block}
//
// In this case we should _not_ interchange and distribute as that would leave
// us with non-affine code (when considering block parallels as statements.

struct SingleRegion {
  Block::iterator begin, end;
};

struct GPUAffineOptPass : public impl::GPUAffineOptPassBase<GPUAffineOptPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    auto context = &getContext();

    auto removeIVs = [&](Operation *op) {
      RewritePatternSet patterns(context);
      populateRemoveIVPatterns(patterns);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto gpuify = [&](Operation *op) {
      SmallPtrSet<Block *, 8> blocksWithBlockPars;
      op->walk([&](Operation *op) {
        if (isBlockPar(op))
          blocksWithBlcokPars.insert(op->getBlock());
      });
      IRRewriter rewriter(context);
      for (auto block : blocksWithBlockPars) {
        SmallVector<std::variant<SingleRegion, Operation *>> regions;
        auto it = block.begin();
        auto getOneRegion = [&]() {
          if (&*it == terminator)
            return false;
          if (isBlockPar(&*it)) {
            regions.push_back(&*it);
            it++;
            return true;
          }
          SingleRegion sr;
          sr.begin = it;
          while (&*it != terminator && !isBlockPar(&*it))
            it++;
          sr.end = it;
          assert(sr.begin != sr.end);
          regions.push_back(sr);
          return true;
        };
        while (getOneRegion())
          ;

        affine::AffineParallelOp first;
        for (auto [i, opOrSingle] : llvm::enumerate(regions)) {
          bool isLast = i + 1 == regions.size();
          if (std::holds_alternative<SingleRegion>(opOrSingle)) {
            auto sr = std::get<SingleRegion>(opOrSingle);

          } else {
            auto op = std::get<Operation *>(opOrSingle);
            first = isBlockPar(op);
            assert(first);
            break;
          }
        }
        assert(first);

        rewriter.createBlock(

        for (auto [i, opOrSingle] : llvm::enumerate(regions)) {
          bool isLast = i + 1 == regions.size();
          if (std::holds_alternative<SingleRegion>(opOrSingle)) {
            auto sr = std::get<SingleRegion>(opOrSingle);

          } else {
          }
        }
      }
    };
    op->walk([&](mlir::gpu::GPUModuleOp gpuModule) {
      const mlir::DataLayoutAnalysis dl(gpuModule);
      gpuModule->walk([&](mlir::LLVM::LLVMFuncOp func) {
        if (func->getAttr("gpu.par.kernel")) {
          LLVM_DEBUG(DBGS << "Before opt:\n" << func << "\n");
          removeIVs(func);
          LLVM_DEBUG(DBGS << "Removed IVs:\n" << func << "\n");
          (void)mlir::convertLLVMToAffineAccess(func, dl, false);
          LLVM_DEBUG(DBGS << "To Affine:\n" << func << "\n");
          (void)distributeParallelLoops(func, "distribute.mincut",
                                        &getContext());
          LLVM_DEBUG(DBGS << "Distributed:\n" << func << "\n");
          PassManager pm(&getContext());
          pm.addPass(createCSEPass());
          pm.addPass(createCanonicalizerPass());
          if (failed(pm.run(func))) {
            signalPassFailure();
            return;
          }
          LLVM_DEBUG(DBGS << "Canonicalized:\n" << func << "\n");
          (void)mlir::gpu::affine_opt::optGlobalSharedMemCopies(func);
          LLVM_DEBUG(DBGS << "After opt:\n" << func << "\n");
          gpuify(func);
          //(void)mlir::gpu::affine_opt::transform(func);
          LLVM_DEBUG(DBGS << "After gpuify:\n" << func << "\n");
        }
      });
    });
  }
};

std::unique_ptr<Pass> mlir::createGPUAffineOptPass() {
  return std::make_unique<GPUAffineOptPass>();
}
