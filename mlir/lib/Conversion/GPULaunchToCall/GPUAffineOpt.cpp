#include "DependenceInfo.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "polly/Support/GICHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/constraint.h"
#include "isl/id.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

#include "ISLUtils.h"
#include "LoopDistribute.h"

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

affine::AffineParallelOp isGridPar(Operation *op) {
  auto gridPar = dyn_cast_or_null<affine::AffineParallelOp>(op);
  if (!gridPar)
    return nullptr;
  if (gridPar->getAttr("gpu.par.grid"))
    return gridPar;
  return nullptr;
}
affine::AffineParallelOp isBlockPar(Operation *op) {
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

static inline void islAssert(const isl_size &size) {
  assert(size != isl_size_error);
}
static inline unsigned unsignedFromIslSize(const isl::size &size) {
  assert(!size.is_error());
  return static_cast<unsigned>(size);
}
static inline unsigned unsignedFromIslSize(const isl_size &size) {
  islAssert(size);
  return static_cast<unsigned>(size);
}

static __isl_give isl_schedule_constraints *
construct_schedule_constraints(struct ppcg_scop *scop, polymer::Scop &S) {
  isl_union_set *domain;
  isl_union_map *dep_raw, *dep;
  isl_union_map *validity, *proximity, *coincidence, *anti_proximity;
  isl_schedule_constraints *sc;

  domain = isl_union_set_copy(scop->domain);
  sc = isl_schedule_constraints_on_domain(domain);
  sc = isl_schedule_constraints_set_context(sc, isl_set_copy(scop->context));
  if (scop->options->live_range_reordering) {
    sc = isl_schedule_constraints_set_conditional_validity(
        sc, isl_union_map_copy(scop->tagged_dep_flow),
        isl_union_map_copy(scop->tagged_dep_order));
    proximity = isl_union_map_copy(scop->dep_flow);
    validity = isl_union_map_copy(proximity);
    validity =
        isl_union_map_union(validity, isl_union_map_copy(scop->dep_forced));
// According to "Scheduling for PPCG": "Note that as explained in Section
// 4.3 below, the false dependence relation is only used for historical
// reasons." Try disabling for now.
#if 0
    proximity =
        isl_union_map_union(proximity, isl_union_map_copy(scop->dep_false));
#endif
    coincidence = isl_union_map_copy(validity);
    coincidence = isl_union_map_subtract(
        coincidence, isl_union_map_copy(scop->independence));
// TODO this is introducing unwanted dependencies, like these, dont know
// why, need to investigate
//   "[P0] -> { RS1[i0, 0, 0] -> RS0[o0, 0, 0] : i0 >= 0 and i0 < o0 < P0;
//   RS0_affine_parallel[i0, 0, 0] -> RS1_affine_parallel[i0, 0, 0] : 0 <=
//   i0 < P0 }"
// Disable for now.
#if 0
    coincidence =
        isl_union_map_union(coincidence, isl_union_map_copy(scop->array_order));
#endif
  } else {
    dep_raw = isl_union_map_copy(scop->dep_flow);
    dep = isl_union_map_copy(scop->dep_false);
    dep = isl_union_map_union(dep, dep_raw);
    dep = isl_union_map_coalesce(dep);
    proximity = isl_union_map_copy(dep);
    coincidence = isl_union_map_copy(dep);
    validity = dep;
  }
  anti_proximity = isl_union_map_copy(scop->dep_async);
  sc = isl_schedule_constraints_set_validity(sc, validity);
  sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
  sc = isl_schedule_constraints_set_proximity(sc, proximity);
  sc = isl_schedule_constraints_set_anti_proximity(sc, anti_proximity);
  isl_union_map *lrs = isl_union_map_copy(scop->atagged_dep_flow);
  lrs = get_maximal_paths(isl::manage(lrs)).release();
  sc = isl_schedule_constraints_set_live_range_span(sc, lrs);

  isl_union_set *arrays = isl_union_map_range(isl_union_set_unwrap(
      isl_union_map_domain(isl_union_map_copy(scop->atagged_dep_flow))));
  arrays = isl_union_set_universe(arrays);
  isl::union_map arraySizes =
      isl::manage(isl_union_map_empty(isl_union_set_get_space(arrays)));

  auto r = isl::manage(arrays).foreach_set([&](isl::set set) -> isl::stat {
    Value v = Value::getFromOpaquePointer(set.get_tuple_id().get_user());
    auto *sai = S.getArray(v);
    assert(sai);
    auto size = sai->getSize();
    if (!size)
      return isl::stat::error();

    auto space = set.get_space();
    space = space.add_dims(isl::dim::out, 1);
    auto sizeSet = isl::set::universe(isl::space(set.ctx(), 0, 1));
    auto cst =
        isl::constraint::alloc_equality(isl::local_space(sizeSet.get_space()));
    cst = cst.set_constant_si(-*size);
    cst = cst.set_coefficient_si(isl::dim::out, 0, 1);
    sizeSet = sizeSet.add_constraint(cst);
    isl::map map = isl::map::from_domain_and_range(set, sizeSet);
    arraySizes = arraySizes.unite(isl::union_map(map));

    return isl::stat::ok();
  });

  if (r.is_ok()) {
    // TODO get this from the gpu module target info
    // 48kB is it KiB or KB???
    const int maxShmemPerBlock = 48 * 1000;
    sc = isl_schedule_constraints_set_caches(sc, 1, &maxShmemPerBlock);
    sc = isl_schedule_constraints_set_array_sizes(sc, arraySizes.release());
  }

  return sc;
}

isl::schedule prepareScheduleForGPU(isl::schedule schedule) {
  auto root = schedule.get_root();

  auto nChildren = root.n_children();
  if (nChildren.is_error())
    return {};
  if ((unsigned)nChildren != 1)
    return {};

  auto child = root.child(0);

  if (!child.isa<isl::schedule_node_band>())
    return {};
  auto band = child.as<isl::schedule_node_band>();

  // TODO need to split
  auto nMember = band.n_member();
  assert(!nMember.is_error());
  if ((unsigned)nMember > 3)
    return {};

  bool allCoincident = true;
  for (unsigned i = 0; i < (unsigned)nMember; i++)
    allCoincident = allCoincident && band.member_get_coincident(i);

  if (!allCoincident)
    return {};

  // TODO make this extensible
  isl::id gridMark = isl::id::alloc(schedule.ctx(), polymer::gridParallelMark,
                                    (void *)(unsigned)nMember);
  isl::schedule_node node = band.insert_mark(gridMark);
  return node.get_schedule();
}

void transform(LLVM::LLVMFuncOp f) {
  using namespace polymer;
  std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(f);
  scop->buildSchedule();
  LLVM_DEBUG({
    llvm::dbgs() << "Schedule:\n";
    isl_schedule_dump(scop->getScheduleTree().get());
    llvm::dbgs() << "Accesses:\n";
    scop->dumpAccesses(llvm::dbgs());
  });

  scop->rescopeStatements(isBlockPar);

  scop->buildSchedule();
  LLVM_DEBUG({
    llvm::dbgs() << "Schedule:\n";
    isl_schedule_dump(scop->getScheduleTree().get());
    llvm::dbgs() << "Accesses:\n";
    scop->dumpAccesses(llvm::dbgs());
  });

  ppcg_scop *ps = computeDeps(*scop);
  isl_schedule_constraints *sc = construct_schedule_constraints(ps, *scop);

  LLVM_DEBUG({
    llvm::dbgs() << "Schedule constraints:\n";
    isl_schedule_constraints_dump(sc);
  });

  isl_schedule *newSchedule = isl_schedule_constraints_compute_schedule(sc);
  LLVM_DEBUG({
    llvm::dbgs() << "New Schedule:\n";
    isl_schedule_dump(newSchedule);
  });

  newSchedule = prepareScheduleForGPU(isl::manage(newSchedule)).release();

  LLVM_DEBUG({
    llvm::dbgs() << "New Schedule Prepared for GPU:\n";
    isl_schedule_dump(newSchedule);
  });

  isl_ast_build *build = isl_ast_build_alloc(scop->getIslCtx());
  isl_ast_node *node =
      isl_ast_build_node_from_schedule(build, isl_schedule_copy(newSchedule));
  LLVM_DEBUG({
    llvm::dbgs() << "New AST:\n";
    isl_ast_node_dump(node);
  });

  auto g = cast<LLVM::LLVMFuncOp>(scop->applySchedule(newSchedule, f));

  scop->cleanup(g);

  if (g) {
    for (auto &b : llvm::make_early_inc_range(f.getRegion().getBlocks()))
      b.erase();

    f.getRegion().getBlocks().splice(f.getRegion().getBlocks().begin(),
                                     g.getRegion().getBlocks());
    g->erase();

    LLVM_DEBUG({
      llvm::dbgs() << "New func:\n";
      llvm::dbgs() << *f << "\n";
    });
  }

  isl_ast_build_free(build);
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

struct Interchange : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp parOp,
                                PatternRewriter &rewriter) const override {
    if (!gpu::affine_opt::isBlockPar(parOp))
      return rewriter.notifyMatchFailure(parOp, "op is not block par");
    Operation *parent = parOp->getParentOp();
    if (gpu::affine_opt::isGridPar(parent))
      return rewriter.notifyMatchFailure(parent, "Parent op is grid par");
    if (parOp->getBlock()->getOperations().size() != 2)
      return rewriter.notifyMatchFailure(parOp, "imperfectly nested");
    auto forOp = dyn_cast<affine::AffineForOp>(parent);
    if (!forOp)
      return rewriter.notifyMatchFailure(parent,
                                         "Parent op is not affine for op");

    auto loc = parOp->getLoc();

    // affine.for
    //   affine.parallel
    //
    // to
    //
    // affine.parallel
    //   affine.for
    rewriter.setInsertionPoint(forOp);
    auto newPar =
        cast<affine::AffineParallelOp>(rewriter.cloneWithoutRegions(*parOp));
    rewriter.createBlock(&newPar.getRegion(), newPar.getRegion().begin(),
                         parOp.getBody()->getArgumentTypes(),
                         parOp.getBody()->getArgumentLocs());
    auto newFor =
        cast<affine::AffineForOp>(rewriter.cloneWithoutRegions(*forOp));
    rewriter.createBlock(&newFor.getRegion(), newFor.getRegion().begin(),
                         forOp.getBody()->getArgumentTypes(),
                         forOp.getBody()->getArgumentLocs());
    rewriter.inlineBlockBefore(parOp.getBody(), newFor.getBody(),
                               newFor.getBody()->begin(),
                               newPar.getBody()->getArguments());
    assert(newFor.getBody()->getTerminator()->getNumResults() == 0);
    rewriter.eraseOp(newFor.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(newFor.getBody());
    rewriter.create<affine::AffineYieldOp>(loc);
    rewriter.setInsertionPointToEnd(newPar.getBody());
    rewriter.create<affine::AffineYieldOp>(loc);

    for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                           newFor.getBody()->getArguments()))
      rewriter.replaceAllUsesWith(oldArg, newArg);

    rewriter.eraseOp(forOp);

    return success();
  }
};

struct FuseBlockPars : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineParallelOp par,
                                PatternRewriter &rewriter) const override {
    if (!gpu::affine_opt::isBlockPar(par))
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
    return rewriter.notifyMatchFailure(par, "not implemented yet");
  }
};

struct AffineVectorStoreLower
    : public OpRewritePattern<affine::AffineVectorStoreOp> {
  using OpRewritePattern<affine::AffineVectorStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineVectorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = cast_or_null<TypeAttr>(op->getAttr("polymer.access.type")).getValue();
    if (!ty)
      return rewriter.notifyMatchFailure(op, "Access type attribute missing.");
    return failure();
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
      // TODO need to forward register stores to loads
      RewritePatternSet patterns(context);
      patterns.insert<FuseBlockPars, Interchange>(context);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto lowerAffine = [&](Operation *op) {
      RewritePatternSet patterns(&getContext());
      populateAffineToStdConversionPatterns(patterns);
      populateAffineToVectorConversionPatterns(patterns);
      affine::populateAffineExpandIndexOpsPatterns(patterns);
      ConversionTarget target(getContext());
      target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                             scf::SCFDialect, vector::VectorDialect>();
      target.addDynamicallyLegalDialect<affine::AffineDialect>(
          [&](Operation *op) { return isa<affine::AffineScopeOp>(op); });
      if (failed(applyPartialConversion(op, target,
                                        std::move(patterns)))) {
        signalPassFailure();
        return;
      }
      // ScopeOp's needs to be preserved untill all other affine operations are
      // lowered as their lowerings depend on the existence of the scope
      op->walk([&](affine::AffineScopeOp op) {
        IRRewriter rewriter(op);
        Block *body = op.getBody();
        Operation *terminator = body->getTerminator();
        rewriter.inlineBlockBefore(body, op, op->getOperands());
        rewriter.replaceOp(op, terminator->getOperands());
        rewriter.eraseOp(terminator);
      });
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
          //(void)mlir::gpu::affine_opt::optGlobalSharedMemCopies(func);
          mlir::gpu::affine_opt::transform(func);
          LLVM_DEBUG(DBGS << "After opt:\n" << func << "\n");
          lowerAffine(func);
          LLVM_DEBUG(DBGS << "After lower affine:\n" << func << "\n");
          gpuify(func);
          LLVM_DEBUG(DBGS << "After gpuify:\n" << func << "\n");

        }
      });
    });
  }
};

std::unique_ptr<Pass> mlir::createGPUAffineOptPass() {
  return std::make_unique<GPUAffineOptPass>();
}
