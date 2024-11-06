#include "LoopUndistribute.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPULaunchToCall/GPULaunchToCall.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
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

#include "DependenceInfo.h"
#include "GPULowering.h"
#include "ISLUtils.h"
#include "LoopDistribute.h"
#include "Utils.h"

using namespace mlir;

#define DEBUG_TYPE "gpu-affine-opt"
#define DBGS (llvm::dbgs() << "gpu-affine-opt: ")

namespace mlir {
#define GEN_PASS_DEF_GPUAFFINEOPTPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using polly::dumpIslObj;

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

bool isBlockPar(Operation *op) {
  if (!!op->getAttr("gpu.par.block")) {
    assert(op->getNumRegions() == 1);
    return true;
  }
  return false;
}
bool isGridPar(Operation *op) {
  if (!!op->getAttr("gpu.par.grid")) {
    assert(op->getNumRegions() == 1);
    return true;
  }
  return false;
}
affine::AffineParallelOp isAffineGridPar(Operation *op) {
  auto gridPar = dyn_cast_or_null<affine::AffineParallelOp>(op);
  if (!gridPar)
    return nullptr;
  if (gridPar->getAttr("gpu.par.grid"))
    return gridPar;
  return nullptr;
}
affine::AffineParallelOp isAffineBlockPar(Operation *op) {
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
[[maybe_unused]]
static inline unsigned unsignedFromIslSize(const isl::size &size) {
  assert(!size.is_error());
  return static_cast<unsigned>(size);
}
[[maybe_unused]]
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
  sc =
      isl_schedule_constraints_set_live_range_span(sc, isl_union_map_copy(lrs));
  // FIXME we would like to put this maximal path computation in isl and not
  // here, but the function is currently written in c++ so that doesn't work.
  lrs = get_maximal_paths(isl::manage(lrs)).release();
  sc = isl_schedule_constraints_set_live_range_maximal_span(sc, lrs);

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

isl::schedule insertGridPar(isl::schedule schedule) {
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

  isl::id gridMark = isl::id::alloc(schedule.ctx(), polymer::gridParallelMark,
                                    (void *)(uintptr_t)(unsigned)nMember);
  isl::schedule_node node = band.insert_mark(gridMark);
  return node.get_schedule();
}

struct PrepScheduleInfo {
  isl::union_set allArrays;
  isl::union_set unallocatedArrays;
  isl::union_set allocatedArrays;
  isl::union_map lrs;
};

/// Tag the @p Relation domain with @p TagId
static __isl_give isl_map *tag(__isl_take isl_map *Relation,
                               __isl_take isl_id *TagId) {
  isl_space *Space = isl_map_get_space(Relation);
  Space = isl_space_drop_dims(Space, isl_dim_out, 0,
                              isl_map_dim(Relation, isl_dim_out));
  Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
  isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
  Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
  return Relation;
}

isl::union_map tag(isl::union_map umap, isl::id id) {
  isl::union_map taggedMap =
      isl::manage(isl_union_map_empty(umap.get_space().release()));
  umap.foreach_map([&](isl::map map) {
    isl::map tagged = isl::manage(tag(map.release(), id.copy()));
    taggedMap = taggedMap.unite(tagged.to_union_map());
    return isl::stat::ok();
  });
  return taggedMap;
}

isl::schedule_node insertArrayExpansion(isl::schedule_node node,
                                        PrepScheduleInfo psi) {

  auto nChildren = unsignedFromIslSize(node.n_children());

  isl::union_map schedule = isl::manage(
      isl_schedule_node_get_prefix_and_node_schedule_union_map(node.get()));

  LLVM_DEBUG(llvm::dbgs() << "Prefix schedule "; dumpIslObj(schedule);
             dumpIslObj(node.get_prefix_schedule_relation()); dumpIslObj(node));

  isl::union_set toAllocate =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  psi.unallocatedArrays.foreach_set([&](isl::set set) {
    isl::id array = set.get_tuple_id();
    isl::union_map taggedSchedule = tag(schedule, array);
    LLVM_DEBUG(llvm::dbgs() << "Tagged prefix schedule ";
               dumpIslObj(taggedSchedule));

    isl::union_map applied = psi.lrs;
    applied = applied.apply_domain(taggedSchedule);
    applied = applied.apply_range(taggedSchedule);

    LLVM_DEBUG(llvm::dbgs() << "Applied schedule "; dumpIslObj(applied));

    isl::union_set deltas = applied.deltas();
    LLVM_DEBUG(llvm::dbgs() << "Deltas "; dumpIslObj(deltas));

    bool coincident = true;
    deltas.foreach_set([&](isl::set set) {
      unsigned size = unsignedFromIslSize(set.dim(isl::dim::set));
      bool allZero = true;
      for (unsigned i = 0; i < size; i++) {
        isl::val v = set.plain_get_val_if_fixed(isl::dim::set, i);
        allZero &= v.is_zero();
      }
      coincident &= allZero;
      return isl::stat::ok();
    });

    LLVM_DEBUG(llvm::dbgs() << "Array ";
               dumpIslObj(set.get_space().get_tuple_id(isl::dim::set));
               llvm::dbgs() << " coincidence: " << coincident << "\n");

    if (!coincident)
      toAllocate =
          toAllocate.unite(isl::set::universe(set.get_space()).to_union_set());

    return isl::stat::ok();
  });

  psi.allocatedArrays = psi.allocatedArrays.unite(toAllocate);
  psi.unallocatedArrays = psi.unallocatedArrays.subtract(toAllocate);

  LLVM_DEBUG(llvm::dbgs() << "Unallocated "; dumpIslObj(psi.unallocatedArrays));
  LLVM_DEBUG(llvm::dbgs() << "Allocated "; dumpIslObj(psi.allocatedArrays));

  isl::union_set toExpand =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  std::map<isl_id *, std::vector<unsigned>> toExpandMap;
  if (node.isa<isl::schedule_node_band>()) {
    auto band = node.as<isl::schedule_node_band>();
    unsigned members = unsignedFromIslSize(band.n_member());
    for (unsigned i = 0; i < members; i++) {
      isl_id_to_id *expansion =
          isl_schedule_node_band_member_get_array_expansion(node.get(), i);
      auto lambda1 = [&](isl::id arrayId, isl::id expansionId) -> isl_stat {
        // if (!toAllocate
        //     .intersect(isl::set::universe(toAllocate.get_space())
        //                .set_tuple_id(isl::manage_copy(arrayId))
        //                .to_union_set())
        //     .is_empty())
        unsigned factor = (unsigned)(uintptr_t)expansionId.get_user();
        // TODO put the expansion number in here
        auto set = isl::set::universe(toAllocate.get_space())
                       .set_tuple_id(arrayId)
                       .to_union_set();
        toExpand = toExpand.unite(set);
        if (i == 0)
          toExpandMap.insert({arrayId.get(), {factor}});
        else
          toExpandMap[arrayId.get()].push_back(factor);
        return isl_stat_ok;
      };
      struct Data {
        decltype(lambda1) &fn;
      } data{lambda1};
      auto lambda2 = [](isl_id *id, isl_id *target_id, void *user) -> isl_stat {
        return static_cast<struct Data *>(user)->fn(isl::manage(id),
                                                    isl::manage(target_id));
      };
      if (isl_id_to_id_foreach(expansion, lambda2, &data) < 0)
        return {};
    }
  }
  for (auto it = toExpandMap.begin(); it != toExpandMap.end();) {
    auto expand = it->second;
    bool expanded =
        !llvm::all_of(expand, [&](unsigned expand) { return expand == 1; });
    if (!expanded)
      it = toExpandMap.erase(it);
    else
      it++;
  }
  LLVM_DEBUG(llvm::dbgs() << "ToExpand "; dumpIslObj(toExpand));

  std::vector<isl::schedule_node> children;
  for (unsigned i = 0; i < (unsigned)nChildren; i++) {
    auto child = node.child(i);
    auto newChild = insertArrayExpansion(child, psi);
    node = newChild.parent();
  }

  if (!toAllocate.is_empty()) {
    // TODO add free function
    polymer::AllocateArrayMarkInfo *aam =
        new polymer::AllocateArrayMarkInfo{toAllocate, toExpandMap};
    isl::id allocateArrayMark =
        isl::id::alloc(schedule.ctx(), polymer::allocateArrayMark, (void *)aam);
    node = node.insert_mark(allocateArrayMark);
  }

  return node;
}

isl::schedule insertArrayExpansion(isl::schedule schedule,
                                   PrepScheduleInfo &psi) {
  auto root = schedule.get_root();
  return insertArrayExpansion(root, psi).get_schedule();
}

isl::schedule prepareScheduleForGPU(isl::schedule schedule,
                                    PrepScheduleInfo &psi) {
  schedule = insertGridPar(schedule);
  schedule = insertArrayExpansion(schedule, psi);
  return schedule;
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

  scop->rescopeStatements(isAffineBlockPar);

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

  // FIXME TODO this is a temporary work around to avoid using the scc
  // clustering part of the isl_scheduler. The incremental scheduling (described
  // in Scheduling for PPCG) creates clusters from the isl_sched_graph and we do
  // not properly convey the arrays and live ranges etc information when
  // creating new clusters and when we merge them together. Once we implemnet
  // that this can be turned off.
  auto r = isl_options_set_schedule_whole_component(scop->getIslCtx(), 1);
  assert(r == isl_stat_ok);

  isl_schedule *newSchedule = isl_schedule_constraints_compute_schedule(
      isl_schedule_constraints_copy(sc));
  LLVM_DEBUG({
    llvm::dbgs() << "New Schedule:\n";
    isl_schedule_dump(newSchedule);
  });

  PrepScheduleInfo psi;
  psi.unallocatedArrays = psi.allArrays =
      isl::manage(isl_schedule_constraints_get_array_size(sc)).domain();
  psi.allocatedArrays =
      isl::manage(isl_union_set_empty(psi.allArrays.get_space().release()));
  psi.lrs = isl::manage(isl_schedule_constraints_get_live_range_span(sc));
  newSchedule = prepareScheduleForGPU(isl::manage(newSchedule), psi).release();
  isl_schedule_constraints_free(sc);

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

namespace {

struct RegisterAllocaReduce : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (mt.getMemorySpaceAsInt() == loop_distribute::registerMemorySpace)
      return rewriter.notifyMatchFailure(ao,
                                         "Not register memory address space");

    SmallVector<affine::AffineLoadOp> loads;
    SmallVector<affine::AffineStoreOp> stores;

    for (auto &use : ao.getResult().getUses()) {
      Operation *op = use.getOwner();
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (loadOp.getMemRef() == ao.getMemref()) {
          loads.push_back(loadOp);
        } else {
          return rewriter.notifyMatchFailure(loadOp,
                                             "alloca not used as the memref?");
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == ao.getMemref()) {
          stores.push_back(storeOp);
        } else {
          return rewriter.notifyMatchFailure(storeOp,
                                             "alloca not used as the memref?");
        }
      } else {
        return rewriter.notifyMatchFailure(op, "non access use");
      }
    }

    affine::AffineParallelOp blockPar = nullptr;
    ao->getParentOp()->walk([&](Operation *op) {
      if (auto par = gpu::affine_opt::isAffineBlockPar(op)) {
        assert(!blockPar);
        blockPar = par;
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    assert(blockPar);

    rewriter.setInsertionPoint(&blockPar.getBody()->front());
    auto newAo = rewriter.create<memref::AllocaOp>(
        ao.getLoc(), MemRefType::get({}, mt.getElementType()));
    auto newMemref = newAo.getMemref();

    for (auto loadOp : loads) {
      rewriter.setInsertionPoint(loadOp);
      rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, newMemref,
                                                  ValueRange{});
    }
    for (auto storeOp : stores) {
      rewriter.setInsertionPoint(storeOp);
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          storeOp, storeOp.getValueToStore(), newMemref, ValueRange{});
    }

    rewriter.replaceOp(ao, newAo->getResults());

    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(mt))
      return rewriter.notifyMatchFailure(ao, "Not shared memory address space");

    auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                /* memspace */ 3);
    auto loc = ao->getLoc();

    unsigned counter = 0;
    SmallString<20> name = SymbolTable::generateSymbolName<20>(

        "shared_mem",
        [&](llvm::StringRef candidate) {
          return SymbolTable::lookupNearestSymbolFrom(
                     ao, StringAttr::get(ao.getContext(), candidate)) !=
                 nullptr;
        },
        counter);

    auto mod = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!mod)
      return rewriter.notifyMatchFailure(ao, "Could not find gpu module");

    rewriter.setInsertionPointToStart(mod.getBody());

    auto initialValue = rewriter.getUnitAttr();
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr(name),
        /* sym_visibility */ mlir::StringAttr(), mlir::TypeAttr::get(type),
        initialValue, mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
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
    MLIRContext *context = &getContext();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

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
    auto expandSubView = [&](Operation *op) {
      RewritePatternSet patterns(context);
      memref::populateExpandStridedMetadataPatterns(patterns);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto registerAllocaReduce = [&](Operation *op) {
      // TODO need to forward register stores to loads.
      // Check if the llvm mem2reg does that for us
      RewritePatternSet patterns(context);
      patterns.insert<RegisterAllocaReduce, SharedMemrefAllocaToGlobal>(
          context);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto gpuify = [&](Operation *op) {
      if (mlir::undistributeLoops(op).failed())
        return failure();
      IRRewriter rewriter(context);
      op->walk([&](affine::AffineBarrierOp barrier) {
        rewriter.setInsertionPoint(barrier);
        rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(barrier);
      });
      op->walk([&](memref::AllocaOp alloca) {});
      return success();
    };
    auto lowerAffine = [&](Operation *op) {
      assert(!op->walk([&](affine::AffineScopeOp op) {
                  return WalkResult::interrupt();
                }).wasInterrupted());
      RewritePatternSet patterns(&getContext());
      populateAffineToStdConversionPatterns(patterns);
      populateAffineToVectorConversionPatterns(patterns);
      affine::populateAffineExpandIndexOpsPatterns(patterns);
      ConversionTarget target(getContext());
      target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                             scf::SCFDialect, vector::VectorDialect>();
      target.addDynamicallyLegalDialect<affine::AffineDialect>(
          [&](Operation *op) { return isa<affine::AffineScopeOp>(op); });
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    };
    auto lowerAccesses = [&](Operation *op) {
      auto dl = dataLayoutAnalysis.getAtOrAbove(op);
      // TODO need to forward register stores to loads
      LowerToLLVMOptions options(&getContext(),
                                 dataLayoutAnalysis.getAtOrAbove(op));
      // TODO need to tweak options.indexBitwidth in some cases? consult
      // LowerGpuOpsToNVVMOpsPass
      options.useBarePtrCallConv = true;
      unsigned indexBitwidth = 64;
      if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
        options.overrideIndexBitwidth(indexBitwidth);

      // TODO do we need this?
      // options.dataLayout = llvm::DataLayout(this->dataLayout);

      LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);

      RewritePatternSet patterns(context);
      converter.addConversion([&](MemRefType type) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(type.getContext(),
                                          type.getMemorySpaceAsInt());
      });
      populateGPULoweringPatterns(patterns, converter);
      ConversionTarget target(getContext());
      target.addIllegalDialect<affine::AffineDialect>();
      target.addIllegalDialect<memref::MemRefDialect>();
      target.addIllegalDialect<vector::VectorDialect>();
      target.addLegalDialect<LLVM::LLVMDialect>();
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    };
    auto canonicalize = [&](Operation *op) {
      PassManager pm(&getContext());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(pm.run(op))) {
        signalPassFailure();
        return;
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
          canonicalize(func);
          LLVM_DEBUG(DBGS << "Canonicalized:\n" << func << "\n");
          //(void)mlir::gpu::affine_opt::optGlobalSharedMemCopies(func);
          mlir::gpu::affine_opt::transform(func);
          LLVM_DEBUG(DBGS << "After opt:\n" << func << "\n");
          (void)gpuify(func);
          LLVM_DEBUG(DBGS << "After gpuify:\n" << func << "\n");
          expandSubView(func);
          LLVM_DEBUG(DBGS << "After expand subview:\n" << func << "\n");
          // Generates global so run on module
          registerAllocaReduce(gpuModule);
          LLVM_DEBUG(DBGS << "After rar:\n" << func << "\n");
          lowerAffine(func);
          LLVM_DEBUG(DBGS << "After lower affine:\n" << func << "\n");
          // Need to handle globals so run on module
          lowerAccesses(gpuModule);
          LLVM_DEBUG(DBGS << "After lower accesses:\n" << func << "\n");
          canonicalize(func);
          LLVM_DEBUG(DBGS << "Canonicalized:\n" << func << "\n");
        }
      });
    });
  }
};

std::unique_ptr<Pass> mlir::createGPUAffineOptPass() {
  return std::make_unique<GPUAffineOptPass>();
}
