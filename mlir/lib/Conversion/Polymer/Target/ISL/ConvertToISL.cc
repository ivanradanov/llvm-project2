//===- EmitOpenScop.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Conversion/Polymer/Transforms/ExtractScopStmt.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "polly/DependenceInfo.h"
#include "polly/ScopInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/schedule.h"

#include <memory>

using namespace mlir;
using namespace mlir::func;
using namespace polymer;
using llvm::dbgs;
using llvm::errs;
using llvm::outs;

#define DEBUG_TYPE "islscop"

namespace mlir {
namespace gpu {
namespace affine_opt {
affine::AffineParallelOp isBlockPar(Operation *op);
}
} // namespace gpu
} // namespace mlir

namespace polymer {

/// Build IslScop from FuncOp.
class IslScopBuilder {
public:
  IslScopBuilder() {}

  /// Build a scop from a common FuncOp.
  std::unique_ptr<IslScop> build(Operation *f);

private:
  /// Find all statements that calls a scop.stmt.
  void gatherStmts(Operation *f, IRMapping &map, IslScop::StmtVec &) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  void buildScopContext(Operation *f, IslScop *scop,
                        affine::FlatAffineValueConstraints &ctx) const;
};

} // namespace polymer

/// Sometimes the domain generated might be malformed. It is always better to
/// inform this at an early stage.
static void sanityCheckDomain(affine::FlatAffineValueConstraints &dom) {
  if (dom.isEmpty()) {
    llvm::errs() << "A domain is found to be empty!";
    dom.dump();
  }
}

/// Build IslScop from a given FuncOp.
std::unique_ptr<IslScop> IslScopBuilder::build(Operation *f) {

  /// Context constraints.
  affine::FlatAffineValueConstraints ctx;

  // Initialize a new Scop per FuncOp. The osl_scop object within it will be
  // created. It doesn't contain any fields, and this may incur some problems,
  // which the validate function won't discover, e.g., no context will cause
  // segfault when printing scop. Please don't just return this object.
  auto scop = std::make_unique<IslScop>();

  // Find all caller/callee pairs in which the callee has the attribute of name
  // SCOP_STMT_ATTR_NAME.
  IRMapping storeMap;
  gatherStmts(f, storeMap, scop->stmts);

  // Build context in it.
  buildScopContext(f, scop.get(), ctx);

  scop->initializeSymbolTable(f, &ctx);

  // Counter for the statement inserted.
  unsigned stmtId = 0;
  for (auto &stmt : scop->stmts) {
    LLVM_DEBUG({
      dbgs() << "Adding relations to statement: \n";
      stmt.getOperation()->dump();
    });

    // Collet the domain
    affine::FlatAffineValueConstraints domain = *stmt.getDomain();
    sanityCheckDomain(domain);

    LLVM_DEBUG({
      dbgs() << "Domain:\n";
      domain.dump();
    });

    // Collect the enclosing ops.
    llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
    stmt.getEnclosingOps(enclosingOps);
    // Get the callee.
    Operation *op = stmt.getOperation();

    LLVM_DEBUG({
      dbgs() << "op:\n";
      op->dump();
    });

    scop->addDomainRelation(stmt, domain);
    {
      LLVM_DEBUG(dbgs() << "Creating access relation for: " << *op << '\n');
      auto addMayStore = [&](Value memref, affine::AffineValueMap map) {
        (void)scop->addAccessRelation(stmt, polly::MemoryAccess::MAY_WRITE,
                                      storeMap.lookupOrDefault(memref), map,
                                      domain);
      };
      auto addMustStore = [&](Value memref, affine::AffineValueMap map) {
        (void)scop->addAccessRelation(stmt, polly::MemoryAccess::MUST_WRITE,
                                      storeMap.lookupOrDefault(memref), map,
                                      domain);
      };
      auto addLoad = [&](Value memref, affine::AffineValueMap map) {
        (void)scop->addAccessRelation(stmt, polly::MemoryAccess::READ,
                                      storeMap.lookupOrDefault(memref), map,
                                      domain);
      };
      bool needToLoadOperands = true;
      bool needToStoreResults = true;
      auto unitMap = AffineMap::get(op->getContext());
      affine::AffineValueMap unitVMap(unitMap, ValueRange{}, ValueRange{});
      if (!isMemoryEffectFree(op)) {
        if (isa<mlir::affine::AffineReadOpInterface>(op) ||
            isa<mlir::affine::AffineWriteOpInterface>(op)) {

          affine::AffineValueMap vMap;
          mlir::Value memref;

          AffineMap map;
          SmallVector<Value, 4> indices;
          if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
            memref = loadOp.getMemRef();
            llvm::append_range(indices, loadOp.getMapOperands());
            map = loadOp.getAffineMap();
            vMap.reset(map, indices);
            addLoad(memref, vMap);
            addMustStore(loadOp.getValue(), unitVMap);
          } else {
            assert(isa<affine::AffineWriteOpInterface>(op) &&
                   "Affine read/write op expected");
            auto storeOp = cast<affine::AffineWriteOpInterface>(op);
            memref = storeOp.getMemRef();
            llvm::append_range(indices, storeOp.getMapOperands());
            map = cast<affine::AffineWriteOpInterface>(op).getAffineMap();
            vMap.reset(map, indices);
            addMustStore(memref, vMap);
            addLoad(storeOp.getValueToStore(), unitVMap);
          }
          needToLoadOperands = false;
          needToStoreResults = false;
        } else {
          assert((isa<memref::AllocOp, memref::AllocaOp>(op)));
          needToLoadOperands = false;
          needToStoreResults = false;
        }
      } else if (auto storeVar = dyn_cast<affine::AffineStoreVar>(op)) {
        assert(storeVar->getNumOperands() == 2);
        Value val = storeMap.lookupOrDefault(storeVar->getOperand(0));
        Value addr = storeMap.lookupOrDefault(storeVar->getOperand(1));
        addLoad(val, unitVMap);
        addMustStore(addr, unitVMap);
        needToLoadOperands = false;
        needToStoreResults = false;
      } else if (auto yield = dyn_cast<affine::AffineYieldOp>(op)) {
        for (auto [res, opr] :
             llvm::zip(ValueRange(yield->getParentOp()->getResults()),
                       ValueRange(yield->getOperands()))) {
          addMustStore(res, unitVMap);
          addLoad(opr, unitVMap);
        }
        needToLoadOperands = false;
        needToStoreResults = false;
      }
      if (needToStoreResults)
        for (auto res : op->getResults())
          addMustStore(res, unitVMap);
      if (needToLoadOperands)
        for (auto opr : op->getOperands())
          addLoad(opr, unitVMap);
    }

    stmtId++;
  }

  scop->buildSchedule(scop->getSequenceScheduleOpList(
      &f->getRegion(0).front().front(), &f->getRegion(0).front().back()));

  return scop;
}

static void createForIterArgAccesses(affine::AffineForOp forOp,
                                     IRMapping &map) {
  OpBuilder builder(forOp);
  for (auto [ba, res] : llvm::zip(ValueRange(forOp.getInitsMutable()),
                                  ValueRange(forOp.getResults())))
    builder.create<affine::AffineStoreVar>(
        forOp.getLoc(), ValueRange{ba, res},
        builder.getStringAttr("for.iv.init"));
  map.map(forOp.getRegionIterArgs(), forOp.getResults());
}

void IslScopBuilder::gatherStmts(Operation *f, IRMapping &map,
                                 IslScop::StmtVec &stmts) const {
  f->walk(
      [&](affine::AffineForOp forOp) { createForIterArgAccesses(forOp, map); });
  unsigned stmtId = 0;
  f->walk([&](mlir::Operation *op) {
    if (isa<affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp>(
            op))
      return;
    if (op == f)
      return;
    std::string calleeName = "S" + std::to_string(stmtId++) + "." +
                             op->getName().getStringRef().str();
    op->setAttr("polymer.stmt.name",
                StringAttr::get(f->getContext(), calleeName));
    stmts.push_back(ScopStmt(op));
  });
}

void IslScopBuilder::buildScopContext(
    Operation *f, IslScop *scop,
    affine::FlatAffineValueConstraints &ctx) const {
  LLVM_DEBUG(dbgs() << "--- Building SCoP context ...\n");

  // First initialize the symbols of the ctx by the order of arg number.
  // This simply aims to make mergeAndAlignVarsWithOthers work.
  SmallVector<Value> symbols;
  auto insertSyms = [&](auto syms) {
    for (Value sym : syms) {
      // Find the insertion position.
      auto it = symbols.begin();
      while (it != symbols.end()) {
        auto lhs = it->getAsOpaquePointer();
        auto rhs = sym.getAsOpaquePointer();
        if (lhs >= rhs)
          break;
        ++it;
      }
      if (it == symbols.end() || *it != sym)
        symbols.insert(it, sym);
    }
  };
  for (auto &stmt : scop->stmts) {
    auto domain = stmt.getDomain();
    SmallVector<Value> syms;
    domain->getValues(domain->getNumDimVars(), domain->getNumDimAndSymbolVars(),
                      &syms);

    insertSyms(syms);
  }
  f->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      insertSyms(loadOp.getMapOperands().drop_front(
          loadOp.getAffineMap().getNumDims()));
    } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      insertSyms(storeOp.getMapOperands().drop_front(
          storeOp.getAffineMap().getNumDims()));
    }
  });

  ctx = affine::FlatAffineValueConstraints(/*numDims=*/0,
                                           /*numSymbols=*/symbols.size());
  ctx.setValues(0, symbols.size(), symbols);

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (auto &stmt : scop->stmts) {
    affine::FlatAffineValueConstraints *domain = stmt.getDomain();
    affine::FlatAffineValueConstraints cst(*domain);

    LLVM_DEBUG(dbgs() << "Statement:\n");
    LLVM_DEBUG(stmt.getOperation()->dump());
    LLVM_DEBUG(dbgs() << "Target domain: \n");
    LLVM_DEBUG(domain->dump());

    LLVM_DEBUG({
      dbgs() << "Domain values: \n";
      SmallVector<Value> values;
      domain->getValues(0, domain->getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });

    ctx.mergeAndAlignVarsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();

    LLVM_DEBUG(dbgs() << "Updated context: \n");
    LLVM_DEBUG(ctx.dump());

    LLVM_DEBUG({
      dbgs() << "Context values: \n";
      SmallVector<Value> values;
      ctx.getValues(0, ctx.getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });
  }

  // Then, create the single context relation in scop.
  scop->addContextRelation(ctx);

  // Finally, given that ctx has all the parameters in it, we will make sure
  // that each domain is aligned with them, i.e., every domain has the same
  // parameter columns (Values & order).
  SmallVector<mlir::Value, 8> symValues;
  ctx.getValues(ctx.getNumDimVars(), ctx.getNumDimAndSymbolVars(), &symValues);

  // Add and align domain SYMBOL columns.
  for (auto &stmt : scop->stmts) {
    affine::FlatAffineValueConstraints *domain = stmt.getDomain();
    // For any symbol missing in the domain, add them directly to the end.
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); ++i) {
      unsigned pos;
      if (!domain->findVar(symValues[i], &pos)) // insert to the back
        domain->appendSymbolVar(symValues[i]);
      else
        LLVM_DEBUG(dbgs() << "Found " << symValues[i] << '\n');
    }

    // Then do the aligning.
    LLVM_DEBUG(domain->dump());
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
      mlir::Value sym = symValues[i];
      unsigned pos;
      domain->findVar(sym, &pos);

      unsigned posAsCtx = i + domain->getNumDimVars();
      LLVM_DEBUG(dbgs() << "Swapping " << posAsCtx << " " << pos << "\n");
      if (pos != posAsCtx)
        domain->swapVar(posAsCtx, pos);
    }

    // for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
    //   mlir::Value sym = symValues[i];
    //   unsigned pos;
    //   // If the symbol can be found in the domain, we put it in the same
    //   // position as the ctx.
    //   if (domain->findVar(sym, &pos)) {
    //     if (pos != i + domain->getNumDimVars())
    //       domain->swapVar(i + domain->getNumDimVars(), pos);
    //   } else {
    //     domain->insertSymbolId(i, sym);
    //   }
    // }
  }
}

std::unique_ptr<IslScop> polymer::createIslFromFuncOp(Operation *f) {
  return IslScopBuilder().build(f);
}
