//===- ScopStmt.cc ----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/IslScop.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace polymer;
using namespace mlir;

static BlockArgument findTopLevelBlockArgument(mlir::Value val) {
  if (val.isa<mlir::BlockArgument>())
    return val.cast<mlir::BlockArgument>();

  mlir::Operation *defOp = val.getDefiningOp();
  assert((defOp && isa<mlir::arith::IndexCastOp>(defOp)) &&
         "Only allow defOp of a parameter to be an IndexCast.");
  return findTopLevelBlockArgument(defOp->getOperand(0));
}

static void
promoteSymbolToTopLevel(mlir::Value val,
                        affine::FlatAffineValueConstraints &domain,
                        llvm::DenseMap<mlir::Value, mlir::Value> &symMap) {
  BlockArgument arg = findTopLevelBlockArgument(val);
  assert(isa<mlir::func::FuncOp>(arg.getOwner()->getParentOp()) &&
         "Found top-level argument should be a FuncOp argument.");
  // NOTE: This cannot pass since the found argument may not be of index type,
  // i.e., it will be index cast later.
  // assert(isValidSymbol(arg) &&
  //        "Found top-level argument should be a valid symbol.");

  unsigned int pos;
  auto res = domain.findVar(val, &pos);
  assert(res && "Provided value should be in the given domain");
  domain.setValue(pos, arg);

  symMap[val] = arg;
}

static void reorderSymbolsByOperandId(affine::FlatAffineValueConstraints &cst) {
  // bubble sort
  for (unsigned i = cst.getNumDimVars(); i < cst.getNumDimAndSymbolVars(); ++i)
    for (unsigned j = i + 1; j < cst.getNumDimAndSymbolVars(); ++j) {
      auto fst = cst.getValue(i).getAsOpaquePointer();
      auto snd = cst.getValue(j).getAsOpaquePointer();
      if (fst > snd)
        cst.swapVar(i, j);
    }
}

void ScopStmt::initializeDomainAndEnclosingOps() {
  // Extract the affine for/if ops enclosing the caller and insert them into the
  // enclosingOps list.
  affine::getEnclosingAffineOps(*op, &enclosingOps);

  // The domain constraints can then be collected from the enclosing ops.
  auto res = succeeded(getIndexSet(enclosingOps, &domain));
  assert(res);

  // Symbol values, which could be a BlockArgument, or the result of DimOp or
  // IndexCastOp, or even an affine.apply. Here we limit the cases to be either
  // BlockArgument or IndexCastOp, and if it is an IndexCastOp, the cast source
  // should be a top-level BlockArgument.
  SmallVector<mlir::Value, 8> symValues;
  llvm::DenseMap<mlir::Value, mlir::Value> symMap;
  domain.getValues(domain.getNumDimVars(), domain.getNumDimAndSymbolVars(),
                   &symValues);

  // Without this things like swapped-bounds.mlir in test cannot work.
  reorderSymbolsByOperandId(domain);
}

void ScopStmt::getArgsValueMapping(IRMapping &argMap) {}

ScopStmt::ScopStmt(Operation *op, IslScop *parent) : op(op), parent(parent) {
  name = cast<StringAttr>(op->getAttr("polymer.stmt.name")).getValue();
  initializeDomainAndEnclosingOps();
}

ScopStmt::~ScopStmt() {
  for (auto *MA : memoryAccesses) {
    islDomain = isl_set_free(islDomain);
    delete MA;
  }
}
ScopStmt::ScopStmt(ScopStmt &&) = default;
ScopStmt &ScopStmt::operator=(ScopStmt &&) = default;

affine::FlatAffineValueConstraints *ScopStmt::getMlirDomain() {
  return &domain;
}

isl::map ScopStmt::getSchedule() const {
  isl::set Domain = getDomain();
  if (Domain.is_empty())
    return isl::map::from_aff(isl::aff(isl::local_space(getDomainSpace())));
  auto Schedule = getParent()->getSchedule();
  if (Schedule.is_null())
    return {};
  Schedule = Schedule.intersect_domain(isl::union_set(Domain));
  if (Schedule.is_empty())
    return isl::map::from_aff(isl::aff(isl::local_space(getDomainSpace())));
  isl::map M = M.from_union_map(Schedule);
  M = M.coalesce();
  M = M.gist_domain(Domain);
  M = M.coalesce();
  return M;
}
void ScopStmt::getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                               bool forOnly) const {
  for (mlir::Operation *op : enclosingOps)
    if (!forOnly || isa<mlir::affine::AffineForOp>(op))
      ops.push_back(op);
}

Operation *ScopStmt::getOperation() const { return op; }
