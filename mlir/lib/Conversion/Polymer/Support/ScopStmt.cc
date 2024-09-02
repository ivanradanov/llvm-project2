//===- ScopStmt.cc ----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/ScopStmt.h"

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
using namespace mlir;
using namespace polymer;

namespace polymer {

class ScopStmtImpl {
public:
  using EnclosingOpList = SmallVector<Operation *, 8>;

  ScopStmtImpl(llvm::StringRef name, Operation *op) : name(name), op(op) {}

  static std::unique_ptr<ScopStmtImpl> get(Operation *op);

  /// A helper function that builds the domain constraints of the
  /// caller, and find and insert all enclosing for/if ops to enclosingOps.
  void initializeDomainAndEnclosingOps();

  void getArgsValueMapping(IRMapping &argMap);

  /// Name of the callee, as well as the scop.stmt. It will also be the
  /// symbol in the OpenScop representation.
  llvm::StringRef name;
  /// The statment operation
  mlir::Operation *op;
  /// The domain of the caller.
  affine::FlatAffineValueConstraints domain;
  /// Enclosing for/if operations for the caller.
  EnclosingOpList enclosingOps;
};

} // namespace polymer

/// Create ScopStmtImpl from only the caller/callee pair.
std::unique_ptr<ScopStmtImpl> ScopStmtImpl::get(Operation *op) {
  // We assume that the callerOp is of type mlir::func::CallOp, and the calleeOp
  // is a mlir::func::FuncOp. If not, these two cast lines will raise error.
  llvm::StringRef name =
      cast<StringAttr>(op->getAttr("polymer.stmt.name")).getValue();

  // Create the stmt instance.
  auto stmt = std::make_unique<ScopStmtImpl>(name, op);

  // Initialize the domain constraints around the caller. The enclosing ops will
  // be figured out as well in this process.
  stmt->initializeDomainAndEnclosingOps();

  return stmt;
}

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

void ScopStmtImpl::initializeDomainAndEnclosingOps() {
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

void ScopStmtImpl::getArgsValueMapping(IRMapping &argMap) {}

ScopStmt::ScopStmt(Operation *op) : impl{ScopStmtImpl::get(op)} {}

ScopStmt::~ScopStmt() = default;
ScopStmt::ScopStmt(ScopStmt &&) = default;
ScopStmt &ScopStmt::operator=(ScopStmt &&) = default;

affine::FlatAffineValueConstraints *ScopStmt::getDomain() const {
  return &(impl->domain);
}

void ScopStmt::getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                               bool forOnly) const {
  for (mlir::Operation *op : impl->enclosingOps)
    if (!forOnly || isa<mlir::affine::AffineForOp>(op))
      ops.push_back(op);
}

Operation *ScopStmt::getCallee() const { return impl->op; }
Operation *ScopStmt::getCaller() const { return impl->op; }

static mlir::Value findBlockArg(mlir::Value v) {
  mlir::Value r = v;
  while (r != nullptr) {
    if (r.isa<BlockArgument>())
      break;

    mlir::Operation *defOp = r.getDefiningOp();
    if (!defOp || defOp->getNumOperands() != 1)
      return nullptr;
    if (!isa<mlir::arith::IndexCastOp>(defOp))
      return nullptr;

    r = defOp->getOperand(0);
  }

  return r;
}

void ScopStmt::getAccessMapAndMemRef(mlir::Operation *op,
                                     mlir::affine::AffineValueMap *vMap,
                                     mlir::Value *memref) const {
  // Map from callee arguments to caller's. impl holds the callee and caller
  // instances.
  IRMapping argMap;
  impl->getArgsValueMapping(argMap);

  AffineMap map;
  SmallVector<Value, 4> indices;
  if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
    *memref = loadOp.getMemRef();
    llvm::append_range(indices, loadOp.getMapOperands());
    map = loadOp.getAffineMap();
  } else {
    assert(isa<affine::AffineWriteOpInterface>(op) &&
           "Affine read/write op expected");
    auto storeOp = cast<affine::AffineWriteOpInterface>(op);
    *memref = storeOp.getMemRef();
    llvm::append_range(indices, storeOp.getMapOperands());
    map = cast<affine::AffineWriteOpInterface>(op).getAffineMap();
  }

  affine::AffineValueMap aMap;
  aMap.reset(map, indices);

  // Replace its operands by what the caller uses.
  SmallVector<mlir::Value, 8> operands;
  for (mlir::Value operand : aMap.getOperands()) {
    mlir::Value origArg = findBlockArg(argMap.lookupOrDefault(operand));
    assert(origArg && "The original value cannot be found as a block argument "
                      "of the top function. Try -canonicalize.");
    // assert(origArg != operand &&
    //        "The found original value shouldn't be the same as the operand.");

    operands.push_back(origArg);
  }

  // Set the access affine::AffineValueMap.
  vMap->reset(aMap.getAffineMap(), operands);
  // Set the memref.
  *memref = argMap.lookupOrDefault(*memref);
}
