//===- IslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "mlir/Conversion/Polymer/Support/ScatteringUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/isl-noexceptions.h"
#include "isl/schedule.h"
#include "isl/space.h"
#include "isl/union_set.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct isl_schedule;
struct isl_union_set;
struct isl_mat;
struct isl_ctx;
struct isl_set;
struct isl_space;
struct isl_basic_set;
struct isl_basic_map;

#define __isl_keep
#define __isl_give
#define __isl_take

namespace mlir {
namespace affine {
class AffineValueMap;
class AffineForOp;
class FlatAffineValueConstraints;
} // namespace affine
class Operation;
class Value;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace polymer {

class IslMLIRBuilder;
class IslScopBuilder;

class IslScop;
class ScopArrayInfo;

class MemoryAccess {
public:
  enum MemoryKind { MT_Array, MT_Value };

  enum AccessType {
    READ = 0x1,
    MUST_WRITE = 0x2,
    MAY_WRITE = 0x3,
    KILL = 0x4,
  };

  isl::id Id;
  isl::map AccessRelation;
  MemoryKind Kind;
  AccessType AccType;
  mlir::Value memref;

  isl::map getAccessRelation() { return AccessRelation; }

  // FIXME generate this obj
  const ScopArrayInfo *getScopArrayInfo() const { return nullptr; }
  isl::id getId() const { return Id; }
  isl::id getArrayId() const {
    return AccessRelation.get_tuple_id(isl::dim::out);
  }
  bool isReductionLike() const { return false; }
  bool isRead() const { return AccType == MemoryAccess::READ; }
  bool isMustWrite() const { return AccType == MemoryAccess::MUST_WRITE; }
  bool isMayWrite() const { return AccType == MemoryAccess::MAY_WRITE; }
  bool isWrite() const { return isMustWrite() || isMayWrite(); }
  bool isKill() const { return AccType == MemoryAccess::KILL; }
};

void makeIslCompatible(std::string &str);

class ScopStmt {
public:
  ScopStmt(mlir::Operation *op, IslScop *parent, llvm::StringRef name,
           bool validAsyncCopy = false);
  ~ScopStmt();

  ScopStmt(ScopStmt &&);
  ScopStmt(const ScopStmt &) = delete;
  ScopStmt &operator=(ScopStmt &&);
  ScopStmt &operator=(const ScopStmt &&) = delete;

  mlir::affine::FlatAffineValueConstraints *getMlirDomain();
  isl::set getDomain() const { return isl::manage(isl_set_copy(islDomain)); }
  isl::space getDomainSpace() const { return getDomain().get_space(); }
  isl::map getSchedule() const;
  IslScop *getParent() const { return parent; }

  /// Get a copy of the enclosing operations.
  void getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                       bool forOnly = false) const;
  /// Get the callee of this scop stmt.
  mlir::Operation *getOperation() const;
  /// Get the access affine::AffineValueMap of an op in the callee and the
  /// memref in the caller scope that this op is using.
  void getAccessMapAndMemRef(mlir::Operation *op,
                             mlir::affine::AffineValueMap *vMap,
                             mlir::Value *memref) const;

  using MemAccessesVector = std::vector<MemoryAccess *>;
  using iterator = MemAccessesVector::iterator;

  iterator begin() { return memoryAccesses.begin(); }
  iterator end() { return memoryAccesses.end(); }
  size_t size() { return memoryAccesses.size(); }

  std::string getName() { return name; }

  bool isValidAsyncCopy() { return validAsyncCopy; }

private:
  bool validAsyncCopy = false;

  // TODO we are leaking this currently
  MemAccessesVector memoryAccesses;

  isl_set *islDomain;

  using EnclosingOpList = llvm::SmallVector<mlir::Operation *, 8>;

  /// A helper function that builds the domain constraints of the
  /// caller, and find and insert all enclosing for/if ops to enclosingOps.
  void initializeDomainAndEnclosingOps();

  void getArgsValueMapping(mlir::IRMapping &argMap);

  /// Name of the callee, as well as the scop.stmt. It will also be the
  /// symbol in the OpenScop representation.
  std::string name;
  /// The statment operation
  mlir::Operation *op;
  /// The domain of the caller.
  mlir::affine::FlatAffineValueConstraints domain;
  /// Enclosing for/if operations for the caller.
  EnclosingOpList enclosingOps;

  IslScop *parent;

  friend IslScop;
};

class ScopArrayInfo final {
public:
  ScopArrayInfo(mlir::Value val, unsigned id) : val(val) {
    using namespace mlir;

    name = "A_";
    if (auto ba = dyn_cast<BlockArgument>(val))
      name += ba.getOwner()->getParentOp()->getName().getStringRef().str() +
              "_arg_" + std::to_string(ba.getArgNumber());
    else
      name += val.getDefiningOp()->getName().getStringRef().str() + "_res";
    name += "_" + std::to_string(id);
    makeIslCompatible(name);
  }
  mlir::Value val;
  std::string name;
  isl::space space;
};

/// A wrapper for the osl_scop struct in the openscop library.
class IslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ValueTable = llvm::DenseMap<mlir::Value, std::string>;
  using MemRefToId = llvm::DenseMap<mlir::Value, std::string>;

  IslScop(mlir::Operation *op);
  ~IslScop();

  /// Add the relation defined by cst to the context of the current scop.
  void addContextRelation(mlir::affine::FlatAffineValueConstraints cst);
  /// Add the domain relation.
  void addDomainRelation(ScopStmt &stmt,
                         mlir::affine::FlatAffineValueConstraints &cst);
  /// Add the access relation.
  mlir::LogicalResult
  addAccessRelation(ScopStmt &, polymer::MemoryAccess::AccessType, mlir::Value,
                    mlir::affine::AffineValueMap &, bool,
                    mlir::affine::FlatAffineValueConstraints &);

  /// Initialize the symbol table.
  void initializeSymbolTable(mlir::Operation *f,
                             mlir::affine::FlatAffineValueConstraints *cst);

  /// Get the symbol table object.
  /// TODO: maybe not expose the symbol table to the external world like this.
  SymbolTable *getSymbolTable();
  ValueTable *getValueTable();

  /// Get the mapping from memref Value to its id.
  MemRefToId *getMemRefIdMap();

  void dumpSchedule(llvm::raw_ostream &os);
  void dumpAccesses(llvm::raw_ostream &os);

  void buildSchedule() {
    buildSchedule(
        getSequenceScheduleOpList(&root->getRegion(0).front().front(),
                                  &root->getRegion(0).front().back()));
  }

  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Operation *begin, mlir::Operation *end);
  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Block *block);

  // FIXME assumption
  isl::set getAssumedContext() const {
    return isl::set::universe(getParamSpace());
  }
  isl::schedule getScheduleTree() const {
    return isl::manage(isl_schedule_copy(schedule));
  }
  isl::union_map getSchedule() const { return getScheduleTree().get_map(); }
  mlir::Operation *applySchedule(__isl_take isl_schedule *newSchedule,
                                 mlir::Operation *f);

  isl_ctx *getIslCtx() const { return IslCtx.get(); }
  std::shared_ptr<isl_ctx> getSharedIslCtx() { return IslCtx; }

  isl::space getParamSpace() const {
    return isl::manage(
        isl_union_set_get_space(isl_schedule_get_domain(schedule)));
  }

  void
  rescopeStatements(std::function<bool(mlir::Operation *op)> shouldRescope);

  ScopArrayInfo &getOrAddArray(mlir::Value memref) {
    auto found =
        llvm::find_if(arrays, [&](auto array) { return array.val == memref; });
    if (found != arrays.end())
      return *found;
    return arrays.emplace_back(memref, arrays.size());
  }

private:
  using StmtVec = std::list<ScopStmt>;
  using iterator = StmtVec::iterator;
  StmtVec stmts;

  using ArrayVec = std::vector<ScopArrayInfo>;
  using array_iterator = ArrayVec::iterator;
  ArrayVec arrays;

public:
  iterator begin() { return stmts.begin(); }
  iterator end() { return stmts.end(); }

  ScopStmt &getStatement(mlir::Operation *op) {
    auto found = llvm::find_if(
        *this, [&](ScopStmt &stmt) { return stmt.getOperation() == op; });
    assert(found != this->end());

    return *found;
  }

  isl::union_map getIndependence() { return independence; }

private:
  mlir::Operation *root;
  isl_schedule *schedule = nullptr;
  unsigned loopId = 0;

  isl::union_map independence;

  void buildSchedule(llvm::SmallVector<mlir::Operation *> ops) {
    loopId = 0;
    schedule = buildSequenceSchedule(ops);
  }

  template <typename T>
  __isl_give isl_schedule *buildLoopSchedule(T loopOp, unsigned depth,
                                             unsigned numDims, bool permutable);
  __isl_give isl_schedule *
  buildParallelSchedule(mlir::affine::AffineParallelOp parallelOp,
                        unsigned depth);
  __isl_give isl_schedule *buildForSchedule(mlir::affine::AffineForOp forOp,
                                            unsigned depth);
  __isl_give isl_schedule *buildLeafSchedule(mlir::Operation *);
  __isl_give isl_schedule *
  buildSequenceSchedule(llvm::SmallVector<mlir::Operation *> ops,
                        unsigned depth = 0);

  ScopStmt &getIslStmt(mlir::Operation *op);
  ScopStmt &getIslStmt(llvm::StringRef);

  __isl_give isl_space *
  setupSpace(__isl_take isl_space *space,
             mlir::affine::FlatAffineValueConstraints &cst, std::string name);

  __isl_give isl_mat *
  createConstraintRows(mlir::affine::FlatAffineValueConstraints &cst,
                       bool isEq);

  /// Create access relation constraints.
  mlir::LogicalResult createAccessRelationConstraints(
      mlir::affine::AffineValueMap &vMap,
      mlir::affine::FlatAffineValueConstraints &cst,
      mlir::affine::FlatAffineValueConstraints &domain);

  void addIndependences();

  /// The internal storage of the Scop.
  // osl_scop *scop;
  std::shared_ptr<isl_ctx> IslCtx;

  /// Number of memrefs recorded.
  MemRefToId memRefIdMap;
  /// Symbol table for MLIR values.
  SymbolTable symbolTable;
  ValueTable valueTable;

  friend class IslMLIRBuilder;
  friend class IslScopBuilder;
};

/// Build IslScop from FuncOp.
class IslScopBuilder {
public:
  IslScopBuilder() {}

  /// Build a scop from a common FuncOp.
  std::unique_ptr<IslScop> build(mlir::Operation *f);

private:
  /// Find all statements that calls a scop.stmt.
  void gatherStmts(mlir::Operation *f, mlir::IRMapping &map, IslScop &) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  void buildScopContext(mlir::Operation *f, IslScop *scop,
                        mlir::affine::FlatAffineValueConstraints &ctx) const;
};

} // namespace polymer

#endif
