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
class ScopStmt;

class ScopArrayInfo final {
public:
  /// Return the isl id for the base pointer.
  isl::id getBasePtrId() const;

  /// Is this array allocated on heap
  ///
  /// This property is only relevant if the array is allocated by Polly instead
  /// of pre-existing. If false, it is allocated using alloca instead malloca.
  bool isOnHeap() const;

  /// Print a readable representation to @p OS.
  ///
  /// @param SizeAsPwAff Print the size as isl::pw_aff
  void print(llvm::raw_ostream &OS, bool SizeAsPwAff = false) const;

  /// Access the ScopArrayInfo associated with an access function.
  static const ScopArrayInfo *getFromAccessFunction(isl::pw_multi_aff PMA);

  /// Access the ScopArrayInfo associated with an isl Id.
  static const ScopArrayInfo *getFromId(isl::id Id);

  /// Get the space of this array access.
  isl::space getSpace() const;

  /// If the array is read only
  bool isReadOnly();

  /// Verify that @p Array is compatible to this ScopArrayInfo.
  ///
  /// Two arrays are compatible if their dimensionality, the sizes of their
  /// dimensions, and their element sizes match.
  ///
  /// @param Array The array to compare against.
  ///
  /// @returns True, if the arrays are compatible, False otherwise.
  bool isCompatibleWith(const ScopArrayInfo *Array) const;
};

class MemoryAccess {
public:
  isl::map getAccessRelation() const;
  const ScopArrayInfo *getScopArrayInfo() const;
  isl::id getId() const;
  isl::id getArrayId() const;
  bool isReductionLike() const;
  bool isRead() const;
  bool isMustWrite() const;
  bool isMayWrite() const;
  bool isWrite() const;
};

/// A wrapper for the osl_scop struct in the openscop library.
class IslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ValueTable = llvm::DenseMap<mlir::Value, std::string>;
  using MemRefToId = llvm::DenseMap<mlir::Value, std::string>;
  using ScopStmtMap = std::map<std::string, ScopStmt>;
  using ScopStmtNames = std::vector<std::string>;

  IslScop();
  ~IslScop();

  /// Simply create a new statement in the linked list scop->statement.
  ScopStmt &createStatement(mlir::Operation *op);

  /// Add the relation defined by cst to the context of the current scop.
  void addContextRelation(mlir::affine::FlatAffineValueConstraints cst);
  /// Add the domain relation.
  void addDomainRelation(ScopStmt &stmt,
                         mlir::affine::FlatAffineValueConstraints &cst);
  /// Add the access relation.
  mlir::LogicalResult
  addAccessRelation(ScopStmt &stmt, polly::MemoryAccess::AccessType type,
                    mlir::Value memref, mlir::affine::AffineValueMap &vMap,
                    mlir::affine::FlatAffineValueConstraints &cst);

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

  void buildSchedule(llvm::SmallVector<mlir::Operation *> ops) {
    loopId = 0;
    schedule = buildSequenceSchedule(ops);
  }

  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Operation *begin, mlir::Operation *end);
  static llvm::SmallVector<mlir::Operation *>
  getSequenceScheduleOpList(mlir::Block *block);

  isl_schedule *getSchedule() { return schedule; }

  mlir::Operation *applySchedule(__isl_take isl_schedule *newSchedule,
                                 mlir::Operation *f);

  isl::space getParamSpace() {
    return isl::manage(
        isl_union_set_get_space(isl_schedule_get_domain(schedule)));
  }

private:
  using StmtVec = std::vector<ScopStmt>;
  using iterator = StmtVec::iterator;
  StmtVec stmts;

public:
  iterator begin() { return stmts.begin(); }
  iterator end() { return stmts.end(); }

private:
  isl_schedule *schedule = nullptr;
  unsigned loopId = 0;

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

  /// The internal storage of the Scop.
  // osl_scop *scop;
  isl_ctx *ctx;

  /// Number of memrefs recorded.
  MemRefToId memRefIdMap;
  /// Symbol table for MLIR values.
  SymbolTable symbolTable;
  ValueTable valueTable;

  friend class IslMLIRBuilder;
  friend class IslScopBuilder;
};

} // namespace polymer

#endif
