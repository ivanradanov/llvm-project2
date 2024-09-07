//===- ScopStmt.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_SCOPSTMT_H
#define POLYMER_SUPPORT_SCOPSTMT_H

#include <memory>

#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include "isl/isl-noexceptions.h"
#include "isl/schedule.h"
#include "isl/space.h"
#include "isl/union_set.h"

namespace mlir {
class Operation;
namespace affine {
class FlatAffineValueConstraints;
class AffineValueMap;
} // namespace affine
namespace func {
class FuncOp;
class CallOp;
} // namespace func
class Value;
} // namespace mlir

namespace polymer {
class ScopStmtImpl;

class MemoryAccess {
public:
  enum MemoryKind { MT_Array, MT_Value };

  enum AccessType {
    READ = 0x1,
    MUST_WRITE = 0x2,
    MAY_WRITE = 0x3,
  };

  isl::id Id;
  isl::map AccessRelation;
  MemoryKind Kind;
  AccessType AccType;

  isl::map getAccessRelation() { return AccessRelation; }

  const ScopArrayInfo *getScopArrayInfo() const;
  isl::id getId() const { return Id; }
  isl::id getArrayId() const {
    return AccessRelation.get_tuple_id(isl::dim::out);
  }
  bool isReductionLike() const { return false; }
  bool isRead() const { return AccType == MemoryAccess::READ; }
  bool isMustWrite() const { return AccType == MemoryAccess::MUST_WRITE; }
  bool isMayWrite() const { return AccType == MemoryAccess::MAY_WRITE; }
  bool isWrite() const { return isMustWrite() || isMayWrite(); }
};

class ScopStmt {
public:
  ScopStmt(mlir::Operation *op);
  ~ScopStmt();

  ScopStmt(ScopStmt &&);
  ScopStmt(const ScopStmt &) = delete;
  ScopStmt &operator=(ScopStmt &&);
  ScopStmt &operator=(const ScopStmt &&) = delete;

  mlir::affine::FlatAffineValueConstraints *getMlirDomain();
  isl::set getDomain() { return isl::manage(isl_set_copy(islDomain)); }

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

private:
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

  friend IslScop;
};
} // namespace polymer

#endif
