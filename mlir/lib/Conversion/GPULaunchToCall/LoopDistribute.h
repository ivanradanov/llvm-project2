#ifndef MLIR_CONVERSION_LOOPDISTRIBUTE_H_
#define MLIR_CONVERSION_LOOPDISTRIBUTE_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace loop_distribute {
static constexpr int64_t registerMemorySpace = 15;
static constexpr int64_t crossingRegisterMemorySpace = 16;
} // namespace loop_distribute

LogicalResult distributeParallelLoops(mlir::Operation *op,
                                      mlir::StringRef method,
                                      mlir::MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_LOOPDISTRIBUTE_H_
