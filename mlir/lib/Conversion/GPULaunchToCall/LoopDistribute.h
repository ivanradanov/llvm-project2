#ifndef MLIR_CONVERSION_LOOPDISTRIBUTE_H_
#define MLIR_CONVERSION_LOOPDISTRIBUTE_H_

#include "mlir/IR/PatternMatch.h"

mlir::LogicalResult distributeParallelLoops(mlir::Operation *op,
                                            mlir::StringRef method,
                                            mlir::MLIRContext *context);

#endif // MLIR_CONVERSION_LOOPDISTRIBUTE_H_
