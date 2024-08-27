#ifndef MLIR_CONVERSION_LOOPDISTRIBUTE_H_
#define MLIR_CONVERSION_LOOPDISTRIBUTE_H_

#include "mlir/IR/PatternMatch.h"

void addDistributePatterns(mlir::RewritePatternSet &patterns, mlir::StringRef method);

#endif // MLIR_CONVERSION_LOOPDISTRIBUTE_H_
