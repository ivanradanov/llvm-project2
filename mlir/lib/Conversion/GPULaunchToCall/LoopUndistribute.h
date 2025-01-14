#ifndef LOOPUNDISTRIBUTE_H_
#define LOOPUNDISTRIBUTE_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {

LogicalResult undistributeLoops(Operation *func);

namespace gpu {
namespace affine_opt {
bool isGridPar(Operation *op);
bool isBlockPar(Operation *op);
affine::AffineParallelOp isAffineGridPar(Operation *op);
affine::AffineParallelOp isAffineBlockPar(Operation *op);
} // namespace affine_opt
} // namespace gpu

} // namespace mlir

#endif // LOOPUNDISTRIBUTE_H_
