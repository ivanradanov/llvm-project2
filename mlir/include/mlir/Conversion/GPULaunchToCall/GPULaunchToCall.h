//===- GPULaunchToCall.h - GPU Launch to Call -----------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPULAUNCHTOCALL_LLVMTOARITH_H
#define MLIR_CONVERSION_GPULAUNCHTOCALL_LLVMTOARITH_H

#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {

class Pass;
class Operation;
class DataLayoutAnalysis;

std::unique_ptr<Pass> createOutlineGPUJitRegionsPass();
std::unique_ptr<Pass> createPromoteScfWhilePass();
std::unique_ptr<Pass> createLLVMToAffineAccessPass();
std::unique_ptr<Pass> createGPULaunchToParallelPass();
std::unique_ptr<Pass> createReshapeMemrefsPass();
std::unique_ptr<Pass> createGPUAffineOptPass();

LogicalResult
convertLLVMToAffineAccess(mlir::Operation *op,
                          const mlir::DataLayoutAnalysis &dataLayoutAnalysis,
                          bool legalizeSymbols);

namespace gpu {
namespace affine_opt {
void optGlobalSharedMemCopies(Operation *root);
}
} // namespace gpu

} // namespace mlir

#endif // MLIR_CONVERSION_GPULAUNCHTOCALL_LLVMTOARITH_H
