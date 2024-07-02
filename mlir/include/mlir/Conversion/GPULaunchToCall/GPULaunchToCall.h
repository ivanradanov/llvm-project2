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

#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<Pass> createOutlineGPUJitRegionsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_GPULAUNCHTOCALL_LLVMTOARITH_H
