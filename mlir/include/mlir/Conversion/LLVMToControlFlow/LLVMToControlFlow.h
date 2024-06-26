//===- LLVMToControlFlow.h - ControlFlow to LLVM -----------*- C++ ------*-===//
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

#ifndef MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H
#define MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTLLVMTOCONTROLFLOWPASS
#include "mlir/Conversion/Passes.h.inc"

namespace cf {

void populateLLVMToControlFlowConversionPatterns(RewritePatternSet &patterns);

} // namespace cf

std::unique_ptr<Pass> createConvertLLVMToControlFlowPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMTOCONTROLFLOW_LLVMTOCONTROLFLOW_H
