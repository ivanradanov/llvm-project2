//===- Llvmtoarith.h - ControlFlow to LLVM -----------*- C++ ------*-===//
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

#ifndef MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H
#define MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTLLVMTOARITHPASS
#include "mlir/Conversion/Passes.h.inc"

namespace arith {

void populateLLVMToArithConversionPatterns(RewritePatternSet &patterns);

} // namespace arith

std::unique_ptr<Pass> createConvertLLVMToArithPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H
