//===-------- Definition of the input generation pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the input generation pass that is used, together with a
// runtime, to generate inputs for code snippets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_INPUTGENERATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_INPUTGENERATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;

/// Public interface to the input generation module pass for instrumenting code
/// to generate inputs (incl. memory states) for code.
class InputGenerationInstrumentPass
    : public PassInfoMixin<InputGenerationInstrumentPass> {
public:
  explicit InputGenerationInstrumentPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
