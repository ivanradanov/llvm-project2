//===- LoopUnrollPass.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPUNROLLANDINTERLEAVEPASS_H
#define LLVM_TRANSFORMS_SCALAR_LOOPUNROLLANDINTERLEAVEPASS_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class Loop;
class LPMUpdater;

class LoopUnrollAndInterleavePass
    : public PassInfoMixin<LoopUnrollAndInterleavePass> {

public:
  explicit LoopUnrollAndInterleavePass() {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

class UnrollAndInterleavePass : public PassInfoMixin<UnrollAndInterleavePass> {

public:
  explicit UnrollAndInterleavePass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLANDINTERLEAVEPASS_H
