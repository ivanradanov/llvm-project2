#ifndef LLVM_TRANSFORMS_CUDALAUNCHFIXUP_H
#define LLVM_TRANSFORMS_CUDALAUNCHFIXUP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Pass;

struct CUDALaunchFixUp : PassInfoMixin<CUDALaunchFixUp> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif
