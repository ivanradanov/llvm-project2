#ifndef LLVM_TRANSFORMS_IPO_MEMRED_H
#define LLVM_TRANSFORMS_IPO_MEMRED_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class MemRedAnalysePass : public PassInfoMixin<MemRedAnalysePass> {
public:
  PreservedAnalyses run(Module &, ModuleAnalysisManager &);
};
} // namespace llvm

#endif
