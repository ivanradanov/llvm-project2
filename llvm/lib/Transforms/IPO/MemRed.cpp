#include "llvm/Transforms/IPO/MemRed.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "../../Target/NVPTX/NVPTXUtilities.h"

using namespace llvm;

static cl::opt<std::string> ClOutDir(
    "memred-outdir", cl::init("."), cl::Hidden,
    cl::desc("memred-dir"));


PreservedAnalyses MemRedPass::run(Module &M,
                                  ModuleAnalysisManager &MAM) {
  auto T = Triple(M.getTargetTriple());
  if (!(T.isNVPTX() || T.isAMDGCN() || T.isAMDGPU()))
    return PreservedAnalyses::none();

  auto &Log = llvm::errs();
  for (auto &F : M) {
    if (!isKernelFunction(F))
      continue;

    Log << "Function @" << demangle(F.getName()) << ":\n";
    for (unsigned I = 0; I < F.getFunctionType()->getNumParams(); I++) {
      if (!F.getArg(I)->getType()->isPointerTy())
        continue;
      Log << " Arg #" << I << ": ";
      for (auto Attr : F.getAttributes().getParamAttrs(I)) {
        Log << Attr.getAsString() << " ";
      }
      Log << "\n";
    }
    Log << "\n";
  }

  return PreservedAnalyses::none();
}
