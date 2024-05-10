#include "llvm/Transforms/IPO/MemRed.h"
#include "../../Target/NVPTX/NVPTXUtilities.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static cl::opt<std::string> ClOutDir("memred-outdir", cl::init("."), cl::Hidden,
                                     cl::desc("memred-dir"));

PreservedAnalyses MemRedAnalysePass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto T = Triple(M.getTargetTriple());
  if (!(T.isNVPTX() || T.isAMDGCN() || T.isAMDGPU()))
    return PreservedAnalyses::none();

  auto &Log = llvm::errs();
  for (auto &F : M) {
    if (!isKernelFunction(F))
      continue;

    Log << "Function " << demangle(F.getName()) << " (@" << F.getName()
        << "):\n";

    Log << "  Memory Effect: ";
    if (F.getMemoryEffects().doesNotAccessMemory())
      Log << "None";
    else if (F.getMemoryEffects().onlyAccessesArgPointees())
      Log << "ArgMemOnly";
    else
      Log << "AnyMem";
    Log << "\n";
    for (auto &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy())
        continue;
      Log << "  Arg #" << Arg.getArgNo() << ":\t";
      Log << "Effect: ";
      if (Arg.hasAttribute(Attribute::ReadOnly))
        Log << "ReadOnly ";
      else if (Arg.hasAttribute(Attribute::WriteOnly))
        Log << "WriteOnly";
      else if (Arg.hasAttribute(Attribute::ReadNone))
        Log << "None     ";
      else
        Log << "Unknown  ";
      Log << " ";

      Log << "Capture: ";
      if (Arg.hasAttribute(Attribute::NoCapture))
        Log << "No";
      else
        Log << "Yes";
      Log << "\n";
    }
    Log << "\n";
  }

  return PreservedAnalyses::none();
}
