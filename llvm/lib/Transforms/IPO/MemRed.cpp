#include "llvm/Transforms/IPO/MemRed.h"
#include "../../Target/NVPTX/NVPTXUtilities.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <fstream>

using namespace llvm;

static cl::opt<std::string>
    ClMemoryAnalysisOut("memred-memory-analysis-out",
                        cl::init("./.memred.memory.analysis.out"), cl::Hidden,
                        cl::desc("memred-dir"));

static cl::opt<std::string> ClMode("memred-mode", cl::init(""));

static constexpr std::array<StringRef, 4> CudaCallsToInstrument = {
    "cudaMalloc", "cudaMemcpy", "cudaMemcpyAsync", "cudaLaunchKernel"};
static constexpr StringRef InstrumentedPrefix = "__memred_";

static bool isDevice(Module &M) {
  auto T = Triple(M.getTargetTriple());
  return T.isNVPTX() || T.isAMDGCN() || T.isAMDGPU();
}

PreservedAnalyses MemRedInstrumentPass::run(Module &M,
                                            ModuleAnalysisManager &MAM) {
  if (ClMode != "trace")
    return PreservedAnalyses::all();
  if (isDevice(M))
    return PreservedAnalyses::all();

  for (auto Name : CudaCallsToInstrument) {
    if (auto *F = M.getFunction(Name)) {
      assert(F->isDeclaration());
      F->setName(InstrumentedPrefix + Name);
    }
  }
  return PreservedAnalyses::none();
}

PreservedAnalyses MemRedAnalysePass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  if (!isDevice(M))
    return PreservedAnalyses::all();

  std::ofstream File(ClMemoryAnalysisOut, std::fstream::app);
  llvm::raw_os_ostream Log(File);
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
