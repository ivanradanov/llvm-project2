

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/InputGenerationImpl.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace llvm;

cl::OptionCategory InputGenCategory("input-gen Options");

static cl::opt<std::string> OutputDir("output-dir", cl::Required,
                                      cl::cat(InputGenCategory));

static cl::opt<std::string> GenRuntime(
    "input-gen-runtime",
    cl::desc("Input gen runtime to link into the instrumented module."),
    cl::cat(InputGenCategory));

static cl::opt<std::string> RunRuntime(
    "input-run-runtime",
    cl::desc("Input gen runtime to link into the instrumented module."),
    cl::cat(InputGenCategory));

static cl::opt<std::string> InputFilename(cl::Positional, cl::init("-"),
                                          cl::desc("Input file"),
                                          cl::cat(InputGenCategory));

static cl::opt<bool> CompileInputGenBinaries("compile-input-gen-binaries",
                                             cl::cat(InputGenCategory));

constexpr char ToolName[] = "input-gen";

[[noreturn]] static void fatalError(const Twine &Message) {
  errs() << Message << "\n";
  exit(1);
}

std::string createTempFile(const Twine &Prefix, StringRef Suffix) {
  std::error_code EC;
  SmallString<128> FileName;
  if ((EC = sys::fs::createTemporaryFile(Prefix, Suffix, FileName)))
    fatalError("Unable to create temp file: " + EC.message());
  return static_cast<std::string>(FileName);
}

ErrorOr<std::string> findClang(const char *Argv0, StringRef Triple) {
  // This just needs to be some symbol in the binary.
  void *P = (void *)(intptr_t)findClang;
  std::string MainExecPath = llvm::sys::fs::getMainExecutable(Argv0, P);
  if (MainExecPath.empty())
    MainExecPath = Argv0;

  ErrorOr<std::string> Path = std::error_code();
  std::string TargetClang = (Triple + "-clang++").str();
  std::string VersionedClang = ("clang++-" + Twine(LLVM_VERSION_MAJOR)).str();
  for (const auto *Name :
       {TargetClang.c_str(), VersionedClang.c_str(), "clang++"}) {
    for (const StringRef Parent : {llvm::sys::path::parent_path(MainExecPath),
                                   llvm::sys::path::parent_path(Argv0)}) {
      // Look for various versions of "clang" first in the MainExecPath parent
      // directory and then in the argv[0] parent directory.
      // On Windows (but not Unix) argv[0] is overwritten with the eqiuvalent
      // of MainExecPath by InitLLVM.
      Path = sys::findProgramByName(Name, Parent);
      if (Path)
        return Path;
    }
  }

  // If no parent directory known, or not found there, look everywhere in PATH
  for (const auto *Name : {"clang", "clang-cl"}) {
    Path = sys::findProgramByName(Name);
    if (Path)
      return Path;
  }
  return Path;
}

static bool writeProgramToFile(int FD, const Module &M) {
  raw_fd_ostream OS(FD, /*shouldClose*/ false);
  WriteBitcodeToFile(M, OS, /*ShouldPreserveUseListOrder*/ false);
  OS.flush();
  if (!OS.has_error())
    return false;
  OS.clear_error();
  return true;
}

class InputGenOrchestration {
public:
  Module &M;
  std::string Clang;
  InputGenOrchestration(Module &M) : M(M){};
  void init(int Argc, char **Argv) {
    if (CompileInputGenBinaries) {
      if (GenRuntime.empty() || RunRuntime.empty())
        fatalError("input-gen: Need to specify input-gen runtimes to compile "
                   "executables.");

      ErrorOr<std::string> ClangOrErr =
          findClang(Argv[0], "ignoring-this-for-now");
      if (ClangOrErr) {
        Clang = *ClangOrErr;
      } else {
        errs() << "input-gen: Unable to find clang\n";
        fatalError("input-gen: Unable to generate input.");
      }
    }
  }

  std::vector<Function *> Functions;

  bool shouldGen(Function &F) { return !F.isDeclaration(); }

  void genAllFunctionsForRuntime(std::string RuntimeName,
                                 IGInstrumentationModeTy Mode) {

    ValueToValueMapTy VMap;
    auto InstrM = CloneModule(M, VMap);

    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModuleInputGenInstrumenter MIGI(*InstrM, MAM, Mode);
    bool Success = MIGI.instrumentModule(*InstrM);
    if (!Success) {
      llvm::outs() << "Instrumenting failed\n";
      return;
    }
    std::string ModeStr = [&]() {
      switch (Mode) {
      case llvm::IG_Generate:
        return "generate";
      case llvm::IG_Run:
        return "run";
      case llvm::IG_Record:
        llvm_unreachable("Unsupported mode");
      }
      llvm_unreachable("Unknown mode");
    }();
    for (size_t It = 0; It < Functions.size(); It++) {
      Function &F = *Functions[It];

      std::string FuncName = F.getName().str();

      std::string EntryBcFileName = OutputDir + "/" + "input-gen.entry." +
                                    FuncName + "." + ModeStr + ".bc";
      std::string EntrySoFileName = OutputDir + "/" + "input-gen.entry." +
                                    FuncName + "." + ModeStr + ".so";

      llvm::outs() << "Handling function @" << F.getName() << "\n";
      llvm::outs() << "Instrumenting...\n";

      auto Success =
          MIGI.instrumentEntryPoint(*InstrM, *cast<Function>(VMap[&F]), true);

      if (!Success) {
        llvm::outs() << "Instrumenting failed\n";
        continue;
      }
    }
    std::string BcFileName =
        OutputDir + "/" + "input-gen.module." + ModeStr + ".bc";
    std::string ExecutableFileName =
        OutputDir + "/" + "input-gen.module." + ModeStr + ".a.out";
    if (!writeModuleToFile(*InstrM, BcFileName)) {
      llvm::outs() << "Writing instrumented module to file failed\n";
      exit(1);
    }
    if (CompileInputGenBinaries) {
      if (!compileExecutable(BcFileName, ExecutableFileName, RuntimeName)) {
        llvm::outs() << "Compiling instrumented module failed\n";
        exit(1);
      }
    }
  }

  void dumpFunctions() {
    std::string AvailFuncsFileName = OutputDir + "/" + "available_functions";
    auto Fs = std::ofstream(AvailFuncsFileName);
    for (auto &F : M.getFunctionList()) {
      if (shouldGen(F)) {
        Fs << F.getName().str() << std::endl;
        Functions.push_back(&F);
      }
    }
  }

  void genAllFunctionForAllRuntimes() {
    genAllFunctionsForRuntime(GenRuntime, IG_Generate);
    genAllFunctionsForRuntime(RunRuntime, IG_Run);
  }

  bool writeModuleToFile(Module &M, std::string FileName) {
    int FD;
    std::error_code EC = sys::fs::openFileForWrite(FileName, FD);
    if (EC) {
      // errc::permission_denied happens on Windows when we try to open a file
      // that has been marked for deletion.
      // if (!(EC == std::errc::file_exists ||
      //       EC == std::errc::permission_denied)) {
      //   ?
      // }
      errs() << ToolName << ": Could not open '" << FileName << "'!\n";
      exit(1);
    }

    if (writeProgramToFile(FD, M)) {
      errs() << ToolName << ": Error emitting bitcode to file '" << FileName
             << "'!\n";
      close(FD);
      exit(1);
    }
    close(FD);
    return true;
  }

  bool compileExecutable(std::string ModuleName, std::string ExecutableName,
                         std::string RuntimeName) {
    if (CompileInputGenBinaries) {
      outs() << "Compiling " << ExecutableName << "\n";
      SmallVector<StringRef, 8> Args = {
          Clang,       "-fopenmp", "-O2", "-ldl",        "-rdynamic",
          RuntimeName, ModuleName, "-o",  ExecutableName};
      std::string ErrMsg;
      int Res = sys::ExecuteAndWait(
          Args[0], Args, /*Env=*/std::nullopt, /*Redirects=*/{},
          /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
      if (Res) {
        errs() << "input-gen: executable compilation failed";
        if (!ErrMsg.empty())
          errs() << ": " << ErrMsg;
        errs() << ".\n";
        errs() << "Args: ";
        for (auto Arg : Args)
          errs() << "\"" << Arg << "\" ";
        errs() << "\n";
        return false;
      }
    }

    return true;
  }
};

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(InputGenCategory);
  cl::ParseCommandLineOptions(argc, argv, "Input gen");

  ExitOnError ExitOnErr("input-gen: ");
  LLVMContext Context;

  SMDiagnostic Diag;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Diag, Context);
  if (!M) {
    Diag.print("input-gen", errs());
    return 1;
  }

  InputGenOrchestration IGO(*M);

  IGO.init(argc, argv);
  IGO.dumpFunctions();
  IGO.genAllFunctionForAllRuntimes();

  return 0;
}
