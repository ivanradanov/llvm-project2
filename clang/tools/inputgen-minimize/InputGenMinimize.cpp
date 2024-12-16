//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs clang based on the info
//  stored in a compilation database.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/TokenBufferTokenManager.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "\tFor example, to run clang-check on all files in a subtree of the\n"
    "\tsource tree, use:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check\n"
    "\n"
    "\tor using a specific build path:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check -p build/path\n"
    "\n"
    "\tNote, that path/in/subtree and current directory should follow the\n"
    "\trules described above.\n"
    "\n");

static cl::OptionCategory ClangCheckCategory("clang-check options");
static const opt::OptTable &Options = getDriverOptTable();
static cl::opt<bool>
    ASTDump("ast-dump",
            cl::desc(Options.getOptionHelpText(options::OPT_ast_dump)),
            cl::cat(ClangCheckCategory));
static cl::opt<bool>
    ASTList("ast-list",
            cl::desc(Options.getOptionHelpText(options::OPT_ast_list)),
            cl::cat(ClangCheckCategory));
static cl::opt<bool>
    ASTPrint("ast-print",
             cl::desc(Options.getOptionHelpText(options::OPT_ast_print)),
             cl::cat(ClangCheckCategory));
static cl::opt<std::string> ASTDumpFilter(
    "ast-dump-filter",
    cl::desc(Options.getOptionHelpText(options::OPT_ast_dump_filter)),
    cl::cat(ClangCheckCategory));
static cl::opt<bool>
    Analyze("analyze",
            cl::desc(Options.getOptionHelpText(options::OPT_analyze)),
            cl::cat(ClangCheckCategory));
static cl::opt<std::string>
    AnalyzerOutput("analyzer-output-path",
                   cl::desc(Options.getOptionHelpText(options::OPT_o)),
                   cl::cat(ClangCheckCategory));

static cl::opt<bool>
    Fixit("fixit", cl::desc(Options.getOptionHelpText(options::OPT_fixit)),
          cl::cat(ClangCheckCategory));
static cl::opt<bool> FixWhatYouCan(
    "fix-what-you-can",
    cl::desc(Options.getOptionHelpText(options::OPT_fix_what_you_can)),
    cl::cat(ClangCheckCategory));

static cl::opt<bool> SyntaxTreeDump("syntax-tree-dump",
                                    cl::desc("dump the syntax tree"),
                                    cl::cat(ClangCheckCategory));
static cl::opt<bool> TokensDump("tokens-dump",
                                cl::desc("dump the preprocessed tokens"),
                                cl::cat(ClangCheckCategory));

namespace {
using namespace clang;

class ASTMinimizer : public ASTConsumer,
                     public RecursiveASTVisitor<ASTMinimizer> {
  typedef RecursiveASTVisitor<ASTMinimizer> base;

public:
  ASTMinimizer(std::unique_ptr<raw_ostream> Out, StringRef FilterString,
               bool DumpDeclTypes = false)
      : Out(Out ? *Out : llvm::outs()), OwnedOut(std::move(Out)),
        FilterString(FilterString), DumpDeclTypes(DumpDeclTypes) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    TranslationUnitDecl *D = Context.getTranslationUnitDecl();

    if (FilterString.empty())
      return print(D);

    TraverseDecl(D);
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool generateEntryPoint(FunctionDecl *FD) {
    std::string Indent = "  ";

    Out << "void __inputrun_entry(char *Args) {\n";

    Out << Indent << "char *CurArg = Args;\n";
    for (unsigned I = 0; I < FD->getNumParams(); I++) {
      ParmVarDecl *Param = FD->getParamDecl(I);

      QualType T = Param->getType();
      std::string TypeStr = T.getAsString();
      Out << Indent << TypeStr << " Arg" << I << " = *(" << TypeStr
          << " *) CurArg;\n";
      Out << Indent << "CurArg += 16;\n";
    }

    Out << Indent << FD->getNameAsString() << "(";
    for (unsigned I = 0; I < FD->getNumParams(); I++) {
      if (I != 0)
        Out << ", ";
      Out << "Arg" << I;
    }
    Out << ");\n";

    Out << "}\n";
    return true;
  }

  bool TraverseDecl(Decl *D) {
    if (D && filterMatches(D)) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        Out << "\n";
        Out << "// =========== RECORDED FUNCTION BEGIN ===========\n";
        print(D);
        Out << "// =========== RECORDED FUNCTION END ===========\n";
        Out << "\n";
        Out << "// =========== GENERATED ENTRY POINT BEGIN ===========\n";
        generateEntryPoint(FD);
        Out << "// =========== GENERATED ENTRY POINT END ===========\n";
        // Don't traverse child nodes to avoid output duplication.
        return true;
      }
    }
    return base::TraverseDecl(D);
  }

private:
  std::string getName(Decl *D) {
    if (isa<NamedDecl>(D))
      return cast<NamedDecl>(D)->getQualifiedNameAsString();
    return "";
  }
  bool filterMatches(Decl *D) {
    return getName(D).find(FilterString) != std::string::npos;
  }
  void print(Decl *D) {
    PrintingPolicy Policy(D->getASTContext().getLangOpts());
    D->print(Out, Policy, /*Indentation=*/0, /*PrintInstantiation=*/true);

    if (DumpDeclTypes) {
      Decl *InnerD = D;
      if (auto *TD = dyn_cast<TemplateDecl>(D))
        if (Decl *TempD = TD->getTemplatedDecl())
          InnerD = TempD;

      // FIXME: Support combining -ast-dump-decl-types with -ast-dump-lookups.
      if (auto *VD = dyn_cast<ValueDecl>(InnerD))
        VD->getType().dump(Out, VD->getASTContext());
      if (auto *TD = dyn_cast<TypeDecl>(InnerD))
        TD->getTypeForDecl()->dump(Out, TD->getASTContext());
    }
  }

  raw_ostream &Out;
  std::unique_ptr<raw_ostream> OwnedOut;

  /// Which declarations or DeclContexts to display.
  std::string FilterString;

  /// Whether to dump the type for each declaration dumped.
  bool DumpDeclTypes;
};

std::unique_ptr<ASTConsumer>
CreateASTMinimizer(std::unique_ptr<raw_ostream> Out, StringRef FilterString) {
  return std::make_unique<ASTMinimizer>(std::move(Out), FilterString);
}

class MimimizeFactory {
public:
  std::unique_ptr<clang::ASTConsumer> newASTConsumer() {
    return CreateASTMinimizer(nullptr, ASTDumpFilter);
  }
};

} // namespace

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, ClangCheckCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  if (Analyze) {
    // Set output path if is provided by user.
    //
    // As the original -o options have been removed by default via the
    // strip-output adjuster, we only need to add the analyzer -o options here
    // when it is provided by users.
    if (!AnalyzerOutput.empty())
      Tool.appendArgumentsAdjuster(
          getInsertArgumentAdjuster(CommandLineArguments{"-o", AnalyzerOutput},
                                    ArgumentInsertPosition::END));

    // Running the analyzer requires --analyze. Other modes can work with the
    // -fsyntax-only option.
    //
    // The syntax-only adjuster is installed by default.
    // Good: It also strips options that trigger extra output, like -save-temps.
    // Bad:  We don't want the -fsyntax-only when executing the static analyzer.
    //
    // To enable the static analyzer, we first strip all -fsyntax-only options
    // and then add an --analyze option to the front.
    Tool.appendArgumentsAdjuster(
        [&](const CommandLineArguments &Args, StringRef /*unused*/) {
          CommandLineArguments AdjustedArgs;
          for (const std::string &Arg : Args)
            if (Arg != "-fsyntax-only")
              AdjustedArgs.emplace_back(Arg);
          return AdjustedArgs;
        });
    Tool.appendArgumentsAdjuster(
        getInsertArgumentAdjuster("--analyze", ArgumentInsertPosition::BEGIN));
  }

  MimimizeFactory CheckFactory;
  std::unique_ptr<FrontendActionFactory> FrontendFactory;

  FrontendFactory = newFrontendActionFactory(&CheckFactory);

  return Tool.run(FrontendFactory.get());
}
