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
#include "clang/AST/Decl.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
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
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory
    InputGenMinimizeCat("inputgen-minimize options");

namespace {
using namespace clang;

#define TRY_TO(CALL_EXPR)                                                      \
  do {                                                                         \
    if (!CALL_EXPR)                                                            \
      return false;                                                            \
  } while (false)

#define INPUTGEN_DEBUG(CODE)                                                   \
  do {                                                                         \
  } while (0)

class DepGatherer : public ASTConsumer,
                    public RecursiveASTVisitor<DepGatherer> {
  typedef RecursiveASTVisitor<DepGatherer> base;

public:
  std::set<const Decl *> &Deps;
  DepGatherer(std::set<const Decl *> &Deps) : Deps(Deps) {}

  // We should not be traversing the contents of namespace or translation
  // units as we are trying to minimize their contents and we do not depend on
  // their entirety.
  bool ShouldTraverse(const Decl *D) {
    return !isa<NamespaceDecl, TranslationUnitDecl>(D);
  }

  bool InsertDep(Decl *D) {
    if (Deps.count(D))
      return false;

    Deps.insert(D);

    const auto &Parents = D->getASTContext().getParents<Decl>(*D);
    for (auto &Parent : Parents) {
      if (const Decl *D = Parent.get<Decl>()) {
        if (ShouldTraverse(D))
          GatherDeps(const_cast<Decl *>(D));
        else
          Deps.insert(D);
      }
    }

    return true;
  }

  bool GatherDepsFromThisDecl(Decl *D) {
    if (!InsertDep(D))
      return true;

    INPUTGEN_DEBUG(llvm::errs() << "Traversing\n"; D->dumpColor(););

    return base::TraverseDecl(D);
  }

  bool VisitTypedefDecl(TypedefDecl *D) {
    TRY_TO(TraverseType(D->getUnderlyingType()));
    return true;
  }

  bool VisitTypeAliasDecl(TypeAliasDecl *D) {
    TRY_TO(TraverseType(D->getUnderlyingType()));
    return true;
  }

  bool VisitDecls(Decl *D) {
    TRY_TO(GatherDeps(D));
    return true;
  }

  bool GatherDeps(Decl *D) {
    for (Decl *RD : D->redecls()) {
      INPUTGEN_DEBUG(llvm::errs() << "Redecl\n"; RD->dumpColor(););
      TRY_TO(GatherDepsFromThisDecl(RD));
    }
    return true;
  }

  std::set<const Type *> HandledTypes;
  void GatherDeps(QualType QT) { GatherDeps(QT.getTypePtr()); }
  void GatherDeps(const Type *T) {
    if (HandledTypes.count(T))
      return;
    HandledTypes.insert(T);
    INPUTGEN_DEBUG(llvm::errs() << "Used Type\n"; T->dump(););

    // TODO is that all types that can have decls?
    if (const TypedefType *TT = T->getAs<TypedefType>()) {
      GatherDeps(TT->getDecl());
    } else if (const RecordType *TT = T->getAs<RecordType>()) {
      GatherDeps(TT->getDecl());
    } else if (const EnumType *TT = T->getAs<EnumType>()) {
      GatherDeps(TT->getDecl());
    }
  }

  bool VisitType(Type *T) {
    INPUTGEN_DEBUG(llvm::errs() << "Traverse Type\n");
    GatherDeps(T);
    return true;
  }

  bool VisitValueDecl(ValueDecl *D) {
    TRY_TO(TraverseType(D->getType()));
    return base::VisitValueDecl(D);
  }

  bool VisitStmt(Stmt *S) {
    INPUTGEN_DEBUG(llvm::errs() << "Visit\n"; S->dumpColor(););
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    llvm::errs() << "Visit DeclRefExpr\n";
    GatherDeps(E->getDecl());
    return true;
  }

  bool VisitCallExpr(CallExpr *E) {
    Decl *FD = E->getCalleeDecl();
    // TRY_TO(GatherDeps(FD));
    //  TODO if we don't have access to the body there are a couple of options:
    //
    //    1. There are multiple decl's and we dont have the one with the body? -
    //    DONE
    //
    //    2. The definition is in another file? We would like integration with
    //    compile_commands.json and to be able to track those down.
    //
    //    3. The definition is in an external library - in that case we would
    //    like to generate the correct #include for it or just print the
    //    declaration.
    //
    //    Looks like there is a way to merge AST's together:
    //    https://clang.llvm.org/docs/LibASTImporter.html can we merge in all
    //    ASTs and then print out the extracted c/c++ from there?
    return true;
  }

private:
};

class ASTMinimizer : public ASTConsumer,
                     public RecursiveASTVisitor<ASTMinimizer> {
  typedef RecursiveASTVisitor<ASTMinimizer> base;

  bool Done = false;

public:
  ASTMinimizer(std::unique_ptr<raw_ostream> Out)
      : Out(Out ? *Out : llvm::outs()), OwnedOut(std::move(Out)) {}

  TranslationUnitDecl *TUD;
  void HandleTranslationUnit(ASTContext &Context) override {
    TUD = Context.getTranslationUnitDecl();
    TraverseDecl(TUD);
  }

  bool shouldWalkTypesOfTypeLocs() const { return false; }

  void generateEntryPoint(FunctionDecl *FD) {
    PrintingPolicy Policy(FD->getASTContext().getLangOpts());
    std::string Indent = "  ";

    Out << "void __inputrun_entry(char *Args) {\n";

    Out << Indent << "char *CurArg = Args;\n";
    for (unsigned I = 0; I < FD->getNumParams(); I++) {
      ParmVarDecl *Param = FD->getParamDecl(I);

      QualType T = Param->getType();
      std::string TypeStr = T.getAsString(Policy);
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
  }

  void PrintWithDeps(Decl *D) {
    std::set<const Decl *> Deps;
    DepGatherer DG(Deps);
    DG.GatherDeps(D);

    // TODO if possible try to omit the inputgen attr
    PrintingPolicy Policy(D->getASTContext().getLangOpts());
    std::function<bool(const Decl *)> Filter = [&Deps](const Decl *D) -> bool {
      return Deps.count(D);
    };
    TUD->print(Out, Policy, /*Indentation=*/0, /*PrintInstantiation=*/false,
               &Filter);
  }

  bool TraverseDecl(Decl *D) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      if (FD->hasAttr<InputGenEntryAttr>()) {
        if (Done) {
          llvm::errs() << "Multiple inputgen entry points found, ignoring "
                       << FD->getName() << "\n";
          return true;
        }
        Out << "\n";
        Out << "// =========== RECORDED FUNCTION BEGIN ===========\n";
        PrintWithDeps(D);
        Out << "// =========== RECORDED FUNCTION END ===========\n";
        Out << "\n";
        Out << "// =========== GENERATED ENTRY POINT BEGIN ===========\n";
        generateEntryPoint(FD);
        Out << "// =========== GENERATED ENTRY POINT END ===========\n";
        Done = true;
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
  void print(Decl *D) {
    // TODO if possible try to omit the inputgen attr
    PrintingPolicy Policy(D->getASTContext().getLangOpts());
    D->print(Out, Policy, /*Indentation=*/0, /*PrintInstantiation=*/true);
  }

  raw_ostream &Out;
  std::unique_ptr<raw_ostream> OwnedOut;
};

std::unique_ptr<ASTConsumer>
CreateASTMinimizer(std::unique_ptr<raw_ostream> Out) {
  return std::make_unique<ASTMinimizer>(std::move(Out));
}

class MimimizeFactory {
public:
  std::unique_ptr<clang::ASTConsumer> newASTConsumer() {
    return CreateASTMinimizer(nullptr);
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
      CommonOptionsParser::create(argc, argv, InputGenMinimizeCat);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  MimimizeFactory CheckFactory;
  std::unique_ptr<FrontendActionFactory> FrontendFactory;

  FrontendFactory = newFrontendActionFactory(&CheckFactory);

  return Tool.run(FrontendFactory.get());
}
