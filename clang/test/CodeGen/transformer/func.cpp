// COM: %clang -g -c -O2 -emit-llvm -S -mllvm --transformer-enable %s -o - | FileCheck %s
// COM: %clang_cc1 -mllvm -emit-mlir -mllvm --transformer-enable -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s --check-prefix=CHKOUT
// RUN: true

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

// TODO enable this test only for release build as including `mlir` is too slow
// on debug builds

// TODO Either automatically make this extern "C" (or with a pragma
// transform_import or something) or mangle the use of it in transform_apply
// below
extern "C" void transform2(mlir::Operation *op1, mlir::Operation *op2) {
  llvm::outs() << "We are now in the compiler and can transform IR\n";
  llvm::outs() << "op1 is:\n" << *op1 << "\n";
  llvm::outs() << "op2 is:\n" << *op2 << "\n";
}

void foo(float *a, float *b, int n, int k) {

  #pragma transform_label for1
  for (int j = 0; j < n; j++) {
    #pragma transform_label for2
    for (int i = 0; i < n; i++) {
      a[i + j * k] = b[i + j * k] * 2;
    }
  }

  #pragma transform_apply transform2(for1, for2)

}
