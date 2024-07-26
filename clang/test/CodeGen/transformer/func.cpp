// COM: %clang -g -c -O2 -emit-llvm -S -mllvm --transformer-enable %s -o - | FileCheck %s
// COM: %clang_cc1 -mllvm -emit-mlir -mllvm --transformer-enable -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s --check-prefix=CHKOUT
// RUN: true

#include <stdio.h>

#pragma transform_import
void transform2(void *op1, void *op2) {
  printf("in lambda %p\n", op1);
  printf("in lambda %p\n", op2);
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
