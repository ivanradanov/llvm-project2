// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s

void foo(float *a, float *b, int n) {
#pragma transform label for1
  for (int i = 0; i < n; i++) {
    a[i] = b[i] * 2;
  }
}
