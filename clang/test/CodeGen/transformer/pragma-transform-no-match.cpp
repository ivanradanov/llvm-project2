// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -S -emit-llvm %s -o - 2>&1 | FileCheck %s

// CHECK: warning: unable to match label for_label


bool test(int);
void body();

void foo() {
  #pragma transform_label for_label
  for (int j = 0; test(j); j++) {
    body();
  }
}
