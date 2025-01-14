// RUN: %clang -g -c -O2 -emit-llvm -mllvm --transformer-enable %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: warning: unable to match label for_label


bool test(int);
void body();

void foo() {
  #pragma transform_label for_label
  for (int j = 0; test(j); j++) {
    body();
  }
}
