// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -S -emit-llvm %s -o - 2>&1 | FileCheck %s

// CHECK-DAG: info: found match for label for_label
// CHECK-DAG: info: found match for label for_label
// CHECK-DAG: info: found match for label for_label2
// CHECK-DAG: info: found match for label for_label2
// CHECK-DAG: info: found match for label for_label2

void body();

void foo(int n) {
  #pragma transform_label for_label
  for (int j = 0; j < n; j++) {
    body();
  }

  #pragma transform_label for_label
  for (int j = 0; j < n; j++) {
    body();
  }
}

template <typename T>
void bar(T n) {
  #pragma transform_label for_label2
  for (T j = 0; j < n; j++) {
    body();
  }
}

template void bar(int n);
template void bar(short n);
template void bar(long n);

