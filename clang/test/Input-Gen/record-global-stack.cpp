// RUN: mkdir -p %t
// RUN: %clangxx -g -O2 %s -mllvm --input-gen-mode=record -o %t/record.a.out -linputgen.record
// RUN: %clangxx -g -O2 %s -mllvm --input-gen-mode=replay -rdynamic -o %t/replay.a.out -linputgen.replay
// RUN: INPUT_RECORD_FILENAME=%t/input.bin %t/record.a.out | FileCheck %s
// RUN: %t/replay.a.out %t/input.bin | FileCheck %s
// RUN: %S/../../../scripts/inputgen_minimize.py --source-file %s --embed-input-file %t/input.bin --output-file %t/minimized.cpp
// RUN: %clangxx %t/minimized.cpp -o %t/minimized.a.out
// RUN: %t/minimized.a.out | FileCheck %s
// RUN: %S/../../../scripts/inputgen_minimize.py --source-file %s --output-file %t/minimized.cpp
//
// FIXME: undefined reference to `__inputrun_function_pointers'
// COM: %clangxx %t/minimized.cpp -o %t/minimized.a.out
// COM: %t/minimized.a.out %t/input.bin | FileCheck %s

// CHECK: Sum: 495

#include <stdio.h>
#include <stdlib.h>
#include <alloca.h>

#define N 10

// FIXME we should make sure the function is noinline without requiring the
// programmer to do this
__attribute__((inputgen_entry))
int foo(int *a, int *b, int *c, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i] * n;
    sum += c[i];
  }
  printf("Sum: %d\n", sum);
  return sum;
}

int b[N];

int main() {
  int *a = (int *)malloc(N * sizeof(*a));
  int *c = (int *)alloca(N * sizeof(*c));

  for (int i = 0; i < N; i++) {
    a[i] = b[i] = i % 10;
  }

  int d = foo(a, b, c, N);
  printf("Output: %d\n", d);

  free(a);
  return 0;
}
