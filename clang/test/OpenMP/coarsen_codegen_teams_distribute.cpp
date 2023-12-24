// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}

extern "C" void func(int start, int end, int step) {
#pragma omp teams distribute ompx_coarsen_distribute(2)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}

#endif

// IR-LABEL:  @func.omp_outlined(
// IR: call{{.*}}@body(
// IR-NOT:!llvm.loop
// IR:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
