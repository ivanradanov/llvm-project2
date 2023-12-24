// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}

extern "C" void func(int start, int end, int step,
                     int startj, int endj, int stepj) {
#pragma omp teams
  {
#pragma omp distribute ompx_coarsen_distribute(2)
    for (int i = start; i < end; i+=step) {
#pragma omp parallel
      {
#pragma omp for ompx_coarsen_for(3)
        for (int j = startj; j < endj; j+=stepj) {
          body(start, end, step, i,
               startj, endj, stepj, j);
        }
      }
    }
  }
}

#endif

// IR-LABEL:  @func.omp_outlined(
// IR call {{.*}} @__kmpc_fork_call({{.*}}@func.omp_outlined.omp_outlined
// IR:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR-LABEL:  @func.omp_outlined.omp_outlined(
// IR: call{{.*}}@body(
// IR:  br label %{{.*}} !llvm.loop ![[LOOP2:.+]]
// IR-DAG:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR-DAG:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
// IR-DAG:  ![[LOOP2]] = distinct{{.*}}![[LOOP3:[0-9]+]]
// IR-DAG:  ![[LOOP3]] = !{!"llvm.loop.unroll_and_interleave.count", i32 3}
