
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR

// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR

// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR

#ifndef HEADER
#define HEADER

// placeholder for loop body code.
extern "C" void body(...) {}

#ifdef CK1
extern "C" void func(int start, int end, int step) {
#pragma omp target teams distribute parallel for ompx_coarsen_distribute(2) ompx_coarsen_for(3)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}
#endif
#ifdef CK2
extern "C" void func(int start, int end, int step,
                     int startj, int endj, int stepj) {
#pragma omp target teams distribute ompx_coarsen_distribute(2)
  for (int i = start; i < end; i+=step)
#pragma omp parallel for ompx_coarsen_for(3)
    for (int j = startj; j < endj; j+=stepj)
      body(start, end, step, i,
           startj, endj, stepj, j);
}
#endif
#ifdef CK3
extern "C" void func(int start, int end, int step,
                     int startj, int endj, int stepj) {
#pragma omp target
  {
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
}
#endif

// IR:  call void @__kmpc_distribute_static_init_
// IR:  call void @__kmpc_parallel_51(
// IR:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR:  call void @__kmpc_for_static_init_
// IR: call{{.*}}@body(
// IR:  br label %{{.*}} !llvm.loop ![[LOOP2:.+]]
// IR-DAG:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR-DAG:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
// IR-DAG:  ![[LOOP2]] = distinct{{.*}}![[LOOP3:[0-9]+]]
// IR-DAG:  ![[LOOP3]] = !{!"llvm.loop.unroll_and_interleave.count", i32 3}
// IR-NOT: [[LOOP1]]
// IR-NOT: [[LOOP3]]
// IR-NOT:  "llvm.loop.unroll_and_interleave.count"

#ifdef CK4
extern "C" void func(int start, int end, int step) {
#pragma omp target parallel for ompx_coarsen_for(2)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}
#endif

// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR2
//
// IR2:  call void @__kmpc_for_static_init_
// IR2: call{{.*}}@body(
// IR2:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR2:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR2:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
// IR2-NOT: [[LOOP1]]
// IR2-NOT:  "llvm.loop.unroll_and_interleave.count"

#ifdef CK5
extern "C" void func(int start, int end, int step) {
#pragma omp target teams distribute ompx_coarsen_distribute(2)
  for (int i = start; i < end; i+=step)
    body(start, end, step, i);
}
#endif

// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR3
//
// IR3:  call void @__kmpc_distribute_static_init_
// IR3: call{{.*}}@body(
// IR3:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR3:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR3:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
// IR3-NOT: [[LOOP1]]
// IR3-NOT:  "llvm.loop.unroll_and_interleave.count"

#ifdef CK6
extern "C" void func(int start, int end, int step,
                     int startj, int endj, int stepj) {
#pragma omp target
  {
#pragma omp teams
    {
#pragma omp distribute ompx_coarsen_distribute(2) dist_schedule(static, 2)
      for (int i = start; i < end; i+=step) {
#pragma omp parallel
        {
#pragma omp for ompx_coarsen_for(3) schedule(static, 3)
          for (int j = startj; j < endj; j+=stepj) {
            body(start, end, step, i,
                 startj, endj, stepj, j);
          }
        }
      }
    }
  }
}
#endif

// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=IR4

// IR4:  call void @__kmpc_distribute_static_init_
// IR4:  call void @__kmpc_parallel_51(
// IR4:  br label %{{.*}} !llvm.loop ![[LOOP:.+]]
// IR4:  call void @__kmpc_for_static_init_
// IR4: call{{.*}}@body(
// IR4:  br label %{{.*}} !llvm.loop ![[LOOP2:.+]]
// IR4-DAG:  ![[LOOP]] = distinct{{.*}}![[LOOP1:[0-9]+]]
// IR4-DAG:  ![[LOOP1]] = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
// IR4-DAG:  ![[LOOP2]] = distinct{{.*}}![[LOOP3:[0-9]+]]
// IR4-DAG:  ![[LOOP3]] = !{!"llvm.loop.unroll_and_interleave.count", i32 3}
// IR4-NOT: [[LOOP1]]
// IR4-NOT: [[LOOP3]]
// IR4-NOT:  "llvm.loop.unroll_and_interleave.count"

#endif
