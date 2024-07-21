// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s

static const char *t1 = R"TRANSFORM(
transform.sequence %arg0: !transform.any_op {} {
  transform.loop.unroll %arg0 { factor = 4 } : !transform.any_op
  transform.yield
}
)TRANSFORM";


void foo(float *a, float *b, int n) {
#pragma transform label for1
  for (int i = 0; i < n; i++) {
    a[i] = b[i] * 2;
  }
#pragma transform run t1(for1)
}
