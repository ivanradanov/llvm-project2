// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s


void foo(float *a, float *b, int n, int k) {

  #pragma transform_label for1
  for (int j = 0; j < n; j++) {
    #pragma transform_label for2
    for (int i = 0; i < n; i++) {
      a[i + j * k] = b[i + j * k] * 2;
    }
  }

  #pragma transform_import
  R"TRANSFORM(
  transform.named_sequence @transform2(%arg0: !transform.any_op, %arg0: !transform.any_op) {
    transform.loop.unroll %arg0 { factor = 2 } : !transform.any_op
    transform.yield
  }
  )TRANSFORM";

  #pragma transform_apply transform2(for2, for4)

  #pragma transform_label another_for
  for (int i = 0; i < n; i++) {
    a[i] = b[i] * 3;
  }

  #pragma transform_import
  R"TRANSFORM(
  transform.named_sequence @transform3(%arg0: !transform.any_op) {
    transform.loop.unroll %arg0 { factor = 3 } : !transform.any_op
    transform.yield
  }
  )TRANSFORM";

  #pragma transform_apply transform3(another_for)

}
