// RUN: %clang_cc1 -mllvm --transformer-enable -triple x86_64-pc-linux-gnu -S -emit-llvm %s -o - | FileCheck %s


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

// CHECK-DAG: @.src = private unnamed_addr constant [77 x i8] c"/scr/ivan/src/Polygeist/llvm-project/clang/test/CodeGen/pragma-transform.cpp\00", align 1
// CHECK-DAG: @__clang_transformer_for_label = private unnamed_addr constant [5 x i8] c"for1\00", align 1
// CHECK-DAG: @__clang_transformer_for_label.1 = private unnamed_addr constant [5 x i8] c"for2\00", align 1
// CHECK-DAG: @__clang_transformer_import = private unnamed_addr constant [188 x i8] c"\0A  transform.named_sequence @transform2(%arg0: !transform.any_op, %arg0: !transform.any_op) {\0A    transform.loop.unroll %arg0 { factor = 2 } : !transform.any_op\0A    transform.yield\0A  }\0A  \00", align 1
// CHECK-DAG: @__clang_transformer_apply_func = private unnamed_addr constant [11 x i8] c"transform2\00", align 1
// CHECK-DAG: @__clang_transformer_apply_arg = private unnamed_addr constant [5 x i8] c"for4\00", align 1
// CHECK-DAG: @__clang_transformer_for_label.2 = private unnamed_addr constant [12 x i8] c"another_for\00", align 1
// CHECK-DAG: @__clang_transformer_for_locs = appending local_unnamed_addr global [3 x { ptr, { ptr, i32, i32 }, { ptr, i32, i32 } }] [{ ptr, { ptr, i32, i32 }, { ptr, i32, i32 } } { ptr @__clang_transformer_for_label, { ptr, i32, i32 } { ptr @.src, i32 7, i32 3 }, { ptr, i32, i32 } { ptr @.src, i32 12, i32 3 } }, { ptr, { ptr, i32, i32 }, { ptr, i32, i32 } } { ptr @__clang_transformer_for_label.1, { ptr, i32, i32 } { ptr @.src, i32 9, i32 5 }, { ptr, i32, i32 } { ptr @.src, i32 11, i32 5 } }, { ptr, { ptr, i32, i32 }, { ptr, i32, i32 } } { ptr @__clang_transformer_for_label.2, { ptr, i32, i32 } { ptr @.src, i32 25, i32 3 }, { ptr, i32, i32 } { ptr @.src, i32 27, i32 3 } }], section "llvm.metadata"
// CHECK-DAG: @__clang_transformer_import.3 = private unnamed_addr constant [162 x i8] c"\0A  transform.named_sequence @transform3(%arg0: !transform.any_op) {\0A    transform.loop.unroll %arg0 { factor = 3 } : !transform.any_op\0A    transform.yield\0A  }\0A  \00", align 1
// CHECK-DAG: @__clang_transformer_import_array = appending local_unnamed_addr global [2 x ptr] [ptr @__clang_transformer_import, ptr @__clang_transformer_import.3], section "llvm.metadata"
// CHECK-DAG: @__clang_transformer_apply_func.4 = private unnamed_addr constant [11 x i8] c"transform3\00", align 1
// CHECK-DAG: @__clang_transformer_apply_array = appending local_unnamed_addr global [7 x ptr] [ptr @__clang_transformer_apply_func, ptr @__clang_transformer_for_label.1, ptr @__clang_transformer_apply_arg, ptr null, ptr @__clang_transformer_apply_func.4, ptr @__clang_transformer_for_label.2, ptr null], section "llvm.metadata"
