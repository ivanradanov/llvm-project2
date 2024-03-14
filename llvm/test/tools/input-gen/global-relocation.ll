; RUN: mkdir -p %t/function-wise/
;
; RUN: (input-gen -g --verify --output-dir %t/function-wise --compile-input-gen-executables --input-gen-runtime %S/../../../../input-gen-runtimes/rt-input-gen.cpp --input-run-runtime %S/../../../../input-gen-runtimes/rt-run.cpp %s -function test && %t/function-wise/input-gen.function.test.generate.a.out %t/function-wise/ 0 1) | FileCheck %s --check-prefix=GGENN
; RUN: %t/function-wise/input-gen.function.test.run.a.out %t/function-wise/input-gen.function.test.generate.a.out.input.0.bin | FileCheck %s --check-prefix=RRUNN
; XFAIL: *
;
; GGENN: OUTPUT EQ
; RRUNN: OUTPUT EQ
;
;
; This test tricks the input generation to make b == &a and then checks if the
; pointer to the global is properly relocated from the obj input memory to the
; global memory.
;
; TODO Currently unsupported. We would need to have a callback after all of the
; __inputrun_global's to then re-re-locate the pointers to globals again. This
; is probably very rare as we would have needed to generate an offset to a
; global as a pointer for an input.
;
; int a;
; int *b;
;
; void eqtest() {
;   if (b == &a)
;     printf("OUTPUT EQ %p %p\n", b, &a);
;   else
;     printf("OUTPUT NE %p %p\n", b, &a);
;
; }
; void test() {
;   *b = 5;
;   for (int i = 0 ; i < 100; i++) {
;     eqtest();
;   }
; }

; ModuleID = 'global-relocation.cpp'
source_filename = "global-relocation.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

@b = dso_local local_unnamed_addr global ptr null, align 8
@a = dso_local global i32 0, align 4
@.str = private unnamed_addr constant [17 x i8] c"OUTPUT EQ %p %p\0A\00", align 1
@.str.1 = private unnamed_addr constant [17 x i8] c"OUTPUT NE %p %p\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @eqtest() local_unnamed_addr #0 {
  %1 = load ptr, ptr @b, align 8, !tbaa !3
  %2 = icmp eq ptr %1, @a
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @a, ptr noundef nonnull @a)
  br label %7

5:                                                ; preds = %0
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %1, ptr noundef nonnull @a)
  br label %7

7:                                                ; preds = %5, %3
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test() local_unnamed_addr #0 {
  %1 = load ptr, ptr @b, align 8, !tbaa !3
  store i32 5, ptr %1, align 4, !tbaa !7
  br label %3

2:                                                ; preds = %11
  ret void

3:                                                ; preds = %0, %11
  %4 = phi i32 [ 0, %0 ], [ %12, %11 ]
  %5 = load ptr, ptr @b, align 8, !tbaa !3
  %6 = icmp eq ptr %5, @a
  br i1 %6, label %7, label %9

7:                                                ; preds = %3
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @a, ptr noundef nonnull @a)
  br label %11

9:                                                ; preds = %3
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %5, ptr noundef nonnull @a)
  br label %11

11:                                               ; preds = %7, %9
  %12 = add nuw nsw i32 %4, 1
  %13 = icmp eq i32 %12, 100
  br i1 %13, label %2, label %3, !llvm.loop !9
}

attributes #0 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"clang version 16.0.6 (Red Hat 16.0.6-2.module+el8.9.0+19521+190d7aba)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
