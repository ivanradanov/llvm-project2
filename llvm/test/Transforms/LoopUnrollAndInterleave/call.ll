; RUN: opt -passes=loop-unroll-and-interleave %s -S --luai-factor=2 | FileCheck %s
; RUN: opt -passes=loop-unroll-and-interleave %s -S --luai-factor=2 --luai-use-dynamic-convergence=1 | FileCheck %s --check-prefix=DRCHECK

; void bar(int *a, int j);
; void foo(int *a, int j) {
;   bar(a, j);
; }
; #pragma omp target teams distribute parallel for map(tofrom: a[0:size]) //schedule(static, 4)
;   for (unsigned long i = 0; i < size; i++) {
;     foo(a, i);
;   }

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DynamicEnvironmentTy = type { i16 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }

@__omp_rtl_debug_kind = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_teams_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_no_nested_parallelism = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@anon.63bf84d140b44cdc04f4584e183c8175.0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@anon.63bf84d140b44cdc04f4584e183c8175.1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @anon.63bf84d140b44cdc04f4584e183c8175.0 }, align 8
@__omp_offloading_58_1cec360__Z6vecaddPii_l14_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_58_1cec360__Z6vecaddPii_l14_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 0, i32 0, i32 0, i32 0 }, ptr addrspacecast (ptr addrspace(1) @anon.63bf84d140b44cdc04f4584e183c8175.1 to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_58_1cec360__Z6vecaddPii_l14_dynamic_environment to ptr) }
@anon.63bf84d140b44cdc04f4584e183c8175.2 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2050, i32 0, i32 22, ptr @anon.63bf84d140b44cdc04f4584e183c8175.0 }, align 8
@anon.63bf84d140b44cdc04f4584e183c8175.3 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @anon.63bf84d140b44cdc04f4584e183c8175.0 }, align 8
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 400

declare i32 @__kmpc_target_init(ptr, ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: nounwind
declare void @__kmpc_distribute_static_init_8u(ptr, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64) local_unnamed_addr #2

; Function Attrs: alwaysinline convergent norecurse nounwind
define internal void @__omp_offloading_58_1cec360__Z6vecaddPii_l14_omp_outlined_omp_outlined(ptr noalias nocapture noundef readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 noundef %.previous.lb., i64 noundef %.previous.ub., i64 noundef %size, ptr noundef %a) #3 {
entry:
  %.omp.lb = alloca i64, align 8, addrspace(5)
  %.omp.ub = alloca i64, align 8, addrspace(5)
  %.omp.stride = alloca i64, align 8, addrspace(5)
  %.omp.is_last = alloca i32, align 4, addrspace(5)
  %sext.mask = and i64 %size, 4294967295
  %cmp.not = icmp eq i64 %sext.mask, 0
  br i1 %cmp.not, label %omp.precond.end, label %omp.precond.then

omp.precond.then:                                 ; preds = %entry
  %.omp.stride.ascast = addrspacecast ptr addrspace(5) %.omp.stride to ptr
  %.omp.is_last.ascast = addrspacecast ptr addrspace(5) %.omp.is_last to ptr
  %.omp.ub.ascast = addrspacecast ptr addrspace(5) %.omp.ub to ptr
  %.omp.lb.ascast = addrspacecast ptr addrspace(5) %.omp.lb to ptr
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %.omp.lb) #2
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %.omp.ub) #2
  store i64 %.previous.lb., ptr addrspace(5) %.omp.lb, align 8, !tbaa !16
  store i64 %.previous.ub., ptr addrspace(5) %.omp.ub, align 8, !tbaa !16
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %.omp.stride) #2
  store i64 1, ptr addrspace(5) %.omp.stride, align 8, !tbaa !16
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.is_last) #2
  store i32 0, ptr addrspace(5) %.omp.is_last, align 4, !tbaa !20
  %0 = load i32, ptr %.global_tid., align 4, !tbaa !20
  call void @__kmpc_for_static_init_8u(ptr addrspacecast (ptr addrspace(1) @anon.63bf84d140b44cdc04f4584e183c8175.3 to ptr), i32 %0, i32 33, ptr nocapture nonnull %.omp.is_last.ascast, ptr nocapture nonnull %.omp.lb.ascast, ptr nocapture nonnull %.omp.ub.ascast, ptr nocapture nonnull %.omp.stride.ascast, i64 1, i64 1) #2
  %1 = load i64, ptr addrspace(5) %.omp.lb, align 8, !tbaa !16
  %add = add i64 %.previous.ub., 1
  %cmp413 = icmp ult i64 %1, %add
  br i1 %cmp413, label %omp.inner.for.body.lr.ph, label %omp.loop.exit

omp.inner.for.body.lr.ph:                         ; preds = %omp.precond.then
  %2 = load i64, ptr addrspace(5) %.omp.stride, align 8
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body, %omp.inner.for.body.lr.ph
  %.omp.iv.014 = phi i64 [ %1, %omp.inner.for.body.lr.ph ], [ %add7, %omp.inner.for.body ]
  %conv6 = trunc i64 %.omp.iv.014 to i32
  tail call void @_Z3fooPii(ptr noundef %a, i32 noundef %conv6) #9
  %add7 = add i64 %2, %.omp.iv.014
  %cmp4 = icmp ult i64 %add7, %add
  br i1 %cmp4, label %omp.inner.for.body, label %omp.loop.exit, !llvm.loop !45

omp.loop.exit:                                    ; preds = %omp.inner.for.body, %omp.precond.then
  tail call void @__kmpc_distribute_static_fini(ptr addrspacecast (ptr addrspace(1) @anon.63bf84d140b44cdc04f4584e183c8175.2 to ptr), i32 %0) #2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.is_last) #2
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %.omp.stride) #2
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %.omp.ub) #2
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %.omp.lb) #2
  br label %omp.precond.end

omp.precond.end:                                  ; preds = %omp.loop.exit, %entry
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, ptr nocapture nofree, i64, i64) local_unnamed_addr #2

; Function Attrs: convergent mustprogress noinline nounwind
define hidden void @_Z3fooPii(ptr noundef %a, i32 noundef %j) local_unnamed_addr #4 {
entry:
  tail call void @_Z3barPii(ptr noundef %a, i32 noundef %j) #9
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_distribute_static_fini(ptr, i32) local_unnamed_addr #2

; Function Attrs: alwaysinline
declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64) local_unnamed_addr #5

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) local_unnamed_addr #2

declare void @__kmpc_target_deinit() local_unnamed_addr

; Function Attrs: convergent nounwind
declare void @_Z3barPii(ptr noundef, i32 noundef) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #7

attributes #0 = { alwaysinline norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="256" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }
attributes #3 = { alwaysinline convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #4 = { convergent mustprogress noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #5 = { alwaysinline }
attributes #6 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { nounwind memory(readwrite) }
attributes #9 = { convergent nounwind }

!omp_offload.info = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9}
!opencl.ocl.version = !{!10}
!llvm.ident = !{!11, !12}

!0 = !{i32 0, i32 88, i32 30327648, !"_Z6vecaddPii", i32 14, i32 0, i32 0}
!2 = !{i32 1, !"amdgpu_code_object_version", i32 400}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"openmp", i32 51}
!5 = !{i32 7, !"openmp-device", i32 51}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 4, !"amdgpu_hostcall", i32 1}
!8 = !{i32 1, !"ThinLTO", i32 0}
!9 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!10 = !{i32 2, i32 0}
!11 = !{!"clang version 18.0.0git (git@github.com:ivanradanov/llvm-project2.git 4fe47d1f12fc3b2544e40919fdf9447c0f094892)"}
!12 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.7.0 23352 d1e13c532a947d0cbfc94759c00dcf152294aa13)"}
!13 = !{!14}
!14 = distinct !{!14, !15, !"__omp_offloading_58_1cec360__Z6vecaddPii_l14_omp_outlined: %.global_tid."}
!15 = distinct !{!15, !"__omp_offloading_58_1cec360__Z6vecaddPii_l14_omp_outlined"}
!16 = !{!17, !17, i64 0}
!17 = !{!"long", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C++ TBAA"}
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !18, i64 0}
!22 = !{!23, !23, i64 0}
!23 = !{!"any pointer", !18, i64 0}
!45 = distinct !{!45, !46}
!46 = !{!"llvm.loop.unroll_and_interleave.count", i32 2}
