// RUN: mlir-opt %s --gpu-launch-to-parallel --split-input-file | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module} {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.mlir.global internal unnamed_addr @_ZZ10stencil_1dPKiPiE4temp() {addr_space = 3 : i32, alignment = 4 : i64, dso_local} : !llvm.array<270 x i32> {
      %0 = llvm.mlir.undef : !llvm.array<270 x i32>
      llvm.return %0 : !llvm.array<270 x i32>
    }
    llvm.func local_unnamed_addr @_Z10stencil_1dPKiPi(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %c-7_i32 = arith.constant -7 : i32
      %c263_i32 = arith.constant 263 : i32
      %c7_i32 = arith.constant 7 : i32
      %0 = llvm.mlir.addressof @_ZZ10stencil_1dPKiPiE4temp : !llvm.ptr<3>
      %1 = llvm.addrspacecast %0 : !llvm.ptr<3> to !llvm.ptr
      %2 = nvvm.read.ptx.sreg.tid.x : i32
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = nvvm.read.ptx.sreg.ntid.x : i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %5, %2 : i32
      %7 = arith.addi %2, %c7_i32 : i32
      %8 = arith.extsi %6 : i32 to i64
      %9 = llvm.getelementptr inbounds %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %10 = llvm.load %9 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
      %11 = arith.extui %7 : i32 to i64
      %12 = llvm.getelementptr inbounds %1[0, %11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
      llvm.store %10, %12 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
      %13 = arith.cmpi ult, %2, %c7_i32 : i32
      scf.if %13 {
        %16 = arith.cmpi slt, %6, %c7_i32 : i32
        %17 = scf.if %16 -> (i32) {
          scf.yield %c0_i32 : i32
        } else {
          %25 = arith.extui %6 : i32 to i64
          %26 = llvm.getelementptr %arg0[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          %27 = llvm.getelementptr %26[-28] : (!llvm.ptr) -> !llvm.ptr, i8
          %28 = llvm.load %27 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
          scf.yield %28 : i32
        }
        %18 = arith.extui %2 : i32 to i64
        %19 = llvm.getelementptr inbounds %1[0, %18] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
        llvm.store %17, %19 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
        %20 = llvm.getelementptr %9[1024] : (!llvm.ptr) -> !llvm.ptr, i8
        %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
        %22 = arith.addi %2, %c263_i32 : i32
        %23 = arith.extui %22 : i32 to i64
        %24 = llvm.getelementptr inbounds %1[0, %23] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
        llvm.store %21, %24 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
      }
      nvvm.barrier0
      %14 = scf.for %arg2 = %c-7_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %c0_i32) -> (i32)  : i32 {
        %16 = arith.addi %arg2, %7 : i32
        %17 = arith.extsi %16 : i32 to i64
        %18 = llvm.getelementptr inbounds %1[0, %17] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
        %19 = llvm.load %18 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
        %20 = arith.addi %19, %arg3 : i32
        scf.yield %20 : i32
      }
      %15 = llvm.getelementptr inbounds %arg1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %14, %15 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
      llvm.return
    }
  }
  llvm.func local_unnamed_addr @foo(%70 : i64, %71 : i64, %72 : i64, %73 : i64, %74 : i64, %75 : i64, %shm : i32, %a1 : !llvm.ptr, %a2 : !llvm.ptr) -> () {
    gpu.launch_func  @__mlir_gpu_module::@_Z10stencil_1dPKiPi blocks in (%70, %71, %72) threads in (%73, %74, %75) : i64 dynamic_shared_memory_size %shm args(%a1 : !llvm.ptr, %a2 : !llvm.ptr)
    llvm.return
  }
}

// CHECK-LABEL:   gpu.module @__mlir_gpu_module [#[[?]]<chip = "sm_80">]  {
// CHECK:           llvm.mlir.global internal unnamed_addr @_ZZ10stencil_1dPKiPiE4temp() {addr_space = 3 : i32, alignment = 4 : i64, dso_local} : !llvm.array<270 x i32> {
// CHECK:             %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<270 x i32>
// CHECK:             llvm.return %[[VAL_0]] : !llvm.array<270 x i32>
// CHECK:           }
// CHECK:           llvm.func local_unnamed_addr @_Z10stencil_1dPKiPi(%[[VAL_1:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_2:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #[[?]]<all>, gpu.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_cpu = "sm_52", target_features = #[[?]]<["+ptx84", "+sm_52"]>} {
// CHECK:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_5:.*]] = arith.constant 8 : i32
// CHECK:             %[[VAL_6:.*]] = arith.constant -7 : i32
// CHECK:             %[[VAL_7:.*]] = arith.constant 263 : i32
// CHECK:             %[[VAL_8:.*]] = arith.constant 7 : i32
// CHECK:             %[[VAL_9:.*]] = llvm.mlir.addressof @_ZZ10stencil_1dPKiPiE4temp : !llvm.ptr<3>
// CHECK:             %[[VAL_10:.*]] = llvm.addrspacecast %[[VAL_9]] : !llvm.ptr<3> to !llvm.ptr
// CHECK:             %[[VAL_11:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:             %[[VAL_12:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:             %[[VAL_13:.*]] = nvvm.read.ptx.sreg.ntid.x : i32
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_11]] : i32
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[VAL_8]] : i32
// CHECK:             %[[VAL_17:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_18:.*]] = llvm.getelementptr inbounds %[[VAL_1]]{{\[}}%[[VAL_17]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:             %[[VAL_19:.*]] = llvm.load %[[VAL_18]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : !llvm.ptr -> i32
// CHECK:             %[[VAL_20:.*]] = arith.extui %[[VAL_16]] : i32 to i64
// CHECK:             %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, %[[VAL_20]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:             llvm.store %[[VAL_19]], %[[VAL_21]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : i32, !llvm.ptr
// CHECK:             %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_8]] : i32
// CHECK:             scf.if %[[VAL_22]] {
// CHECK:               %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_8]] : i32
// CHECK:               %[[VAL_24:.*]] = scf.if %[[VAL_23]] -> (i32) {
// CHECK:                 scf.yield %[[VAL_4]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_25:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:                 %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_25]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_26]][-28] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:                 %[[VAL_28:.*]] = llvm.load %[[VAL_27]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : !llvm.ptr -> i32
// CHECK:                 scf.yield %[[VAL_28]] : i32
// CHECK:               }
// CHECK:               %[[VAL_29:.*]] = arith.extui %[[VAL_11]] : i32 to i64
// CHECK:               %[[VAL_30:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, %[[VAL_29]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               llvm.store %[[VAL_24]], %[[VAL_30]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : i32, !llvm.ptr
// CHECK:               %[[VAL_31:.*]] = llvm.getelementptr %[[VAL_18]][1024] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:               %[[VAL_32:.*]] = llvm.load %[[VAL_31]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : !llvm.ptr -> i32
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_11]], %[[VAL_7]] : i32
// CHECK:               %[[VAL_34:.*]] = arith.extui %[[VAL_33]] : i32 to i64
// CHECK:               %[[VAL_35:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, %[[VAL_34]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               llvm.store %[[VAL_32]], %[[VAL_35]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : i32, !llvm.ptr
// CHECK:             }
// CHECK:             nvvm.barrier0
// CHECK:             %[[VAL_36:.*]] = scf.for %[[VAL_37:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_3]] iter_args(%[[VAL_38:.*]] = %[[VAL_4]]) -> (i32)  : i32 {
// CHECK:               %[[VAL_39:.*]] = arith.addi %[[VAL_37]], %[[VAL_16]] : i32
// CHECK:               %[[VAL_40:.*]] = arith.extsi %[[VAL_39]] : i32 to i64
// CHECK:               %[[VAL_41:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, %[[VAL_40]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               %[[VAL_42:.*]] = llvm.load %[[VAL_41]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : !llvm.ptr -> i32
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_42]], %[[VAL_38]] : i32
// CHECK:               scf.yield %[[VAL_43]] : i32
// CHECK:             }
// CHECK:             %[[VAL_44:.*]] = llvm.getelementptr inbounds %[[VAL_2]]{{\[}}%[[VAL_17]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:             llvm.store %[[VAL_36]], %[[VAL_44]] {alignment = 4 : i64, tbaa = [#[[$ATTR_3]]]} : i32, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }

// CHECK-LABEL:   llvm.func local_unnamed_addr @foo(
// CHECK-SAME:                                      %[[VAL_0:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_1:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_2:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_3:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_4:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_5:[^:]*]]: i64,
// CHECK-SAME:                                      %[[VAL_6:[^:]*]]: i32,
// CHECK-SAME:                                      %[[VAL_7:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                      %[[VAL_8:[^:]*]]: !llvm.ptr) {
// CHECK:           gpu.launch_func  @__mlir_gpu_module::@_Z10stencil_1dPKiPi blocks in (%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) threads in (%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) : i64 dynamic_shared_memory_size %[[VAL_6]] args(%[[VAL_7]] : !llvm.ptr, %[[VAL_8]] : !llvm.ptr)
// CHECK:           llvm.return
// CHECK:         }

