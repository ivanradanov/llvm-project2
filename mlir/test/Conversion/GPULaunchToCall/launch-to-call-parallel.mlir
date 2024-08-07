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
  llvm.mlir.global private unnamed_addr constant @".str"("Usage: %s <length> <repeat>\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.1"("length is a multiple of %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global internal unnamed_addr @_ZZ4mainE4done(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global private unnamed_addr constant @".str.2"("cuda stencil1d-cuda stencil_1d.cu\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.3"("0\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"("Average kernel execution time: %f (s)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("Error at %d: %d (host) != %d (device)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("PASS\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("FAIL\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"("MY_TIMING_FILE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @stdout() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
  llvm.mlir.global private unnamed_addr constant @".str.12"("a\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @stderr() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
  llvm.mlir.global private unnamed_addr constant @".str.13"("Could not open timing file %s, errno %d, %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.14"("HOSTNAME\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.15"("unidetified_compiler\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.16"("%s,%s,%.17g,%s,%s,%d,\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global internal unnamed_addr @__cuda_gpubin_handle() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @_Z25__device_stub__stencil_1dPKiPi(%arg0: !llvm.ptr {llvm.noalias, llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.noundef}) attributes {passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["uniform-work-group-size", "true"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @_Z25__device_stub__stencil_1dPKiPi : !llvm.ptr
    %1 = llvm.alloca %c1_i32 x !llvm.struct<"struct.dim3", (i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.struct<"struct.dim3", (i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.call @__cudaPopCallConfiguration(%1, %2, %3, %4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> i64
    %7 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %8 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> i64
    %9 = llvm.getelementptr inbounds %1[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %10 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> i32
    %11 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> i64
    %12 = llvm.getelementptr inbounds %2[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %13 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i32
    llvm.call @__mlir_launch_coerced_kernel__Z25__device_stub__stencil_1dPKiPi(%0, %8, %10, %11, %13, %6, %7, %arg0, %arg1) : (!llvm.ptr, i64, i32, i64, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @__cudaPopCallConfiguration(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = ub.poison : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = ub.poison : i64
    %2 = ub.poison : !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %3 = llvm.mlir.addressof @".str" : !llvm.ptr
    %4 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %c256_i32 = arith.constant 256 : i32
    %5 = llvm.mlir.zero : !llvm.ptr
    %c10_i32 = arith.constant 10 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %6 = llvm.mlir.addressof @_Z25__device_stub__stencil_1dPKiPi : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZZ4mainE4done : !llvm.ptr
    %true = arith.constant true
    %8 = llvm.mlir.constant(1.000000e-09 : f64) : f64
    %9 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %c0_i8 = arith.constant 0 : i8
    %10 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %11 = llvm.mlir.addressof @stdout : !llvm.ptr
    %12 = llvm.mlir.addressof @".str.14" : !llvm.ptr
    %13 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %14 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %15 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %16 = llvm.mlir.addressof @".str.15" : !llvm.ptr
    %17 = llvm.mlir.addressof @stderr : !llvm.ptr
    %18 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %19 = llvm.mlir.constant(9.99999971E-10 : f32) : f32
    %20 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %21 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %c7_i64 = arith.constant 7 : i64
    %c-7_i32 = arith.constant -7 : i32
    %c13_i64 = arith.constant 13 : i64
    %22 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %23 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %c14_i32 = arith.constant 14 : i32
    %24 = llvm.alloca %c1_i32 x !llvm.array<1 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %25 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %26 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %27 = llvm.alloca %c1_i32 x !llvm.struct<"struct.timespec", (i64, i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %28 = llvm.alloca %c1_i32 x !llvm.struct<"struct.timespec", (i64, i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %29 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %30:2 = scf.if %29 -> (i32, i32) {
      %32 = llvm.getelementptr inbounds %arg1[8] : (!llvm.ptr) -> !llvm.ptr, i8
      %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
      %34 = llvm.call tail @__isoc23_strtol(%33, %5, %c10_i32) {no_unwind} : (!llvm.ptr, !llvm.ptr, i32) -> i64
      %35 = arith.trunci %34 : i64 to i32
      %36 = llvm.getelementptr inbounds %arg1[16] : (!llvm.ptr) -> !llvm.ptr, i8
      %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
      %38 = llvm.call tail @__isoc23_strtol(%37, %5, %c10_i32) {no_unwind} : (!llvm.ptr, !llvm.ptr, i32) -> i64
      %39 = arith.shli %35, %c2_i32 : i32
      %40 = arith.addi %35, %c7_i32 : i32
      %41 = arith.shli %40, %c2_i32 : i32
      %42 = arith.extsi %41 : i32 to i64
      %43 = llvm.call tail @malloc(%42) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, will_return} : (i64) -> !llvm.ptr
      %44 = arith.extsi %39 : i32 to i64
      %45 = llvm.call tail @malloc(%44) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, will_return} : (i64) -> !llvm.ptr
      %46 = arith.maxsi %40, %c0_i32 : i32
      %47 = arith.extui %46 : i32 to i64
      scf.for %arg2 = %c0_i64 to %47 step %c1_i64  : i64 {
        %67 = llvm.getelementptr inbounds %43[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %68 = arith.trunci %arg2 : i64 to i32
        llvm.store %68, %67 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : i32, !llvm.ptr
      }
      %48 = arith.trunci %38 : i64 to i32
      llvm.intr.lifetime.start 8, %25 : !llvm.ptr
      llvm.intr.lifetime.start 8, %26 : !llvm.ptr
      %49 = llvm.call @cudaMalloc(%25, %42) : (!llvm.ptr, i64) -> i32
      %50 = llvm.call @cudaMalloc(%26, %44) : (!llvm.ptr, i64) -> i32
      %51 = llvm.load %25 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
      %52 = llvm.call @cudaMemcpy(%51, %43, %42, %c1_i32) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
      %53 = arith.divsi %35, %c256_i32 : i32
      %54 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %55 = llvm.call @_ZNSt6chrono3_V212steady_clock3nowEv() {no_unwind} : () -> i64
      llvm.intr.lifetime.start 16, %27 : !llvm.ptr
      llvm.intr.lifetime.start 16, %28 : !llvm.ptr
      %56 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %57 = llvm.call @clock_gettime(%c1_i32, %27) {no_unwind} : (i32, !llvm.ptr) -> i32
      %58 = arith.maxsi %48, %c0_i32 : i32
      scf.for %arg2 = %c0_i32 to %58 step %c1_i32  : i32 {
        %67 = llvm.load %25 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
        %68 = llvm.load %26 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
        %70 = llvm.sext %53 : i32 to i64
        %71 = llvm.sext %c1_i32 : i32 to i64
        %72 = llvm.sext %c1_i32 : i32 to i64
        %73 = llvm.sext %c256_i32 : i32 to i64
        %74 = llvm.sext %c1_i32 : i32 to i64
        %75 = llvm.sext %c1_i32 : i32 to i64
        gpu.launch_func  @__mlir_gpu_module::@_Z10stencil_1dPKiPi blocks in (%70, %71, %72) threads in (%73, %74, %75) : i64 dynamic_shared_memory_size %c0_i32 args(%67 : !llvm.ptr, %68 : !llvm.ptr)
      }
      %59 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %60 = llvm.call @_ZNSt6chrono3_V212steady_clock3nowEv() {no_unwind} : () -> i64
      %61 = arith.subi %60, %55 : i64
      %62 = llvm.call @cudaDeviceSynchronize() : () -> i32
      %63 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i1
      %64 = scf.if %63 -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        llvm.store %true, %7 {alignment = 4 : i64} : i1, !llvm.ptr
        %67 = llvm.call @clock_gettime(%c1_i32, %28) {no_unwind} : (i32, !llvm.ptr) -> i32
        %68 = llvm.load %28 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTS8timespec", members = {<#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i64
        %69 = llvm.load %27 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTS8timespec", members = {<#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i64
        %70 = arith.subi %68, %69 : i64
        %71 = llvm.getelementptr inbounds %28[8] : (!llvm.ptr) -> !llvm.ptr, i8
        %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTS8timespec", members = {<#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> i64
        %73 = llvm.getelementptr inbounds %27[8] : (!llvm.ptr) -> !llvm.ptr, i8
        %74 = llvm.load %73 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTS8timespec", members = {<#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 8>}>, access_type = <id = "long", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 8>]} : !llvm.ptr -> i64
        %75 = arith.subi %72, %74 : i64
        %76 = arith.sitofp %70 : i64 to f64
        %77 = arith.sitofp %75 : i64 to f64
        %78 = llvm.intr.fmuladd(%77, %8, %76)  : (f64, f64, f64) -> f64
        %79 = llvm.call @getenv(%9) {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, no_unwind} : (!llvm.ptr) -> !llvm.ptr
        %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
        %81 = llvm.ptrtoint %5 : !llvm.ptr to i64
        %82 = arith.cmpi eq, %80, %81 : i64
        %83:2 = scf.if %82 -> (!llvm.ptr, i32) {
          scf.yield %2, %c0_i32 : !llvm.ptr, i32
        } else {
          %90 = llvm.load %79 {alignment = 1 : i64} : !llvm.ptr -> i8
          %91 = arith.index_castui %90 : i8 to index
          %92:3 = scf.index_switch %91 -> !llvm.ptr, i32, i32
          case 0 {
            scf.yield %2, %c0_i32, %c1_i32 : !llvm.ptr, i32, i32
          }
          case 45 {
            %95 = llvm.getelementptr inbounds %79[1] : (!llvm.ptr) -> !llvm.ptr, i8
            %96 = llvm.load %95 {alignment = 1 : i64} : !llvm.ptr -> i8
            %97 = arith.cmpi eq, %96, %c0_i8 : i8
            %98 = arith.extui %97 : i1 to i32
            scf.yield %2, %c0_i32, %98 : !llvm.ptr, i32, i32
          }
          default {
            scf.yield %2, %0, %c0_i32 : !llvm.ptr, i32, i32
          }
          %93 = arith.index_castui %92#2 : i32 to index
          %94:2 = scf.index_switch %93 -> !llvm.ptr, i32
          case 0 {
            %95 = llvm.call @fopen(%79, %10) {no_unwind} : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
            scf.yield %95, %c1_i32 : !llvm.ptr, i32
          }
          default {
            scf.yield %92#0, %92#1 : !llvm.ptr, i32
          }
          scf.yield %94#0, %94#1 : !llvm.ptr, i32
        }
        %84 = arith.index_castui %83#1 : i32 to index
        %85 = scf.index_switch %84 -> !llvm.ptr
        case 0 {
          %90 = llvm.load %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
          scf.yield %90 : !llvm.ptr
        }
        default {
          scf.yield %83#0 : !llvm.ptr
        }
        %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
        %87 = llvm.ptrtoint %5 : !llvm.ptr to i64
        %88 = arith.cmpi eq, %86, %87 : i64
        %89 = arith.extui %88 : i1 to i32
        scf.if %88 {
          %90 = llvm.load %17 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
          %91 = llvm.call tail @__errno_location() {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : () -> !llvm.ptr
          %92 = llvm.load %91 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
          %93 = llvm.call @strerror(%92) {no_unwind} : (i32) -> !llvm.ptr
          %94 = llvm.call @fprintf(%90, %18, %79, %92, %93) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
          llvm.call @exit(%c1_i32) {no_unwind} : (i32) -> ()
        } else {
          llvm.intr.lifetime.start 1, %24 : !llvm.ptr
          llvm.store %c0_i8, %24 {alignment = 1 : i64} : i8, !llvm.ptr
          %90 = llvm.call @getenv(%12) {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, no_unwind} : (!llvm.ptr) -> !llvm.ptr
          %91 = llvm.ptrtoint %90 : !llvm.ptr to i64
          %92 = llvm.ptrtoint %5 : !llvm.ptr to i64
          %93 = arith.cmpi eq, %91, %92 : i64
          %94 = arith.select %93, %24, %90 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
          %95 = llvm.call @fprintf(%85, %13, %14, %15, %78, %94, %16, %c1_i32) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, i32) -> i32
          %96 = llvm.load %11 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
          %97 = llvm.ptrtoint %85 : !llvm.ptr to i64
          %98 = llvm.ptrtoint %96 : !llvm.ptr to i64
          %99 = arith.cmpi eq, %97, %98 : i64
          %100 = llvm.load %17 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
          %101 = llvm.ptrtoint %85 : !llvm.ptr to i64
          %102 = llvm.ptrtoint %100 : !llvm.ptr to i64
          %103 = arith.cmpi eq, %101, %102 : i64
          %104 = arith.select %99, %true, %103 {fastmathFlags = #llvm.fastmath<none>} : i1
          scf.if %104 {
          } else {
            %105 = llvm.call @fclose(%85) {no_unwind} : (!llvm.ptr) -> i32
          }
          llvm.intr.lifetime.end 1, %24 : !llvm.ptr
        }
        scf.yield %89 : i32
      }
      %65 = arith.index_castui %64 : i32 to index
      %66:2 = scf.index_switch %65 -> i32, i32
      case 0 {
        %67 = arith.sitofp %61 : i64 to f32
        %68 = llvm.fmul %67, %19  : f32
        %69 = arith.sitofp %48 : i32 to f32
        %70 = llvm.fdiv %68, %69  : f32
        %71 = llvm.fpext %70 : f32 to f64
        %72 = llvm.call @printf(%20, %71) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, f64) -> i32
        %73 = llvm.load %26 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
        %74 = llvm.call @cudaMemcpy(%45, %73, %44, %c2_i32) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
        %75:6 = scf.while (%arg2 = %c15_i64, %arg3 = %c0_i64) : (i64, i64) -> (i64, i64, i32, i32, i64, i32) {
          %88 = arith.cmpi eq, %arg3, %c14_i64 : i64
          %89:6 = scf.if %88 -> (i64, i64, i32, i32, i32, i32) {
            scf.yield %1, %1, %c2_i32, %c0_i32, %0, %0 : i64, i64, i32, i32, i32, i32
          } else {
            %91 = scf.for %arg4 = %arg3 to %arg2 step %c1_i64 iter_args(%arg5 = %c0_i32) -> (i32)  : i64 {
              %99 = arith.cmpi ult, %arg4, %c7_i64 : i64
              %100 = scf.if %99 -> (i32) {
                scf.yield %c0_i32 : i32
              } else {
                %102 = llvm.getelementptr inbounds %43[%arg4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
                %103 = llvm.load %102 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
                %104 = arith.addi %103, %c-7_i32 : i32
                scf.yield %104 : i32
              }
              %101 = arith.addi %100, %arg5 : i32
              scf.yield %101 : i32
            }
            %92 = llvm.getelementptr inbounds %45[%arg3] : (!llvm.ptr, i64) -> !llvm.ptr, i32
            %93 = llvm.load %92 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
            %94 = arith.cmpi eq, %91, %93 : i32
            %95 = arith.cmpi ne, %91, %93 : i32
            %96 = arith.extui %95 : i1 to i32
            %97 = arith.extui %94 : i1 to i32
            %98:2 = scf.if %94 -> (i64, i64) {
              %99 = arith.addi %arg3, %c1_i64 : i64
              %100 = arith.addi %arg2, %c1_i64 : i64
              scf.yield %100, %99 : i64, i64
            } else {
              scf.yield %1, %1 : i64, i64
            }
            scf.yield %98#0, %98#1, %96, %97, %93, %91 : i64, i64, i32, i32, i32, i32
          }
          %90 = arith.trunci %89#3 : i32 to i1
          scf.condition(%90) %89#0, %89#1, %89#4, %89#5, %arg3, %89#2 : i64, i64, i32, i32, i64, i32
        } do {
        ^bb0(%arg2: i64, %arg3: i64, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32):
          scf.yield %arg2, %arg3 : i64, i64
        }
        %76 = arith.index_castui %75#5 : i32 to index
        %77 = scf.index_switch %76 -> !llvm.ptr
        case 1 {
          %88 = arith.cmpi ugt, %75#4, %c13_i64 : i64
          %89 = arith.trunci %75#4 : i64 to i32
          %90 = llvm.call @printf(%22, %89, %75#3, %75#2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32, i32) -> i32
          %91 = arith.select %88, %21, %23 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
          scf.yield %91 : !llvm.ptr
        }
        default {
          scf.yield %21 : !llvm.ptr
        }
        %78 = arith.maxsi %35, %c14_i32 : i32
        %79 = arith.extui %78 : i32 to i64
        %80:6 = scf.while (%arg2 = %c14_i64, %arg3 = %c7_i64) : (i64, i64) -> (i64, i64, i32, i32, i64, i32) {
          %88 = arith.cmpi eq, %arg2, %79 : i64
          %89:6 = scf.if %88 -> (i64, i64, i32, i32, i32, i32) {
            scf.yield %1, %1, %c2_i32, %c0_i32, %0, %0 : i64, i64, i32, i32, i32, i32
          } else {
            %91 = arith.addi %arg2, %c7_i64 : i64
            %92:3 = scf.while (%arg4 = %arg3, %arg5 = %c0_i32) : (i64, i32) -> (i64, i32, i32) {
              %100 = arith.cmpi ugt, %arg4, %91 : i64
              %101 = arith.cmpi ule, %arg4, %91 : i64
              %102:2 = scf.if %100 -> (i64, i32) {
                scf.yield %1, %0 : i64, i32
              } else {
                %103 = llvm.getelementptr inbounds %43[%arg4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
                %104 = llvm.load %103 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
                %105 = arith.addi %104, %arg5 : i32
                %106 = arith.addi %arg4, %c1_i64 : i64
                scf.yield %106, %105 : i64, i32
              }
              scf.condition(%101) %102#0, %102#1, %arg5 : i64, i32, i32
            } do {
            ^bb0(%arg4: i64, %arg5: i32, %arg6: i32):
              scf.yield %arg4, %arg5 : i64, i32
            }
            %93 = llvm.getelementptr inbounds %45[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i32
            %94 = llvm.load %93 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "int", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> i32
            %95 = arith.cmpi eq, %92#2, %94 : i32
            %96 = arith.cmpi ne, %92#2, %94 : i32
            %97 = arith.extui %96 : i1 to i32
            %98 = arith.extui %95 : i1 to i32
            %99:2 = scf.if %95 -> (i64, i64) {
              %100 = arith.addi %arg2, %c1_i64 : i64
              %101 = arith.addi %arg3, %c1_i64 : i64
              scf.yield %100, %101 : i64, i64
            } else {
              scf.yield %1, %1 : i64, i64
            }
            scf.yield %99#0, %99#1, %97, %98, %94, %92#2 : i64, i64, i32, i32, i32, i32
          }
          %90 = arith.trunci %89#3 : i32 to i1
          scf.condition(%90) %89#0, %89#1, %89#4, %89#5, %arg2, %89#2 : i64, i64, i32, i32, i64, i32
        } do {
        ^bb0(%arg2: i64, %arg3: i64, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i32):
          scf.yield %arg2, %arg3 : i64, i64
        }
        %81 = arith.index_castui %80#5 : i32 to index
        %82 = scf.index_switch %81 -> !llvm.ptr
        case 1 {
          %88 = arith.trunci %80#4 : i64 to i32
          %89 = llvm.call @printf(%22, %88, %80#3, %80#2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32, i32) -> i32
          scf.yield %23 : !llvm.ptr
        }
        default {
          scf.yield %77 : !llvm.ptr
        }
        %83 = llvm.call @puts(%82) {no_unwind} : (!llvm.ptr) -> i32
        llvm.call @free(%43) {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, will_return} : (!llvm.ptr) -> ()
        llvm.call @free(%45) {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, will_return} : (!llvm.ptr) -> ()
        %84 = llvm.load %25 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
        %85 = llvm.call @cudaFree(%84) : (!llvm.ptr) -> i32
        %86 = llvm.load %26 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
        %87 = llvm.call @cudaFree(%86) : (!llvm.ptr) -> i32
        llvm.intr.lifetime.end 16, %28 : !llvm.ptr
        llvm.intr.lifetime.end 16, %27 : !llvm.ptr
        llvm.intr.lifetime.end 8, %26 : !llvm.ptr
        llvm.intr.lifetime.end 8, %25 : !llvm.ptr
        scf.yield %c0_i32, %c0_i32 : i32, i32
      }
      default {
        scf.yield %0, %c1_i32 : i32, i32
      }
      scf.yield %66#0, %66#1 : i32, i32
    } else {
      %32 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : !llvm.ptr -> !llvm.ptr
      %33 = llvm.call tail @printf(%3, %32) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr) -> i32
      %34 = llvm.call tail @printf(%4, %c256_i32) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32) -> i32
      scf.yield %c1_i32, %c0_i32 : i32, i32
    }
    cf.switch %30#1 : i32, [
      default: ^bb2,
      0: ^bb1(%30#0 : i32)
    ]
  ^bb1(%31: i32):  // pred: ^bb0
    llvm.return %31 : i32
  ^bb2:  // pred: ^bb0
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, passthrough = ["mustprogress", "nofree", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
  llvm.func local_unnamed_addr @cudaMalloc(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @cudaMemcpy(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @_ZNSt6chrono3_V212steady_clock3nowEv() -> i64 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @clock_gettime(i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, passthrough = ["mustprogress", ["allockind", "4"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
  llvm.func local_unnamed_addr @cudaFree(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @__isoc23_strtol(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @getenv(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @fopen(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @fprintf(!llvm.ptr {llvm.nocapture, llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @__errno_location() -> !llvm.ptr attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
  llvm.func local_unnamed_addr @strerror(i32 {llvm.noundef}) -> !llvm.ptr attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @exit(i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", "noreturn", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @fclose(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func local_unnamed_addr @__cudaRegisterFunction(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func local_unnamed_addr @__cudaRegisterFatBinary(!llvm.ptr) -> !llvm.ptr
  llvm.func local_unnamed_addr @__cudaRegisterFatBinaryEnd(!llvm.ptr)
  llvm.func local_unnamed_addr @__cudaUnregisterFatBinary(!llvm.ptr)
  llvm.func internal @__cuda_module_dtor() attributes {dso_local} {
    %0 = llvm.mlir.addressof @__cuda_gpubin_handle : !llvm.ptr
    %1 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.call tail @__cudaUnregisterFatBinary(%1) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @atexit(!llvm.ptr) -> i32 attributes {passthrough = ["nofree"]}
  llvm.func local_unnamed_addr @__mlir_launch_coerced_kernel__Z25__device_stub__stencil_1dPKiPi(!llvm.ptr, i64, i32, i64, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func local_unnamed_addr @__mlir_launch_kernel__Z25__device_stub__stencil_1dPKiPi(!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]}
}

// CHECK-LABEL:   gpu.module @__mlir_gpu_module [#[[?]]<chip = "sm_80">]  {
// CHECK:           llvm.mlir.global internal unnamed_addr @_ZZ10stencil_1dPKiPiE4temp() {addr_space = 3 : i32, alignment = 4 : i64, dso_local} : !llvm.array<270 x i32> {
// CHECK:             %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<270 x i32>
// CHECK:             llvm.return %[[VAL_0]] : !llvm.array<270 x i32>
// CHECK:           }
// CHECK:           llvm.func @__mlir.par.kernel._Z10stencil_1dPKiPi(%[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64, %[[VAL_6:.*]]: i64, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_9:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {gpu.kernel, gpu.par.kernel} {
// CHECK:             %[[VAL_10:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:             %[[VAL_11:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_4]] : i64 to index
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK:             %[[VAL_14:.*]] = arith.index_cast %[[VAL_2]] : i64 to index
// CHECK:             %[[VAL_15:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:             affine.parallel (%[[VAL_16:.*]]) = (0) to (%[[VAL_15]]) {
// CHECK:               affine.parallel (%[[VAL_17:.*]]) = (0) to (%[[VAL_14]]) {
// CHECK:                 affine.parallel (%[[VAL_18:.*]]) = (0) to (%[[VAL_13]]) {
// CHECK:                   %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:                   %[[VAL_20:.*]] = llvm.alloca %[[VAL_19]] x !llvm.array<270 x i32> : (i32) -> !llvm.ptr<3>
// CHECK:                   affine.parallel (%[[VAL_21:.*]]) = (0) to (%[[VAL_12]]) {
// CHECK:                     affine.parallel (%[[VAL_22:.*]]) = (0) to (%[[VAL_11]]) {
// CHECK:                       affine.parallel (%[[VAL_23:.*]]) = (0) to (%[[VAL_10]]) {
// CHECK:                         %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:                         %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK:                         %[[VAL_26:.*]] = arith.constant 8 : i32
// CHECK:                         %[[VAL_27:.*]] = arith.constant -7 : i32
// CHECK:                         %[[VAL_28:.*]] = arith.constant 263 : i32
// CHECK:                         %[[VAL_29:.*]] = arith.constant 7 : i32
// CHECK:                         %[[VAL_30:.*]] = llvm.addrspacecast %[[VAL_20]] : !llvm.ptr<3> to !llvm.ptr
// CHECK:                         %[[VAL_31:.*]] = arith.index_cast %[[VAL_21]] : index to i32
// CHECK:                         %[[VAL_32:.*]] = arith.index_cast %[[VAL_16]] : index to i32
// CHECK:                         %[[VAL_33:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:                         %[[VAL_34:.*]] = arith.muli %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:                         %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_31]] : i32
// CHECK:                         %[[VAL_36:.*]] = arith.addi %[[VAL_31]], %[[VAL_29]] : i32
// CHECK:                         %[[VAL_37:.*]] = arith.extsi %[[VAL_35]] : i32 to i64
// CHECK:                         %[[VAL_38:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_37]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                         %[[VAL_39:.*]] = llvm.load %[[VAL_38]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                         %[[VAL_40:.*]] = arith.extui %[[VAL_36]] : i32 to i64
// CHECK:                         %[[VAL_41:.*]] = llvm.getelementptr inbounds %[[VAL_30]][0, %[[VAL_40]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:                         llvm.store %[[VAL_39]], %[[VAL_41]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:                         %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_31]], %[[VAL_29]] : i32
// CHECK:                         scf.if %[[VAL_42]] {
// CHECK:                           %[[VAL_43:.*]] = arith.cmpi slt, %[[VAL_35]], %[[VAL_29]] : i32
// CHECK:                           %[[VAL_44:.*]] = scf.if %[[VAL_43]] -> (i32) {
// CHECK:                             scf.yield %[[VAL_25]] : i32
// CHECK:                           } else {
// CHECK:                             %[[VAL_45:.*]] = arith.extui %[[VAL_35]] : i32 to i64
// CHECK:                             %[[VAL_46:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_45]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                             %[[VAL_47:.*]] = llvm.getelementptr %[[VAL_46]][-28] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:                             %[[VAL_48:.*]] = llvm.load %[[VAL_47]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                             scf.yield %[[VAL_48]] : i32
// CHECK:                           }
// CHECK:                           %[[VAL_49:.*]] = arith.extui %[[VAL_31]] : i32 to i64
// CHECK:                           %[[VAL_50:.*]] = llvm.getelementptr inbounds %[[VAL_30]][0, %[[VAL_49]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:                           llvm.store %[[VAL_44]], %[[VAL_50]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:                           %[[VAL_51:.*]] = llvm.getelementptr %[[VAL_38]][1024] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:                           %[[VAL_52:.*]] = llvm.load %[[VAL_51]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                           %[[VAL_53:.*]] = arith.addi %[[VAL_31]], %[[VAL_28]] : i32
// CHECK:                           %[[VAL_54:.*]] = arith.extui %[[VAL_53]] : i32 to i64
// CHECK:                           %[[VAL_55:.*]] = llvm.getelementptr inbounds %[[VAL_30]][0, %[[VAL_54]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:                           llvm.store %[[VAL_52]], %[[VAL_55]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:                         }
// CHECK:                         nvvm.barrier0
// CHECK:                         %[[VAL_56:.*]] = scf.for %[[VAL_57:.*]] = %[[VAL_27]] to %[[VAL_26]] step %[[VAL_24]] iter_args(%[[VAL_58:.*]] = %[[VAL_25]]) -> (i32)  : i32 {
// CHECK:                           %[[VAL_59:.*]] = arith.addi %[[VAL_57]], %[[VAL_36]] : i32
// CHECK:                           %[[VAL_60:.*]] = arith.extsi %[[VAL_59]] : i32 to i64
// CHECK:                           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_30]][0, %[[VAL_60]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:                           %[[VAL_62:.*]] = llvm.load %[[VAL_61]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                           %[[VAL_63:.*]] = arith.addi %[[VAL_62]], %[[VAL_58]] : i32
// CHECK:                           scf.yield %[[VAL_63]] : i32
// CHECK:                         }
// CHECK:                         %[[VAL_64:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_37]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                         llvm.store %[[VAL_56]], %[[VAL_64]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:                       } {gpu.par.block}
// CHECK:                     } {gpu.par.block}
// CHECK:                   } {gpu.par.block}
// CHECK:                 } {gpu.par.grid}
// CHECK:               } {gpu.par.grid}
// CHECK:             } {gpu.par.grid}
// CHECK:             llvm.return
// CHECK:           }
// CHECK:           llvm.func local_unnamed_addr @_Z10stencil_1dPKiPi(%[[VAL_65:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_66:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #[[?]]<all>, gpu.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_cpu = "sm_52", target_features = #[[?]]<["+ptx84", "+sm_52"]>} {
// CHECK:             %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_68:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_69:.*]] = arith.constant 8 : i32
// CHECK:             %[[VAL_70:.*]] = arith.constant -7 : i32
// CHECK:             %[[VAL_71:.*]] = arith.constant 263 : i32
// CHECK:             %[[VAL_72:.*]] = arith.constant 7 : i32
// CHECK:             %[[VAL_73:.*]] = llvm.mlir.addressof @_ZZ10stencil_1dPKiPiE4temp : !llvm.ptr<3>
// CHECK:             %[[VAL_74:.*]] = llvm.addrspacecast %[[VAL_73]] : !llvm.ptr<3> to !llvm.ptr
// CHECK:             %[[VAL_75:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:             %[[VAL_76:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:             %[[VAL_77:.*]] = nvvm.read.ptx.sreg.ntid.x : i32
// CHECK:             %[[VAL_78:.*]] = arith.muli %[[VAL_76]], %[[VAL_77]] : i32
// CHECK:             %[[VAL_79:.*]] = arith.addi %[[VAL_78]], %[[VAL_75]] : i32
// CHECK:             %[[VAL_80:.*]] = arith.addi %[[VAL_75]], %[[VAL_72]] : i32
// CHECK:             %[[VAL_81:.*]] = arith.extsi %[[VAL_79]] : i32 to i64
// CHECK:             %[[VAL_82:.*]] = llvm.getelementptr inbounds %[[VAL_65]]{{\[}}%[[VAL_81]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:             %[[VAL_83:.*]] = llvm.load %[[VAL_82]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:             %[[VAL_84:.*]] = arith.extui %[[VAL_80]] : i32 to i64
// CHECK:             %[[VAL_85:.*]] = llvm.getelementptr inbounds %[[VAL_74]][0, %[[VAL_84]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:             llvm.store %[[VAL_83]], %[[VAL_85]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:             %[[VAL_86:.*]] = arith.cmpi ult, %[[VAL_75]], %[[VAL_72]] : i32
// CHECK:             scf.if %[[VAL_86]] {
// CHECK:               %[[VAL_87:.*]] = arith.cmpi slt, %[[VAL_79]], %[[VAL_72]] : i32
// CHECK:               %[[VAL_88:.*]] = scf.if %[[VAL_87]] -> (i32) {
// CHECK:                 scf.yield %[[VAL_68]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_89:.*]] = arith.extui %[[VAL_79]] : i32 to i64
// CHECK:                 %[[VAL_90:.*]] = llvm.getelementptr %[[VAL_65]]{{\[}}%[[VAL_89]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_91:.*]] = llvm.getelementptr %[[VAL_90]][-28] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:                 %[[VAL_92:.*]] = llvm.load %[[VAL_91]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                 scf.yield %[[VAL_92]] : i32
// CHECK:               }
// CHECK:               %[[VAL_93:.*]] = arith.extui %[[VAL_75]] : i32 to i64
// CHECK:               %[[VAL_94:.*]] = llvm.getelementptr inbounds %[[VAL_74]][0, %[[VAL_93]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               llvm.store %[[VAL_88]], %[[VAL_94]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:               %[[VAL_95:.*]] = llvm.getelementptr %[[VAL_82]][1024] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:               %[[VAL_96:.*]] = llvm.load %[[VAL_95]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:               %[[VAL_97:.*]] = arith.addi %[[VAL_75]], %[[VAL_71]] : i32
// CHECK:               %[[VAL_98:.*]] = arith.extui %[[VAL_97]] : i32 to i64
// CHECK:               %[[VAL_99:.*]] = llvm.getelementptr inbounds %[[VAL_74]][0, %[[VAL_98]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               llvm.store %[[VAL_96]], %[[VAL_99]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:             }
// CHECK:             nvvm.barrier0
// CHECK:             %[[VAL_100:.*]] = scf.for %[[VAL_101:.*]] = %[[VAL_70]] to %[[VAL_69]] step %[[VAL_67]] iter_args(%[[VAL_102:.*]] = %[[VAL_68]]) -> (i32)  : i32 {
// CHECK:               %[[VAL_103:.*]] = arith.addi %[[VAL_101]], %[[VAL_80]] : i32
// CHECK:               %[[VAL_104:.*]] = arith.extsi %[[VAL_103]] : i32 to i64
// CHECK:               %[[VAL_105:.*]] = llvm.getelementptr inbounds %[[VAL_74]][0, %[[VAL_104]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
// CHECK:               %[[VAL_106:.*]] = llvm.load %[[VAL_105]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:               %[[VAL_107:.*]] = arith.addi %[[VAL_106]], %[[VAL_102]] : i32
// CHECK:               scf.yield %[[VAL_107]] : i32
// CHECK:             }
// CHECK:             %[[VAL_108:.*]] = llvm.getelementptr inbounds %[[VAL_66]]{{\[}}%[[VAL_81]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:             llvm.store %[[VAL_100]], %[[VAL_108]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str"("Usage: %[[VAL_109:.*]] <length> <repeat>\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.1"("length is a multiple of %[[VAL_110:.*]]\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global internal unnamed_addr @_ZZ4mainE4done(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.2"("cuda stencil1d-cuda stencil_1d.cu\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.3"("0\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.4"("Average kernel execution time: %[[VAL_111:.*]] (s)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.5"("Error at %[[VAL_110]]: %[[VAL_110]] (host) != %[[VAL_110]] (device)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.7"("PASS\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.8"("FAIL\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.9"("MY_TIMING_FILE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global external local_unnamed_addr @stdout() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.12"("a\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global external local_unnamed_addr @stderr() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.13"("Could not open timing file %[[VAL_109]], errno %[[VAL_110]], %[[VAL_109]]\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.14"("HOSTNAME\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.15"("unidetified_compiler\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK:         llvm.mlir.global private unnamed_addr constant @".str.16"("%[[VAL_109]],%[[VAL_109]],%[[VAL_112:.*]],%[[VAL_109]],%[[VAL_109]],%[[VAL_110]],\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

// CHECK-LABEL:   llvm.mlir.global internal unnamed_addr @__cuda_gpubin_handle() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @_Z25__device_stub__stencil_1dPKiPi(
// CHECK-SAME:                                                  %[[VAL_0:[^:]*]]: !llvm.ptr {llvm.noalias, llvm.noundef},
// CHECK-SAME:                                                  %[[VAL_1:[^:]*]]: !llvm.ptr {llvm.noalias, llvm.noundef}) attributes {passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["uniform-work-group-size", "true"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @_Z25__device_stub__stencil_1dPKiPi : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"struct.dim3", (i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"struct.dim3", (i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_2]] x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.call @__cudaPopCallConfiguration(%[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_6]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_7]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_4]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_4]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] {alignment = 8 : i64} : !llvm.ptr -> i32
// CHECK:           %[[VAL_14:.*]] = llvm.load %[[VAL_5]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_5]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:           %[[VAL_16:.*]] = llvm.load %[[VAL_15]] {alignment = 8 : i64} : !llvm.ptr -> i32
// CHECK:           llvm.call @__mlir_launch_coerced_kernel__Z25__device_stub__stencil_1dPKiPi(%[[VAL_3]], %[[VAL_11]], %[[VAL_13]], %[[VAL_14]], %[[VAL_16]], %[[VAL_9]], %[[VAL_10]], %[[VAL_0]], %[[VAL_1]]) : (!llvm.ptr, i64, i32, i64, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func local_unnamed_addr @__cudaPopCallConfiguration(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL:   llvm.func local_unnamed_addr @main(
// CHECK-SAME:                                       %[[VAL_0:[^:]*]]: i32 {llvm.noundef},
// CHECK-SAME:                                       %[[VAL_1:[^:]*]]: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK:           %[[VAL_2:.*]] = ub.poison : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = ub.poison : i64
// CHECK:           %[[VAL_5:.*]] = ub.poison : !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.addressof @".str.1" : !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_13:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 7 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.addressof @_Z25__device_stub__stencil_1dPKiPi : !llvm.ptr
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.addressof @_ZZ4mainE4done : !llvm.ptr
// CHECK:           %[[VAL_19:.*]] = arith.constant true
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(1.000000e-09 : f64) : f64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.addressof @".str.9" : !llvm.ptr
// CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.addressof @".str.12" : !llvm.ptr
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.addressof @stdout : !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.addressof @".str.14" : !llvm.ptr
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.addressof @".str.16" : !llvm.ptr
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.addressof @".str.2" : !llvm.ptr
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @".str.3" : !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.addressof @".str.15" : !llvm.ptr
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.addressof @stderr : !llvm.ptr
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.addressof @".str.13" : !llvm.ptr
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(9.99999971E-10 : f32) : f32
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.addressof @".str.4" : !llvm.ptr
// CHECK:           %[[VAL_34:.*]] = arith.constant 15 : i64
// CHECK:           %[[VAL_35:.*]] = arith.constant 14 : i64
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.addressof @".str.7" : !llvm.ptr
// CHECK:           %[[VAL_37:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_38:.*]] = arith.constant -7 : i32
// CHECK:           %[[VAL_39:.*]] = arith.constant 13 : i64
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.addressof @".str.5" : !llvm.ptr
// CHECK:           %[[VAL_41:.*]] = llvm.mlir.addressof @".str.8" : !llvm.ptr
// CHECK:           %[[VAL_42:.*]] = arith.constant 14 : i32
// CHECK:           %[[VAL_43:.*]] = llvm.alloca %[[VAL_6]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_44:.*]] = llvm.alloca %[[VAL_6]] x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_45:.*]] = llvm.alloca %[[VAL_6]] x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_46:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"struct.timespec", (i64, i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_47:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"struct.timespec", (i64, i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_49:.*]]:2 = scf.if %[[VAL_48]] -> (i32, i32) {
// CHECK:             %[[VAL_50:.*]] = llvm.getelementptr inbounds %[[VAL_1]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:             %[[VAL_51:.*]] = llvm.load %[[VAL_50]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:             %[[VAL_52:.*]] = llvm.call tail @__isoc23_strtol(%[[VAL_51]], %[[VAL_12]], %[[VAL_13]]) {no_unwind} : (!llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK:             %[[VAL_53:.*]] = arith.trunci %[[VAL_52]] : i64 to i32
// CHECK:             %[[VAL_54:.*]] = llvm.getelementptr inbounds %[[VAL_1]][16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:             %[[VAL_55:.*]] = llvm.load %[[VAL_54]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:             %[[VAL_56:.*]] = llvm.call tail @__isoc23_strtol(%[[VAL_55]], %[[VAL_12]], %[[VAL_13]]) {no_unwind} : (!llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK:             %[[VAL_57:.*]] = arith.shli %[[VAL_53]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_58:.*]] = arith.addi %[[VAL_53]], %[[VAL_14]] : i32
// CHECK:             %[[VAL_59:.*]] = arith.shli %[[VAL_58]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_60:.*]] = arith.extsi %[[VAL_59]] : i32 to i64
// CHECK:             %[[VAL_61:.*]] = llvm.call tail @malloc(%[[VAL_60]]) {memory_effects = #[[?]]<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, will_return} : (i64) -> !llvm.ptr
// CHECK:             %[[VAL_62:.*]] = arith.extsi %[[VAL_57]] : i32 to i64
// CHECK:             %[[VAL_63:.*]] = llvm.call tail @malloc(%[[VAL_62]]) {memory_effects = #[[?]]<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, will_return} : (i64) -> !llvm.ptr
// CHECK:             %[[VAL_64:.*]] = arith.maxsi %[[VAL_58]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_65:.*]] = arith.extui %[[VAL_64]] : i32 to i64
// CHECK:             scf.for %[[VAL_66:.*]] = %[[VAL_15]] to %[[VAL_65]] step %[[VAL_16]]  : i64 {
// CHECK:               %[[VAL_67:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}%[[VAL_66]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:               %[[VAL_68:.*]] = arith.trunci %[[VAL_66]] : i64 to i32
// CHECK:               llvm.store %[[VAL_68]], %[[VAL_67]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : i32, !llvm.ptr
// CHECK:             }
// CHECK:             %[[VAL_69:.*]] = arith.trunci %[[VAL_56]] : i64 to i32
// CHECK:             llvm.intr.lifetime.start 8, %[[VAL_44]] : !llvm.ptr
// CHECK:             llvm.intr.lifetime.start 8, %[[VAL_45]] : !llvm.ptr
// CHECK:             %[[VAL_70:.*]] = llvm.call @cudaMalloc(%[[VAL_44]], %[[VAL_60]]) : (!llvm.ptr, i64) -> i32
// CHECK:             %[[VAL_71:.*]] = llvm.call @cudaMalloc(%[[VAL_45]], %[[VAL_62]]) : (!llvm.ptr, i64) -> i32
// CHECK:             %[[VAL_72:.*]] = llvm.load %[[VAL_44]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:             %[[VAL_73:.*]] = llvm.call @cudaMemcpy(%[[VAL_72]], %[[VAL_61]], %[[VAL_60]], %[[VAL_6]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:             %[[VAL_74:.*]] = arith.divsi %[[VAL_53]], %[[VAL_11]] : i32
// CHECK:             %[[VAL_75:.*]] = llvm.call @cudaDeviceSynchronize() : () -> i32
// CHECK:             %[[VAL_76:.*]] = llvm.call @_ZNSt6chrono3_V212steady_clock3nowEv() {no_unwind} : () -> i64
// CHECK:             llvm.intr.lifetime.start 16, %[[VAL_46]] : !llvm.ptr
// CHECK:             llvm.intr.lifetime.start 16, %[[VAL_47]] : !llvm.ptr
// CHECK:             %[[VAL_77:.*]] = llvm.call @cudaDeviceSynchronize() : () -> i32
// CHECK:             %[[VAL_78:.*]] = llvm.call @clock_gettime(%[[VAL_6]], %[[VAL_46]]) {no_unwind} : (i32, !llvm.ptr) -> i32
// CHECK:             %[[VAL_79:.*]] = arith.maxsi %[[VAL_69]], %[[VAL_7]] : i32
// CHECK:             scf.for %[[VAL_80:.*]] = %[[VAL_7]] to %[[VAL_79]] step %[[VAL_6]]  : i32 {
// CHECK:               %[[VAL_81:.*]] = llvm.load %[[VAL_44]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_82:.*]] = llvm.load %[[VAL_45]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_83:.*]] = llvm.sext %[[VAL_74]] : i32 to i64
// CHECK:               %[[VAL_84:.*]] = llvm.sext %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_85:.*]] = llvm.sext %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_86:.*]] = llvm.sext %[[VAL_11]] : i32 to i64
// CHECK:               %[[VAL_87:.*]] = llvm.sext %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_88:.*]] = llvm.sext %[[VAL_6]] : i32 to i64
// CHECK:               gpu.launch_func  @__mlir_gpu_module::@__mlir.par.kernel._Z10stencil_1dPKiPi blocks in (%[[VAL_83]], %[[VAL_84]], %[[VAL_85]]) threads in (%[[VAL_86]], %[[VAL_87]], %[[VAL_88]]) : i64 dynamic_shared_memory_size %[[VAL_7]] args(%[[VAL_81]] : !llvm.ptr, %[[VAL_82]] : !llvm.ptr)
// CHECK:             }
// CHECK:             %[[VAL_89:.*]] = llvm.call @cudaDeviceSynchronize() : () -> i32
// CHECK:             %[[VAL_90:.*]] = llvm.call @_ZNSt6chrono3_V212steady_clock3nowEv() {no_unwind} : () -> i64
// CHECK:             %[[VAL_91:.*]] = arith.subi %[[VAL_90]], %[[VAL_76]] : i64
// CHECK:             %[[VAL_92:.*]] = llvm.call @cudaDeviceSynchronize() : () -> i32
// CHECK:             %[[VAL_93:.*]] = llvm.load %[[VAL_18]] {alignment = 4 : i64} : !llvm.ptr -> i1
// CHECK:             %[[VAL_94:.*]] = scf.if %[[VAL_93]] -> (i32) {
// CHECK:               scf.yield %[[VAL_7]] : i32
// CHECK:             } else {
// CHECK:               llvm.store %[[VAL_19]], %[[VAL_18]] {alignment = 4 : i64} : i1, !llvm.ptr
// CHECK:               %[[VAL_95:.*]] = llvm.call @clock_gettime(%[[VAL_6]], %[[VAL_47]]) {no_unwind} : (i32, !llvm.ptr) -> i32
// CHECK:               %[[VAL_96:.*]] = llvm.load %[[VAL_47]] {alignment = 8 : i64, tbaa = [#[[$ATTR_8]]]} : !llvm.ptr -> i64
// CHECK:               %[[VAL_97:.*]] = llvm.load %[[VAL_46]] {alignment = 8 : i64, tbaa = [#[[$ATTR_8]]]} : !llvm.ptr -> i64
// CHECK:               %[[VAL_98:.*]] = arith.subi %[[VAL_96]], %[[VAL_97]] : i64
// CHECK:               %[[VAL_99:.*]] = llvm.getelementptr inbounds %[[VAL_47]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:               %[[VAL_100:.*]] = llvm.load %[[VAL_99]] {alignment = 8 : i64, tbaa = [#[[$ATTR_9]]]} : !llvm.ptr -> i64
// CHECK:               %[[VAL_101:.*]] = llvm.getelementptr inbounds %[[VAL_46]][8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:               %[[VAL_102:.*]] = llvm.load %[[VAL_101]] {alignment = 8 : i64, tbaa = [#[[$ATTR_9]]]} : !llvm.ptr -> i64
// CHECK:               %[[VAL_103:.*]] = arith.subi %[[VAL_100]], %[[VAL_102]] : i64
// CHECK:               %[[VAL_104:.*]] = arith.sitofp %[[VAL_98]] : i64 to f64
// CHECK:               %[[VAL_105:.*]] = arith.sitofp %[[VAL_103]] : i64 to f64
// CHECK:               %[[VAL_106:.*]] = llvm.intr.fmuladd(%[[VAL_105]], %[[VAL_20]], %[[VAL_104]])  : (f64, f64, f64) -> f64
// CHECK:               %[[VAL_107:.*]] = llvm.call @getenv(%[[VAL_21]]) {memory_effects = #[[?]]<other = read, argMem = read, inaccessibleMem = read>, no_unwind} : (!llvm.ptr) -> !llvm.ptr
// CHECK:               %[[VAL_108:.*]] = llvm.ptrtoint %[[VAL_107]] : !llvm.ptr to i64
// CHECK:               %[[VAL_109:.*]] = llvm.ptrtoint %[[VAL_12]] : !llvm.ptr to i64
// CHECK:               %[[VAL_110:.*]] = arith.cmpi eq, %[[VAL_108]], %[[VAL_109]] : i64
// CHECK:               %[[VAL_111:.*]]:2 = scf.if %[[VAL_110]] -> (!llvm.ptr, i32) {
// CHECK:                 scf.yield %[[VAL_5]], %[[VAL_7]] : !llvm.ptr, i32
// CHECK:               } else {
// CHECK:                 %[[VAL_112:.*]] = llvm.load %[[VAL_107]] {alignment = 1 : i64} : !llvm.ptr -> i8
// CHECK:                 %[[VAL_113:.*]] = arith.index_castui %[[VAL_112]] : i8 to index
// CHECK:                 %[[VAL_114:.*]]:3 = scf.index_switch %[[VAL_113]] -> !llvm.ptr, i32, i32
// CHECK:                 case 0 {
// CHECK:                   scf.yield %[[VAL_5]], %[[VAL_7]], %[[VAL_6]] : !llvm.ptr, i32, i32
// CHECK:                 }
// CHECK:                 case 45 {
// CHECK:                   %[[VAL_115:.*]] = llvm.getelementptr inbounds %[[VAL_107]][1] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:                   %[[VAL_116:.*]] = llvm.load %[[VAL_115]] {alignment = 1 : i64} : !llvm.ptr -> i8
// CHECK:                   %[[VAL_117:.*]] = arith.cmpi eq, %[[VAL_116]], %[[VAL_22]] : i8
// CHECK:                   %[[VAL_118:.*]] = arith.extui %[[VAL_117]] : i1 to i32
// CHECK:                   scf.yield %[[VAL_5]], %[[VAL_7]], %[[VAL_118]] : !llvm.ptr, i32, i32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_5]], %[[VAL_2]], %[[VAL_7]] : !llvm.ptr, i32, i32
// CHECK:                 }
// CHECK:                 %[[VAL_119:.*]] = arith.index_castui %[[VAL_114]]#2 : i32 to index
// CHECK:                 %[[VAL_120:.*]]:2 = scf.index_switch %[[VAL_119]] -> !llvm.ptr, i32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_121:.*]] = llvm.call @fopen(%[[VAL_107]], %[[VAL_23]]) {no_unwind} : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK:                   scf.yield %[[VAL_121]], %[[VAL_6]] : !llvm.ptr, i32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_114]]#0, %[[VAL_114]]#1 : !llvm.ptr, i32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_120]]#0, %[[VAL_120]]#1 : !llvm.ptr, i32
// CHECK:               }
// CHECK:               %[[VAL_122:.*]] = arith.index_castui %[[VAL_123:.*]]#1 : i32 to index
// CHECK:               %[[VAL_124:.*]] = scf.index_switch %[[VAL_122]] -> !llvm.ptr
// CHECK:               case 0 {
// CHECK:                 %[[VAL_125:.*]] = llvm.load %[[VAL_24]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:                 scf.yield %[[VAL_125]] : !llvm.ptr
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_123]]#0 : !llvm.ptr
// CHECK:               }
// CHECK:               %[[VAL_126:.*]] = llvm.ptrtoint %[[VAL_124]] : !llvm.ptr to i64
// CHECK:               %[[VAL_127:.*]] = llvm.ptrtoint %[[VAL_12]] : !llvm.ptr to i64
// CHECK:               %[[VAL_128:.*]] = arith.cmpi eq, %[[VAL_126]], %[[VAL_127]] : i64
// CHECK:               %[[VAL_129:.*]] = arith.extui %[[VAL_128]] : i1 to i32
// CHECK:               scf.if %[[VAL_128]] {
// CHECK:                 %[[VAL_130:.*]] = llvm.load %[[VAL_30]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:                 %[[VAL_131:.*]] = llvm.call tail @__errno_location() {memory_effects = #[[?]]<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : () -> !llvm.ptr
// CHECK:                 %[[VAL_132:.*]] = llvm.load %[[VAL_131]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                 %[[VAL_133:.*]] = llvm.call @strerror(%[[VAL_132]]) {no_unwind} : (i32) -> !llvm.ptr
// CHECK:                 %[[VAL_134:.*]] = llvm.call @fprintf(%[[VAL_130]], %[[VAL_31]], %[[VAL_107]], %[[VAL_132]], %[[VAL_133]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:                 llvm.call @exit(%[[VAL_6]]) {no_unwind} : (i32) -> ()
// CHECK:               } else {
// CHECK:                 llvm.intr.lifetime.start 1, %[[VAL_43]] : !llvm.ptr
// CHECK:                 llvm.store %[[VAL_22]], %[[VAL_43]] {alignment = 1 : i64} : i8, !llvm.ptr
// CHECK:                 %[[VAL_135:.*]] = llvm.call @getenv(%[[VAL_25]]) {memory_effects = #[[?]]<other = read, argMem = read, inaccessibleMem = read>, no_unwind} : (!llvm.ptr) -> !llvm.ptr
// CHECK:                 %[[VAL_136:.*]] = llvm.ptrtoint %[[VAL_135]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_137:.*]] = llvm.ptrtoint %[[VAL_12]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_138:.*]] = arith.cmpi eq, %[[VAL_136]], %[[VAL_137]] : i64
// CHECK:                 %[[VAL_139:.*]] = arith.select %[[VAL_138]], %[[VAL_43]], %[[VAL_135]] {fastmathFlags = #[[?]]<none>} : !llvm.ptr
// CHECK:                 %[[VAL_140:.*]] = llvm.call @fprintf(%[[VAL_124]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_106]], %[[VAL_139]], %[[VAL_29]], %[[VAL_6]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, !llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:                 %[[VAL_141:.*]] = llvm.load %[[VAL_24]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:                 %[[VAL_142:.*]] = llvm.ptrtoint %[[VAL_124]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_143:.*]] = llvm.ptrtoint %[[VAL_141]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_144:.*]] = arith.cmpi eq, %[[VAL_142]], %[[VAL_143]] : i64
// CHECK:                 %[[VAL_145:.*]] = llvm.load %[[VAL_30]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
// CHECK:                 %[[VAL_146:.*]] = llvm.ptrtoint %[[VAL_124]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_147:.*]] = llvm.ptrtoint %[[VAL_145]] : !llvm.ptr to i64
// CHECK:                 %[[VAL_148:.*]] = arith.cmpi eq, %[[VAL_146]], %[[VAL_147]] : i64
// CHECK:                 %[[VAL_149:.*]] = arith.select %[[VAL_144]], %[[VAL_19]], %[[VAL_148]] {fastmathFlags = #[[?]]<none>} : i1
// CHECK:                 scf.if %[[VAL_149]] {
// CHECK:                 } else {
// CHECK:                   %[[VAL_150:.*]] = llvm.call @fclose(%[[VAL_124]]) {no_unwind} : (!llvm.ptr) -> i32
// CHECK:                 }
// CHECK:                 llvm.intr.lifetime.end 1, %[[VAL_43]] : !llvm.ptr
// CHECK:               }
// CHECK:               scf.yield %[[VAL_129]] : i32
// CHECK:             }
// CHECK:             %[[VAL_151:.*]] = arith.index_castui %[[VAL_94]] : i32 to index
// CHECK:             %[[VAL_152:.*]]:2 = scf.index_switch %[[VAL_151]] -> i32, i32
// CHECK:             case 0 {
// CHECK:               %[[VAL_153:.*]] = arith.sitofp %[[VAL_91]] : i64 to f32
// CHECK:               %[[VAL_154:.*]] = llvm.fmul %[[VAL_153]], %[[VAL_32]]  : f32
// CHECK:               %[[VAL_155:.*]] = arith.sitofp %[[VAL_69]] : i32 to f32
// CHECK:               %[[VAL_156:.*]] = llvm.fdiv %[[VAL_154]], %[[VAL_155]]  : f32
// CHECK:               %[[VAL_157:.*]] = llvm.fpext %[[VAL_156]] : f32 to f64
// CHECK:               %[[VAL_158:.*]] = llvm.call @printf(%[[VAL_33]], %[[VAL_157]]) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, f64) -> i32
// CHECK:               %[[VAL_159:.*]] = llvm.load %[[VAL_45]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_160:.*]] = llvm.call @cudaMemcpy(%[[VAL_63]], %[[VAL_159]], %[[VAL_62]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:               %[[VAL_161:.*]]:6 = scf.while (%[[VAL_162:.*]] = %[[VAL_34]], %[[VAL_163:.*]] = %[[VAL_15]]) : (i64, i64) -> (i64, i64, i32, i32, i64, i32) {
// CHECK:                 %[[VAL_164:.*]] = arith.cmpi eq, %[[VAL_163]], %[[VAL_35]] : i64
// CHECK:                 %[[VAL_165:.*]]:6 = scf.if %[[VAL_164]] -> (i64, i64, i32, i32, i32, i32) {
// CHECK:                   scf.yield %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_7]], %[[VAL_2]], %[[VAL_2]] : i64, i64, i32, i32, i32, i32
// CHECK:                 } else {
// CHECK:                   %[[VAL_166:.*]] = scf.for %[[VAL_167:.*]] = %[[VAL_163]] to %[[VAL_162]] step %[[VAL_16]] iter_args(%[[VAL_168:.*]] = %[[VAL_7]]) -> (i32)  : i64 {
// CHECK:                     %[[VAL_169:.*]] = arith.cmpi ult, %[[VAL_167]], %[[VAL_37]] : i64
// CHECK:                     %[[VAL_170:.*]] = scf.if %[[VAL_169]] -> (i32) {
// CHECK:                       scf.yield %[[VAL_7]] : i32
// CHECK:                     } else {
// CHECK:                       %[[VAL_171:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}%[[VAL_167]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                       %[[VAL_172:.*]] = llvm.load %[[VAL_171]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                       %[[VAL_173:.*]] = arith.addi %[[VAL_172]], %[[VAL_38]] : i32
// CHECK:                       scf.yield %[[VAL_173]] : i32
// CHECK:                     }
// CHECK:                     %[[VAL_174:.*]] = arith.addi %[[VAL_170]], %[[VAL_168]] : i32
// CHECK:                     scf.yield %[[VAL_174]] : i32
// CHECK:                   }
// CHECK:                   %[[VAL_175:.*]] = llvm.getelementptr inbounds %[[VAL_63]]{{\[}}%[[VAL_163]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                   %[[VAL_176:.*]] = llvm.load %[[VAL_175]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                   %[[VAL_177:.*]] = arith.cmpi eq, %[[VAL_166]], %[[VAL_176]] : i32
// CHECK:                   %[[VAL_178:.*]] = arith.cmpi ne, %[[VAL_166]], %[[VAL_176]] : i32
// CHECK:                   %[[VAL_179:.*]] = arith.extui %[[VAL_178]] : i1 to i32
// CHECK:                   %[[VAL_180:.*]] = arith.extui %[[VAL_177]] : i1 to i32
// CHECK:                   %[[VAL_181:.*]]:2 = scf.if %[[VAL_177]] -> (i64, i64) {
// CHECK:                     %[[VAL_182:.*]] = arith.addi %[[VAL_163]], %[[VAL_16]] : i64
// CHECK:                     %[[VAL_183:.*]] = arith.addi %[[VAL_162]], %[[VAL_16]] : i64
// CHECK:                     scf.yield %[[VAL_183]], %[[VAL_182]] : i64, i64
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_4]], %[[VAL_4]] : i64, i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_184:.*]]#0, %[[VAL_184]]#1, %[[VAL_179]], %[[VAL_180]], %[[VAL_176]], %[[VAL_166]] : i64, i64, i32, i32, i32, i32
// CHECK:                 }
// CHECK:                 %[[VAL_185:.*]] = arith.trunci %[[VAL_186:.*]]#3 : i32 to i1
// CHECK:                 scf.condition(%[[VAL_185]]) %[[VAL_186]]#0, %[[VAL_186]]#1, %[[VAL_186]]#4, %[[VAL_186]]#5, %[[VAL_163]], %[[VAL_186]]#2 : i64, i64, i32, i32, i64, i32
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_187:.*]]: i64, %[[VAL_188:.*]]: i64, %[[VAL_189:.*]]: i32, %[[VAL_190:.*]]: i32, %[[VAL_191:.*]]: i64, %[[VAL_192:.*]]: i32):
// CHECK:                 scf.yield %[[VAL_187]], %[[VAL_188]] : i64, i64
// CHECK:               }
// CHECK:               %[[VAL_193:.*]] = arith.index_castui %[[VAL_194:.*]]#5 : i32 to index
// CHECK:               %[[VAL_195:.*]] = scf.index_switch %[[VAL_193]] -> !llvm.ptr
// CHECK:               case 1 {
// CHECK:                 %[[VAL_196:.*]] = arith.cmpi ugt, %[[VAL_194]]#4, %[[VAL_39]] : i64
// CHECK:                 %[[VAL_197:.*]] = arith.trunci %[[VAL_194]]#4 : i64 to i32
// CHECK:                 %[[VAL_198:.*]] = llvm.call @printf(%[[VAL_40]], %[[VAL_197]], %[[VAL_194]]#3, %[[VAL_194]]#2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32, i32) -> i32
// CHECK:                 %[[VAL_199:.*]] = arith.select %[[VAL_196]], %[[VAL_36]], %[[VAL_41]] {fastmathFlags = #[[?]]<none>} : !llvm.ptr
// CHECK:                 scf.yield %[[VAL_199]] : !llvm.ptr
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_36]] : !llvm.ptr
// CHECK:               }
// CHECK:               %[[VAL_200:.*]] = arith.maxsi %[[VAL_53]], %[[VAL_42]] : i32
// CHECK:               %[[VAL_201:.*]] = arith.extui %[[VAL_200]] : i32 to i64
// CHECK:               %[[VAL_202:.*]]:6 = scf.while (%[[VAL_203:.*]] = %[[VAL_35]], %[[VAL_204:.*]] = %[[VAL_37]]) : (i64, i64) -> (i64, i64, i32, i32, i64, i32) {
// CHECK:                 %[[VAL_205:.*]] = arith.cmpi eq, %[[VAL_203]], %[[VAL_201]] : i64
// CHECK:                 %[[VAL_206:.*]]:6 = scf.if %[[VAL_205]] -> (i64, i64, i32, i32, i32, i32) {
// CHECK:                   scf.yield %[[VAL_4]], %[[VAL_4]], %[[VAL_3]], %[[VAL_7]], %[[VAL_2]], %[[VAL_2]] : i64, i64, i32, i32, i32, i32
// CHECK:                 } else {
// CHECK:                   %[[VAL_207:.*]] = arith.addi %[[VAL_203]], %[[VAL_37]] : i64
// CHECK:                   %[[VAL_208:.*]]:3 = scf.while (%[[VAL_209:.*]] = %[[VAL_204]], %[[VAL_210:.*]] = %[[VAL_7]]) : (i64, i32) -> (i64, i32, i32) {
// CHECK:                     %[[VAL_211:.*]] = arith.cmpi ugt, %[[VAL_209]], %[[VAL_207]] : i64
// CHECK:                     %[[VAL_212:.*]] = arith.cmpi ule, %[[VAL_209]], %[[VAL_207]] : i64
// CHECK:                     %[[VAL_213:.*]]:2 = scf.if %[[VAL_211]] -> (i64, i32) {
// CHECK:                       scf.yield %[[VAL_4]], %[[VAL_2]] : i64, i32
// CHECK:                     } else {
// CHECK:                       %[[VAL_214:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}%[[VAL_209]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                       %[[VAL_215:.*]] = llvm.load %[[VAL_214]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                       %[[VAL_216:.*]] = arith.addi %[[VAL_215]], %[[VAL_210]] : i32
// CHECK:                       %[[VAL_217:.*]] = arith.addi %[[VAL_209]], %[[VAL_16]] : i64
// CHECK:                       scf.yield %[[VAL_217]], %[[VAL_216]] : i64, i32
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_212]]) %[[VAL_218:.*]]#0, %[[VAL_218]]#1, %[[VAL_210]] : i64, i32, i32
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_219:.*]]: i64, %[[VAL_220:.*]]: i32, %[[VAL_221:.*]]: i32):
// CHECK:                     scf.yield %[[VAL_219]], %[[VAL_220]] : i64, i32
// CHECK:                   }
// CHECK:                   %[[VAL_222:.*]] = llvm.getelementptr inbounds %[[VAL_63]]{{\[}}%[[VAL_203]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:                   %[[VAL_223:.*]] = llvm.load %[[VAL_222]] {alignment = 4 : i64, tbaa = [#[[$ATTR_5]]]} : !llvm.ptr -> i32
// CHECK:                   %[[VAL_224:.*]] = arith.cmpi eq, %[[VAL_225:.*]]#2, %[[VAL_223]] : i32
// CHECK:                   %[[VAL_226:.*]] = arith.cmpi ne, %[[VAL_225]]#2, %[[VAL_223]] : i32
// CHECK:                   %[[VAL_227:.*]] = arith.extui %[[VAL_226]] : i1 to i32
// CHECK:                   %[[VAL_228:.*]] = arith.extui %[[VAL_224]] : i1 to i32
// CHECK:                   %[[VAL_229:.*]]:2 = scf.if %[[VAL_224]] -> (i64, i64) {
// CHECK:                     %[[VAL_230:.*]] = arith.addi %[[VAL_203]], %[[VAL_16]] : i64
// CHECK:                     %[[VAL_231:.*]] = arith.addi %[[VAL_204]], %[[VAL_16]] : i64
// CHECK:                     scf.yield %[[VAL_230]], %[[VAL_231]] : i64, i64
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_4]], %[[VAL_4]] : i64, i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_232:.*]]#0, %[[VAL_232]]#1, %[[VAL_227]], %[[VAL_228]], %[[VAL_223]], %[[VAL_225]]#2 : i64, i64, i32, i32, i32, i32
// CHECK:                 }
// CHECK:                 %[[VAL_233:.*]] = arith.trunci %[[VAL_234:.*]]#3 : i32 to i1
// CHECK:                 scf.condition(%[[VAL_233]]) %[[VAL_234]]#0, %[[VAL_234]]#1, %[[VAL_234]]#4, %[[VAL_234]]#5, %[[VAL_203]], %[[VAL_234]]#2 : i64, i64, i32, i32, i64, i32
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_235:.*]]: i64, %[[VAL_236:.*]]: i64, %[[VAL_237:.*]]: i32, %[[VAL_238:.*]]: i32, %[[VAL_239:.*]]: i64, %[[VAL_240:.*]]: i32):
// CHECK:                 scf.yield %[[VAL_235]], %[[VAL_236]] : i64, i64
// CHECK:               }
// CHECK:               %[[VAL_241:.*]] = arith.index_castui %[[VAL_242:.*]]#5 : i32 to index
// CHECK:               %[[VAL_243:.*]] = scf.index_switch %[[VAL_241]] -> !llvm.ptr
// CHECK:               case 1 {
// CHECK:                 %[[VAL_244:.*]] = arith.trunci %[[VAL_242]]#4 : i64 to i32
// CHECK:                 %[[VAL_245:.*]] = llvm.call @printf(%[[VAL_40]], %[[VAL_244]], %[[VAL_242]]#3, %[[VAL_242]]#2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32, i32) -> i32
// CHECK:                 scf.yield %[[VAL_41]] : !llvm.ptr
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_195]] : !llvm.ptr
// CHECK:               }
// CHECK:               %[[VAL_246:.*]] = llvm.call @puts(%[[VAL_243]]) {no_unwind} : (!llvm.ptr) -> i32
// CHECK:               llvm.call @free(%[[VAL_61]]) {memory_effects = #[[?]]<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, will_return} : (!llvm.ptr) -> ()
// CHECK:               llvm.call @free(%[[VAL_63]]) {memory_effects = #[[?]]<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, will_return} : (!llvm.ptr) -> ()
// CHECK:               %[[VAL_247:.*]] = llvm.load %[[VAL_44]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_248:.*]] = llvm.call @cudaFree(%[[VAL_247]]) : (!llvm.ptr) -> i32
// CHECK:               %[[VAL_249:.*]] = llvm.load %[[VAL_45]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:               %[[VAL_250:.*]] = llvm.call @cudaFree(%[[VAL_249]]) : (!llvm.ptr) -> i32
// CHECK:               llvm.intr.lifetime.end 16, %[[VAL_47]] : !llvm.ptr
// CHECK:               llvm.intr.lifetime.end 16, %[[VAL_46]] : !llvm.ptr
// CHECK:               llvm.intr.lifetime.end 8, %[[VAL_45]] : !llvm.ptr
// CHECK:               llvm.intr.lifetime.end 8, %[[VAL_44]] : !llvm.ptr
// CHECK:               scf.yield %[[VAL_7]], %[[VAL_7]] : i32, i32
// CHECK:             }
// CHECK:             default {
// CHECK:               scf.yield %[[VAL_2]], %[[VAL_6]] : i32, i32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_152]]#0, %[[VAL_152]]#1 : i32, i32
// CHECK:           } else {
// CHECK:             %[[VAL_251:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64, tbaa = [#[[$ATTR_6]]]} : !llvm.ptr -> !llvm.ptr
// CHECK:             %[[VAL_252:.*]] = llvm.call tail @printf(%[[VAL_9]], %[[VAL_251]]) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:             %[[VAL_253:.*]] = llvm.call tail @printf(%[[VAL_10]], %[[VAL_11]]) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32) -> i32
// CHECK:             scf.yield %[[VAL_6]], %[[VAL_7]] : i32, i32
// CHECK:           }
// CHECK:           cf.switch %[[VAL_254:.*]]#1 : i32, [
// CHECK:             default: ^bb2,
// CHECK:             0: ^bb1(%[[VAL_254]]#0 : i32)
// CHECK:           ]
// CHECK:         ^bb1(%[[VAL_255:.*]]: i32):
// CHECK:           llvm.return %[[VAL_255]] : i32
// CHECK:         ^bb2:
// CHECK:           llvm.unreachable
// CHECK:         }
// CHECK:         llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory_effects = #[[?]]<other = none, argMem = none, inaccessibleMem = readwrite>, no_unwind, passthrough = ["mustprogress", "nofree", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
// CHECK:         llvm.func local_unnamed_addr @cudaMalloc(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> i32 attributes {passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @cudaMemcpy(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @_ZNSt6chrono3_V212steady_clock3nowEv() -> i64 attributes {no_unwind, passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @clock_gettime(i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32 attributes {no_unwind, passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {memory_effects = #[[?]]<other = none, argMem = readwrite, inaccessibleMem = readwrite>, no_unwind, passthrough = ["mustprogress", ["allockind", "4"], ["alloc-family", "malloc"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
// CHECK:         llvm.func local_unnamed_addr @cudaFree(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @__isoc23_strtol(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {no_unwind, passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @getenv(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {memory_effects = #[[?]]<other = read, argMem = read, inaccessibleMem = read>, no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @fopen(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @fprintf(!llvm.ptr {llvm.nocapture, llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @__errno_location() -> !llvm.ptr attributes {memory_effects = #[[?]]<other = none, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return}
// CHECK:         llvm.func local_unnamed_addr @strerror(i32 {llvm.noundef}) -> !llvm.ptr attributes {no_unwind, passthrough = {{\[\[}}"no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @exit(i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", "noreturn", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @fclose(!llvm.ptr {llvm.nocapture, llvm.noundef}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #[[?]]<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK:         llvm.func local_unnamed_addr @__cudaRegisterFunction(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.func local_unnamed_addr @__cudaRegisterFatBinary(!llvm.ptr) -> !llvm.ptr
// CHECK:         llvm.func local_unnamed_addr @__cudaRegisterFatBinaryEnd(!llvm.ptr)
// CHECK:         llvm.func local_unnamed_addr @__cudaUnregisterFatBinary(!llvm.ptr)

// CHECK-LABEL:   llvm.func internal @__cuda_module_dtor() attributes {dso_local} {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.addressof @__cuda_gpubin_handle : !llvm.ptr
// CHECK:           %[[VAL_1:.*]] = llvm.load %[[VAL_0]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
// CHECK:           llvm.call tail @__cudaUnregisterFatBinary(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func local_unnamed_addr @atexit(!llvm.ptr) -> i32 attributes {passthrough = ["nofree"]}
// CHECK:         llvm.func local_unnamed_addr @__mlir_launch_coerced_kernel__Z25__device_stub__stencil_1dPKiPi(!llvm.ptr, i64, i32, i64, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK:         llvm.func local_unnamed_addr @__mlir_launch_kernel__Z25__device_stub__stencil_1dPKiPi(!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK:         llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]}

