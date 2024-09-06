// REQUIRES; asserts
// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" -debug-only=gpu-affine-opt 2>&1 | FileCheck %s --check-prefix=SCHEDULE

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "_ZTS4dim3", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 4>, <#tbaa_type_desc2, 8>}>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "_ZTSZ22_ConvertSMVer2ArchNameiiE13sSMtoArchName", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc3, 8>}>
#tbaa_type_desc6 = #llvm.tbaa_type_desc<id = "_ZTSZ19_ConvertSMVer2CoresiiE10sSMtoCores", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 4>}>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 4>
#tbaa_tag6 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 8>
#tbaa_tag7 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag8 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc3, offset = 8>
#tbaa_tag9 = #llvm.tbaa_tag<base_type = #tbaa_type_desc6, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag10 = #llvm.tbaa_tag<base_type = #tbaa_type_desc6, access_type = #tbaa_type_desc2, offset = 4>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module} {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.comdat @__llvm_global_comdat {
      llvm.comdat_selector @_ZZ13MatrixMulCUDAILi16EEvPfS0_S0_iiE2As any
      llvm.comdat_selector @_ZZ13MatrixMulCUDAILi16EEvPfS0_S0_iiE2Bs any
      llvm.comdat_selector @_ZZ13MatrixMulCUDAILi32EEvPfS0_S0_iiE2As any
      llvm.comdat_selector @_ZZ13MatrixMulCUDAILi32EEvPfS0_S0_iiE2Bs any
      llvm.comdat_selector @_Z13MatrixMulCUDAILi16EEvPfS0_S0_ii any
      llvm.comdat_selector @_Z13MatrixMulCUDAILi32EEvPfS0_S0_ii any
    }
    llvm.func private local_unnamed_addr @__mlir.par.kernel._Z13MatrixMulCUDAILi16EEvPfS0_S0_ii_32764(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg8: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg9: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg10: i32 {llvm.noundef}, %arg11: i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_Z13MatrixMulCUDAILi16EEvPfS0_S0_ii) attributes {gpu.par.kernel, sym_visibility = "private"} {
      %c4_i32 = arith.constant 4 : i32
      %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %c16_i32 = arith.constant 16 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %1 = arith.index_cast %arg1 : i64 to index
      %2 = arith.index_cast %arg0 : i64 to index
      affine.parallel (%arg12, %arg13, %arg14) = (0, 0, 0) to (symbol(%2), symbol(%1), 1) {
        %3 = llvm.alloca %c1_i32 x !llvm.array<16 x array<16 x f32>> : (i32) -> !llvm.ptr<3>
        %4 = llvm.alloca %c1_i32 x !llvm.array<16 x array<16 x f32>> : (i32) -> !llvm.ptr<3>
        affine.parallel (%arg15, %arg16, %arg17) = (0, 0, 0) to (16, 16, 1) {
          %5 = llvm.addrspacecast %3 : !llvm.ptr<3> to !llvm.ptr
          %6 = llvm.addrspacecast %4 : !llvm.ptr<3> to !llvm.ptr
          %7 = arith.index_cast %arg12 : index to i32
          %8 = arith.index_cast %arg13 : index to i32
          %9 = arith.index_cast %arg15 : index to i32
          %10 = arith.index_cast %arg16 : index to i32
          %11 = arith.shli %arg10, %c4_i32 : i32
          %12 = arith.muli %11, %8 : i32
          %13 = arith.addi %12, %arg10 : i32
          %14 = arith.shli %7, %c4_i32 : i32
          %15 = arith.shli %arg11, %c4_i32 : i32
          %16 = arith.muli %10, %arg10 : i32
          %17 = arith.addi %16, %9 : i32
          %18 = arith.extui %10 : i32 to i64
          %19 = arith.extui %9 : i32 to i64
          %20 = llvm.getelementptr inbounds %5[0, %18, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x array<16 x f32>>
          %21 = arith.muli %10, %arg11 : i32
          %22 = arith.addi %21, %9 : i32
          %23 = llvm.getelementptr inbounds %6[0, %18, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x array<16 x f32>>
          %24:2 = scf.for %arg18 = %12 to %13 step %c16_i32 iter_args(%arg19 = %14, %arg20 = %0) -> (i32, f32)  : i32 {
            %31 = arith.addi %17, %arg18 : i32
            %32 = arith.extsi %31 : i32 to i64
            %33 = llvm.getelementptr inbounds %arg8[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %34 = llvm.load %33 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
            llvm.store %34, %20 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : f32, !llvm.ptr
            %35 = arith.addi %22, %arg19 : i32
            %36 = arith.extsi %35 : i32 to i64
            %37 = llvm.getelementptr inbounds %arg9[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
            %38 = llvm.load %37 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
            llvm.store %38, %23 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : f32, !llvm.ptr
            "affine.barrier"(%arg15, %arg16, %arg17) : (index, index, index) -> ()
            %39 = scf.for %arg21 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg22 = %arg20) -> (f32)  : i32 {
              %41 = arith.extui %arg21 : i32 to i64
              %42 = llvm.getelementptr inbounds %5[0, %18, %41] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x array<16 x f32>>
              %43 = llvm.load %42 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
              %44 = llvm.getelementptr inbounds %6[0, %41, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<16 x array<16 x f32>>
              %45 = llvm.load %44 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f32
              %46 = llvm.fmul %43, %45  {fastmathFlags = #llvm.fastmath<contract>} : f32
              %47 = llvm.fadd %arg22, %46  {fastmathFlags = #llvm.fastmath<contract>} : f32
              scf.yield %47 : f32
            }
            "affine.barrier"(%arg15, %arg16, %arg17) : (index, index, index) -> ()
            %40 = arith.addi %arg19, %15 : i32
            scf.yield %40, %39 : i32, f32
          }
          %25 = arith.muli %15, %8 : i32
          %26 = arith.addi %14, %9 : i32
          %27 = arith.addi %26, %21 : i32
          %28 = arith.addi %27, %25 : i32
          %29 = arith.extsi %28 : i32 to i64
          %30 = llvm.getelementptr inbounds %arg7[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
          llvm.store %24#1, %30 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : f32, !llvm.ptr
        } {gpu.par.block}
      } {gpu.par.grid}
      llvm.return
    }
  }
}

// SCHEDULE: domain: "[P0, P1, P2, P3] -> { S26.affine.yield[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S20.affine.vector_load[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S18.affine.load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S32.affine.vector_store[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S3.memref.ataddr[]; S25.llvm.fadd[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S15.affine.vector_load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S33.affine.yield[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S17.affine.yield[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S1.memref.ataddr[]; S6.arith.index_cast[]; S13.affine.vector_load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S27.affine.store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S5.arith.index_cast[]; S31.llvm.bitcast[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S16.affine.vector_store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S19.affine.store_var[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S21.llvm.bitcast[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S4.arith.index_cast[]; S0.llvm.mlir.constant[]; S7.arith.index_cast[]; S22.affine.vector_load[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S28.affine.yield[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S30.affine.load[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S34.affine.yield[i0, i1, 0] : 0 <= i0 < P1 and 0 <= i1 < P0; S24.llvm.fmul[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S14.affine.vector_store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S12.affine.yield[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S2.memref.ataddr[]; S29.affine.yield[i0, i1, 0, i3] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2; S11.affine.store[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S23.llvm.bitcast[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15 }"
// SCHEDULE: child:
// SCHEDULE:   sequence:
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S0.llvm.mlir.constant[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S1.memref.ataddr[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S2.memref.ataddr[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S3.memref.ataddr[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S4.arith.index_cast[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S5.arith.index_cast[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S6.arith.index_cast[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S7.arith.index_cast[] }"
// SCHEDULE:   - filter: "[P0, P1, P2, P3] -> { S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S33.affine.yield[i0, i1, i2, i3, i4, i5]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6]; S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6]; S30.affine.load[i0, i1, i2, i3, i4, i5]; S34.affine.yield[i0, i1, i2]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6]; S12.affine.yield[i0, i1, i2, i3, i4, i5]; S29.affine.yield[i0, i1, i2, i3]; S11.affine.store[i0, i1, i2, i3, i4, i5]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:     child:
// SCHEDULE:       schedule: "[P0, P1, P2, P3] -> L16.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i0)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i0)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S29.affine.yield[i0, i1, i2, i3] -> [(i0)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i0)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i0)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i0)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i0)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i0)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i0)]; S34.affine.yield[i0, i1, i2] -> [(i0)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i0)] }]"
// SCHEDULE:       permutable: 1
// SCHEDULE:       child:
// SCHEDULE:         schedule: "[P0, P1, P2, P3] -> L15.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i1)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i1)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S29.affine.yield[i0, i1, i2, i3] -> [(i1)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i1)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i1)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i1)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i1)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i1)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i1)]; S34.affine.yield[i0, i1, i2] -> [(i1)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i1)] }]"
// SCHEDULE:         permutable: 1
// SCHEDULE:         child:
// SCHEDULE:           schedule: "[P0, P1, P2, P3] -> L14.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i2)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i2)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S29.affine.yield[i0, i1, i2, i3] -> [(i2)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i2)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i2)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i2)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i2)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i2)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i2)]; S34.affine.yield[i0, i1, i2] -> [(i2)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i2)] }]"
// SCHEDULE:           permutable: 1
// SCHEDULE:           child:
// SCHEDULE:             sequence:
// SCHEDULE:             - filter: "[P0, P1, P2, P3] -> { S11.affine.store[i0, i1, i2, i3, i4, i5]; S12.affine.yield[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:               child:
// SCHEDULE:                 schedule: "[P0, P1, P2, P3] -> L2.affine.parallel[{ S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i3)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i3)] }]"
// SCHEDULE:                 permutable: 1
// SCHEDULE:                 child:
// SCHEDULE:                   schedule: "[P0, P1, P2, P3] -> L1.affine.parallel[{ S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i4)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i4)] }]"
// SCHEDULE:                   permutable: 1
// SCHEDULE:                   child:
// SCHEDULE:                     schedule: "[P0, P1, P2, P3] -> L0.affine.parallel[{ S12.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i5)]; S11.affine.store[i0, i1, i2, i3, i4, i5] -> [(i5)] }]"
// SCHEDULE:                     permutable: 1
// SCHEDULE:                     child:
// SCHEDULE:                       sequence:
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S11.affine.store[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S12.affine.yield[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:             - filter: "[P0, P1, P2, P3] -> { S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S29.affine.yield[i0, i1, i2, i3]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:               child:
// SCHEDULE:                 schedule: "[P0, P1, P2, P3] -> L10.affine.for[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S29.affine.yield[i0, i1, i2, i3] -> [(i3)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i3)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i3)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i3)] }]"
// SCHEDULE:                 child:
// SCHEDULE:                   sequence:
// SCHEDULE:                   - filter: "[P0, P1, P2, P3] -> { S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6]; S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                     child:
// SCHEDULE:                       schedule: "[P0, P1, P2, P3] -> L5.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i4)] }]"
// SCHEDULE:                       permutable: 1
// SCHEDULE:                       child:
// SCHEDULE:                         schedule: "[P0, P1, P2, P3] -> L4.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i5)] }]"
// SCHEDULE:                         permutable: 1
// SCHEDULE:                         child:
// SCHEDULE:                           schedule: "[P0, P1, P2, P3] -> L3.affine.parallel[{ S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i6)] }]"
// SCHEDULE:                           permutable: 1
// SCHEDULE:                           child:
// SCHEDULE:                             sequence:
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S14.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S16.affine.vector_store[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S17.affine.yield[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                   - filter: "[P0, P1, P2, P3] -> { S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7]; S28.affine.yield[i0, i1, i2, i3, i4, i5, i6]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                     child:
// SCHEDULE:                       schedule: "[P0, P1, P2, P3] -> L9.affine.parallel[{ S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i4)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i4)] }]"
// SCHEDULE:                       permutable: 1
// SCHEDULE:                       child:
// SCHEDULE:                         schedule: "[P0, P1, P2, P3] -> L8.affine.parallel[{ S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i5)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i5)] }]"
// SCHEDULE:                         permutable: 1
// SCHEDULE:                         child:
// SCHEDULE:                           schedule: "[P0, P1, P2, P3] -> L7.affine.parallel[{ S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S18.affine.load[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)]; S27.affine.store[i0, i1, i2, i3, i4, i5, i6] -> [(i6)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i6)] }]"
// SCHEDULE:                           permutable: 1
// SCHEDULE:                           child:
// SCHEDULE:                             sequence:
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S18.affine.load[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S19.affine.store_var[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                               child:
// SCHEDULE:                                 schedule: "[P0, P1, P2, P3] -> L6.affine.for[{ S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)]; S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] -> [(i7)] }]"
// SCHEDULE:                                 child:
// SCHEDULE:                                   sequence:
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S20.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S21.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S22.affine.vector_load[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S23.llvm.bitcast[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S24.llvm.fmul[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S25.llvm.fadd[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                                   - filter: "[P0, P1, P2, P3] -> { S26.affine.yield[i0, i1, i2, i3, i4, i5, i6, i7] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S27.affine.store[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                             - filter: "[P0, P1, P2, P3] -> { S28.affine.yield[i0, i1, i2, i3, i4, i5, i6] }"
// SCHEDULE:                   - filter: "[P0, P1, P2, P3] -> { S29.affine.yield[i0, i1, i2, i3] }"
// SCHEDULE:             - filter: "[P0, P1, P2, P3] -> { S33.affine.yield[i0, i1, i2, i3, i4, i5]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5]; S30.affine.load[i0, i1, i2, i3, i4, i5]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:               child:
// SCHEDULE:                 schedule: "[P0, P1, P2, P3] -> L13.affine.parallel[{ S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i3)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i3)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i3)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i3)] }]"
// SCHEDULE:                 permutable: 1
// SCHEDULE:                 child:
// SCHEDULE:                   schedule: "[P0, P1, P2, P3] -> L12.affine.parallel[{ S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i4)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i4)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i4)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i4)] }]"
// SCHEDULE:                   permutable: 1
// SCHEDULE:                   child:
// SCHEDULE:                     schedule: "[P0, P1, P2, P3] -> L11.affine.parallel[{ S30.affine.load[i0, i1, i2, i3, i4, i5] -> [(i5)]; S33.affine.yield[i0, i1, i2, i3, i4, i5] -> [(i5)]; S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [(i5)]; S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] -> [(i5)] }]"
// SCHEDULE:                     permutable: 1
// SCHEDULE:                     child:
// SCHEDULE:                       sequence:
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S30.affine.load[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S31.llvm.bitcast[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S32.affine.vector_store[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:                       - filter: "[P0, P1, P2, P3] -> { S33.affine.yield[i0, i1, i2, i3, i4, i5] }"
// SCHEDULE:             - filter: "[P0, P1, P2, P3] -> { S34.affine.yield[i0, i1, i2] }"
// SCHEDULE: domain: "[P0, P1, P2, P3] -> { S26.affine.yield[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S20.affine.vector_load[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S18.affine.load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S32.affine.vector_store[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S3.memref.ataddr[]; S25.llvm.fadd[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S15.affine.vector_load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S33.affine.yield[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S17.affine.yield[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S1.memref.ataddr[]; S6.arith.index_cast[]; S13.affine.vector_load[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S27.affine.store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S5.arith.index_cast[]; S31.llvm.bitcast[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S16.affine.vector_store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S19.affine.store_var[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S21.llvm.bitcast[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S4.arith.index_cast[]; S0.llvm.mlir.constant[]; S7.arith.index_cast[]; S22.affine.vector_load[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S28.affine.yield[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S30.affine.load[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S34.affine.yield[i0, i1, 0] : 0 <= i0 < P1 and 0 <= i1 < P0; S24.llvm.fmul[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15; S14.affine.vector_store[i0, i1, 0, i3, i4, i5, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15; S12.affine.yield[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S2.memref.ataddr[]; S29.affine.yield[i0, i1, 0, i3] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2; S11.affine.store[i0, i1, 0, i3, i4, 0] : 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15; S23.llvm.bitcast[i0, i1, 0, i3, i4, i5, 0, i7] : 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 and 0 <= i7 <= 15 }"
// SCHEDULE: accesses:
// SCHEDULE:   - S0.llvm.mlir.constant:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A1[] :  }"
// SCHEDULE:   - S1.memref.ataddr:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A3[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A2[] :  }"
// SCHEDULE:   - S2.memref.ataddr:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A5[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A4[] :  }"
// SCHEDULE:   - S3.memref.ataddr:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A7[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A6[] :  }"
// SCHEDULE:   - S4.arith.index_cast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A9[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A8[] :  }"
// SCHEDULE:   - S5.arith.index_cast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A11[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A10[] :  }"
// SCHEDULE:   - S6.arith.index_cast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A13[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A12[] :  }"
// SCHEDULE:   - S7.arith.index_cast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A15[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [] -> A14[] :  }"
// SCHEDULE:   - S8.memref.alloca:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S9.memref.alloca:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S10.memref.alloca:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S11.affine.store:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5] -> A16[o0, o1, o2] : o0 = i3 and o1 = i4 and o2 = i5 }"
// SCHEDULE:   - S12.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S13.affine.vector_load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { S13.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [] : i2 = 0 and i6 = 0 and 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A17[] :  }"
// SCHEDULE:   - S14.affine.vector_store:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A18[o0] : o0 = 4i4 + 64i5 }"
// SCHEDULE:   - S15.affine.vector_load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { S15.affine.vector_load[i0, i1, i2, i3, i4, i5, i6] -> [] : i2 = 0 and i6 = 0 and 0 <= i0 < P1 and 0 <= i1 < P0 and i3 >= 0 and 16i3 < P2 and 0 <= i4 <= 15 and 0 <= i5 <= 15 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A19[] :  }"
// SCHEDULE:   - S16.affine.vector_store:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A20[o0] : o0 = 4i4 + 64i5 }"
// SCHEDULE:   - S17.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S18.affine.load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A16[o0, o1, o2] : o0 = i4 and o1 = i5 and o2 = i6 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A21[] :  }"
// SCHEDULE:   - S19.affine.store_var:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A21[] :  }"
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A22[] :  }"
// SCHEDULE:   - S20.affine.vector_load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A18[o0] : o0 = 64i5 + 4i7 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A23[] :  }"
// SCHEDULE:   - S21.llvm.bitcast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A23[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A24[] :  }"
// SCHEDULE:   - S22.affine.vector_load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A20[o0] : o0 = 4i4 + 64i7 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A25[] :  }"
// SCHEDULE:   - S23.llvm.bitcast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A25[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A26[] :  }"
// SCHEDULE:   - S24.llvm.fmul:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A24[] :  }"
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A26[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A27[] :  }"
// SCHEDULE:   - S25.llvm.fadd:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A22[] :  }"
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A27[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A28[] :  }"
// SCHEDULE:   - S26.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A28[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6, i7] -> A22[] :  }"
// SCHEDULE:   - S27.affine.store:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5, i6] -> A16[o0, o1, o2] : o0 = i4 and o1 = i5 and o2 = i6 }"
// SCHEDULE:   - S28.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S29.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S30.affine.load:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5] -> A16[o0, o1, o2] : o0 = i3 and o1 = i4 and o2 = i5 }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5] -> A29[] :  }"
// SCHEDULE:   - S31.llvm.bitcast:
// SCHEDULE:       reads:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5] -> A29[] :  }"
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { [i0, i1, i2, i3, i4, i5] -> A30[] :  }"
// SCHEDULE:   - S32.affine.vector_store:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:         - "[P0, P1, P2, P3] -> { S32.affine.vector_store[i0, i1, i2, i3, i4, i5] -> [] : i2 = 0 and i5 = 0 and 0 <= i0 < P1 and 0 <= i1 < P0 and 0 <= i3 <= 15 and 0 <= i4 <= 15 }"
// SCHEDULE:   - S33.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S34.affine.yield:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
// SCHEDULE:   - S35.llvm.return:
// SCHEDULE:       reads:
// SCHEDULE:       writes:
