// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> ((s0 - 1) floordiv 16 + 1)>

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
// CHECK-LABEL:   gpu.module @__mlir_gpu_module
// CHECK:           llvm.comdat @__llvm_global_comdat {
// CHECK:             llvm.comdat_selector @_ZZ13MatrixMulCUDAILi16EEvPfS0_S0_iiE2As any
// CHECK:             llvm.comdat_selector @_ZZ13MatrixMulCUDAILi16EEvPfS0_S0_iiE2Bs any
// CHECK:             llvm.comdat_selector @_ZZ13MatrixMulCUDAILi32EEvPfS0_S0_iiE2As any
// CHECK:             llvm.comdat_selector @_ZZ13MatrixMulCUDAILi32EEvPfS0_S0_iiE2Bs any
// CHECK:             llvm.comdat_selector @_Z13MatrixMulCUDAILi16EEvPfS0_S0_ii any
// CHECK:             llvm.comdat_selector @_Z13MatrixMulCUDAILi32EEvPfS0_S0_ii any
// CHECK:           }
// CHECK:           llvm.func private local_unnamed_addr @__mlir.par.kernel._Z13MatrixMulCUDAILi16EEvPfS0_S0_ii_32764(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %[[VAL_8:.*]]: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_9:.*]]: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_10:.*]]: i32 {llvm.noundef}, %[[VAL_11:.*]]: i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_Z13MatrixMulCUDAILi16EEvPfS0_S0_ii)
// CHECK:             %[[VAL_12:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK:             %[[VAL_13:.*]] = "memref.ataddr"(%[[VAL_9]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_14:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_15:.*]] = "memref.ataddr"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_16:.*]] = arith.index_cast %[[VAL_11]] : i32 to index
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_11]] : i32 to index
// CHECK:             %[[VAL_18:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:             %[[VAL_19:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:             %[[VAL_20:.*]] = arith.index_cast %[[VAL_11]] : i32 to index
// CHECK:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_11]] : i32 to index
// CHECK:             %[[VAL_22:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:             %[[VAL_23:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:             %[[VAL_24:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK:             affine.parallel (%[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]]) = (0, 0, 0) to (symbol(%[[VAL_24]]), symbol(%[[VAL_23]]), 1) {
// CHECK:               %[[VAL_28:.*]] = memref.alloca() : memref<1024xi8, 3>
// CHECK:               %[[VAL_29:.*]] = memref.alloca() : memref<1024xi8, 3>
// CHECK:               affine.parallel (%[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]]) = (0, 0, 0) to (16, 16, 1) {
// CHECK:                 %[[VAL_33:.*]] = affine.for %[[VAL_34:.*]] = 0 to #[[$ATTR_0]](){{\[}}%[[VAL_22]]] iter_args(%[[VAL_35:.*]] = %[[VAL_12]]) -> (f32) {
// CHECK:                   %[[VAL_36:.*]] = affine.vector_load %[[VAL_14]][(%[[VAL_31]] * symbol(%[[VAL_19]])) * 4 + %[[VAL_30]] * 4 + %[[VAL_34]] * 64 + (%[[VAL_26]] * (symbol(%[[VAL_18]]) * 16)) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:                   affine.vector_store %[[VAL_36]], %[[VAL_28]]{{\[}}%[[VAL_31]] * 64 + %[[VAL_30]] * 4] : memref<1024xi8, 3>, vector<4xi8>
// CHECK:                   %[[VAL_37:.*]] = affine.vector_load %[[VAL_13]][(%[[VAL_31]] * symbol(%[[VAL_17]])) * 4 + %[[VAL_30]] * 4 + %[[VAL_25]] * 64 + (%[[VAL_34]] * (symbol(%[[VAL_16]]) * 16)) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:                   affine.vector_store %[[VAL_37]], %[[VAL_29]]{{\[}}%[[VAL_31]] * 64 + %[[VAL_30]] * 4] : memref<1024xi8, 3>, vector<4xi8>
// CHECK:                   "affine.barrier"(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (index, index, index) -> ()
// CHECK:                   %[[VAL_38:.*]] = affine.for %[[VAL_39:.*]] = 0 to 16 iter_args(%[[VAL_40:.*]] = %[[VAL_35]]) -> (f32) {
// CHECK:                     %[[VAL_41:.*]] = affine.vector_load %[[VAL_28]]{{\[}}%[[VAL_31]] * 64 + %[[VAL_39]] * 4] : memref<1024xi8, 3>, vector<4xi8>
// CHECK:                     %[[VAL_42:.*]] = llvm.bitcast %[[VAL_41]] : vector<4xi8> to f32
// CHECK:                     %[[VAL_43:.*]] = affine.vector_load %[[VAL_29]]{{\[}}%[[VAL_39]] * 64 + %[[VAL_30]] * 4] : memref<1024xi8, 3>, vector<4xi8>
// CHECK:                     %[[VAL_44:.*]] = llvm.bitcast %[[VAL_43]] : vector<4xi8> to f32
// CHECK:                     %[[VAL_45:.*]] = llvm.fmul %[[VAL_42]], %[[VAL_44]]  {fastmathFlags = #[[?]]<contract>} : f32
// CHECK:                     %[[VAL_46:.*]] = llvm.fadd %[[VAL_40]], %[[VAL_45]]  {fastmathFlags = #[[?]]<contract>} : f32
// CHECK:                     affine.yield %[[VAL_46]] : f32
// CHECK:                   }
// CHECK:                   "affine.barrier"(%[[VAL_30]], %[[VAL_31]], %[[VAL_32]]) : (index, index, index) -> ()
// CHECK:                   affine.yield %[[VAL_38]] : f32
// CHECK:                 }
// CHECK:                 %[[VAL_47:.*]] = llvm.bitcast %[[VAL_33]] : f32 to vector<4xi8>
// CHECK:                 affine.vector_store %[[VAL_47]], %[[VAL_15]]{{\[}}%[[VAL_25]] * 64 + %[[VAL_30]] * 4 + (%[[VAL_31]] * symbol(%[[VAL_21]])) * 4 + (%[[VAL_26]] * (symbol(%[[VAL_20]]) * 16)) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:               } {gpu.par.block}
// CHECK:             } {gpu.par.grid}
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }
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
