// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c256_i32 = arith.constant 256 : i32
      %c7_i32 = arith.constant 7 : i32
      %c263_i32 = arith.constant 263 : i32
      %c-7_i32 = arith.constant -7 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.index_cast %arg0 : i64 to index
      affine.parallel (%arg9) = (0) to (symbol(%0)) {
        %1 = llvm.alloca %c1_i32 x !llvm.array<270 x i32> : (i32) -> !llvm.ptr<3>
        affine.parallel (%arg10) = (0) to (256) {
          %2 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
          %3 = arith.index_cast %arg10 : index to i32
          %4 = arith.index_cast %arg9 : index to i32
          %5 = arith.muli %4, %c256_i32 : i32
          %6 = arith.addi %5, %3 : i32
          %7 = arith.addi %3, %c7_i32 : i32
          %8 = arith.extsi %6 : i32 to i64
          %9 = llvm.getelementptr inbounds %arg7[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          %10 = llvm.load %9 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
          %11 = arith.extui %7 : i32 to i64
          %12 = llvm.getelementptr inbounds %2[0, %11] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
          llvm.store %10, %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
          %13 = arith.cmpi ult, %3, %c7_i32 : i32
          scf.if %13 {
            %16 = arith.cmpi slt, %6, %c7_i32 : i32
            %17 = scf.if %16 -> (i32) {
              scf.yield %c0_i32 : i32
            } else {
              %25 = arith.extui %6 : i32 to i64
              %26 = llvm.getelementptr %arg7[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i32
              %27 = llvm.getelementptr %26[-28] : (!llvm.ptr) -> !llvm.ptr, i8
              %28 = llvm.load %27 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
              scf.yield %28 : i32
            }
            %18 = arith.extui %3 : i32 to i64
            %19 = llvm.getelementptr inbounds %2[0, %18] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
            llvm.store %17, %19 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
            %20 = llvm.getelementptr %9[1024] : (!llvm.ptr) -> !llvm.ptr, i8
            %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
            %22 = arith.addi %3, %c263_i32 : i32
            %23 = arith.extui %22 : i32 to i64
            %24 = llvm.getelementptr inbounds %2[0, %23] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
            llvm.store %21, %24 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
          }
          nvvm.barrier0
          %14 = scf.for %arg11 = %c-7_i32 to %c8_i32 step %c1_i32 iter_args(%arg12 = %c0_i32) -> (i32)  : i32 {
            %16 = arith.addi %arg11, %7 : i32
            %17 = arith.extsi %16 : i32 to i64
            %18 = llvm.getelementptr inbounds %2[0, %17] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<270 x i32>
            %19 = llvm.load %18 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
            %20 = arith.addi %19, %arg12 : i32
            scf.yield %20 : i32
          }
          %15 = llvm.getelementptr inbounds %arg8[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          llvm.store %14, %15 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
        } {gpu.par.block.x}
      } {gpu.par.grid.x}
      llvm.return
    }
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0) : (-d0 + 6 >= 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (-(d0 * 256 + d1) + 6 >= 0)>

// CHECK-LABEL:   gpu.module @__mlir_gpu_module [#[[?]]<chip = "sm_80">]  {
// CHECK:           llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_8:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #[[?]]<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #[[?]]<["+ptx84", "+sm_52"]>} {
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_10:.*]] = "memref.ataddr"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_11:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK:             affine.parallel (%[[VAL_13:.*]]) = (0) to (symbol(%[[VAL_12]])) {
// CHECK:               %[[VAL_14:.*]] = memref.alloca() : memref<1080xi8, 3>
// CHECK:               affine.parallel (%[[VAL_15:.*]]) = (0) to (256) {
// CHECK:                 %[[VAL_16:.*]] = affine.parallel (%[[VAL_17:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                   %[[VAL_18:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 1024 + %[[VAL_15]] * 4 + %[[VAL_17]]] : memref<?xi8>
// CHECK:                   %[[VAL_19:.*]] = vector.broadcast %[[VAL_18]] : i8 to vector<4xi8>
// CHECK:                   affine.yield %[[VAL_19]] : vector<4xi8>
// CHECK:                 } {affine.vector.load}
// CHECK:                 affine.parallel (%[[VAL_20:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_21:.*]] = vector.extract %[[VAL_16]]{{\[}}%[[VAL_20]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_21]], %[[VAL_14]]{{\[}}%[[VAL_15]] * 4 + %[[VAL_20]] + 28] : memref<1080xi8, 3>
// CHECK:                 } {affine.vector.store}
// CHECK:                 affine.if #[[$ATTR_0]](%[[VAL_15]]) {
// CHECK:                   %[[VAL_22:.*]] = affine.if #[[$ATTR_1]](%[[VAL_13]], %[[VAL_15]]) -> i32 {
// CHECK:                     affine.yield %[[VAL_9]] : i32
// CHECK:                   } else {
// CHECK:                     %[[VAL_23:.*]] = affine.parallel (%[[VAL_24:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                       %[[VAL_25:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 1024 + %[[VAL_15]] * 4 + %[[VAL_24]] - 28] : memref<?xi8>
// CHECK:                       %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i8 to vector<4xi8>
// CHECK:                       affine.yield %[[VAL_26]] : vector<4xi8>
// CHECK:                     } {affine.vector.load}
// CHECK:                     %[[VAL_27:.*]] = llvm.bitcast %[[VAL_23]] : vector<4xi8> to i32
// CHECK:                     affine.yield %[[VAL_27]] : i32
// CHECK:                   }
// CHECK:                   %[[VAL_28:.*]] = llvm.bitcast %[[VAL_22]] : i32 to vector<4xi8>
// CHECK:                   affine.parallel (%[[VAL_29:.*]]) = (0) to (4) {
// CHECK:                     %[[VAL_30:.*]] = vector.extract %[[VAL_28]]{{\[}}%[[VAL_29]]] : i8 from vector<4xi8>
// CHECK:                     affine.store %[[VAL_30]], %[[VAL_14]]{{\[}}%[[VAL_15]] * 4 + %[[VAL_29]]] : memref<1080xi8, 3>
// CHECK:                   } {affine.vector.store}
// CHECK:                   %[[VAL_31:.*]] = affine.parallel (%[[VAL_32:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_33:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 1024 + %[[VAL_15]] * 4 + %[[VAL_32]] + 1024] : memref<?xi8>
// CHECK:                     %[[VAL_34:.*]] = vector.broadcast %[[VAL_33]] : i8 to vector<4xi8>
// CHECK:                     affine.yield %[[VAL_34]] : vector<4xi8>
// CHECK:                   } {affine.vector.load}
// CHECK:                   affine.parallel (%[[VAL_35:.*]]) = (0) to (4) {
// CHECK:                     %[[VAL_36:.*]] = vector.extract %[[VAL_31]]{{\[}}%[[VAL_35]]] : i8 from vector<4xi8>
// CHECK:                     affine.store %[[VAL_36]], %[[VAL_14]]{{\[}}%[[VAL_15]] * 4 + %[[VAL_35]] + 1052] : memref<1080xi8, 3>
// CHECK:                   } {affine.vector.store}
// CHECK:                 }
// CHECK:                 nvvm.barrier0
// CHECK:                 %[[VAL_37:.*]] = affine.for %[[VAL_38:.*]] = -7 to 8 iter_args(%[[VAL_39:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:                   %[[VAL_40:.*]] = affine.parallel (%[[VAL_41:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_42:.*]] = affine.load %[[VAL_14]]{{\[}}%[[VAL_38]] * 4 + %[[VAL_15]] * 4 + %[[VAL_41]] + 28] : memref<1080xi8, 3>
// CHECK:                     %[[VAL_43:.*]] = vector.broadcast %[[VAL_42]] : i8 to vector<4xi8>
// CHECK:                     affine.yield %[[VAL_43]] : vector<4xi8>
// CHECK:                   } {affine.vector.load}
// CHECK:                   %[[VAL_44:.*]] = llvm.bitcast %[[VAL_40]] : vector<4xi8> to i32
// CHECK:                   %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_39]] : i32
// CHECK:                   affine.yield %[[VAL_45]] : i32
// CHECK:                 }
// CHECK:                 %[[VAL_46:.*]] = llvm.bitcast %[[VAL_37]] : i32 to vector<4xi8>
// CHECK:                 affine.parallel (%[[VAL_47:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_48:.*]] = vector.extract %[[VAL_46]]{{\[}}%[[VAL_47]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_48]], %[[VAL_11]]{{\[}}%[[VAL_13]] * 1024 + %[[VAL_15]] * 4 + %[[VAL_47]]] : memref<?xi8>
// CHECK:                 } {affine.vector.store}
// CHECK:               } {gpu.par.block.x}
// CHECK:             } {gpu.par.grid.x}
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }

