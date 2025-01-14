// RUN: mlir-opt %s --gpu-affine-opt --canonicalize --split-input-file | FileCheck %s

#set = affine_set<(d0) : (-d0 + 6 >= 0)>
#set1 = affine_set<(d0, d1) : (-(d0 * 256 + d1) + 6 >= 0)>
module {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c0_i32 = arith.constant 0 : i32
      %0 = "memref.ataddr"(%arg7) : (!llvm.ptr) -> memref<?x4xi8>
      %1 = "memref.ataddr"(%arg8) : (!llvm.ptr) -> memref<?x256x4xi8>
      %2 = arith.index_cast %arg0 : i64 to index
      affine.parallel (%arg9) = (0) to (symbol(%2)) {
        %alloca = memref.alloca() : memref<270x4xi8, 3>
        affine.parallel (%arg10) = (0) to (256) {
          %3 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
            %6 = affine.load %0[%arg9 * 256 + %arg10, %arg11] : memref<?x4xi8>
            %7 = vector.broadcast %6 : i8 to vector<4xi8>
            affine.yield %7 : vector<4xi8>
          } {affine.vector.load}
          affine.parallel (%arg11) = (0) to (4) {
            %6 = vector.extract %3[%arg11] : i8 from vector<4xi8>
            affine.store %6, %alloca[%arg10 + 7, %arg11] : memref<270x4xi8, 3>
          } {affine.vector.store}
          affine.if #set(%arg10) {
            %6 = affine.if #set1(%arg9, %arg10) -> i32 {
              affine.yield %c0_i32 : i32
            } else {
              %9 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
                %11 = affine.load %0[%arg9 * 256 + %arg10 - 7, %arg11] : memref<?x4xi8>
                %12 = vector.broadcast %11 : i8 to vector<4xi8>
                affine.yield %12 : vector<4xi8>
              } {affine.vector.load}
              %10 = llvm.bitcast %9 : vector<4xi8> to i32
              affine.yield %10 : i32
            }
            %7 = llvm.bitcast %6 : i32 to vector<4xi8>
            affine.parallel (%arg11) = (0) to (4) {
              %9 = vector.extract %7[%arg11] : i8 from vector<4xi8>
              affine.store %9, %alloca[%arg10, %arg11] : memref<270x4xi8, 3>
            } {affine.vector.store}
            %8 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
              %9 = affine.load %0[%arg9 * 256 + %arg10 + 256, %arg11] : memref<?x4xi8>
              %10 = vector.broadcast %9 : i8 to vector<4xi8>
              affine.yield %10 : vector<4xi8>
            } {affine.vector.load}
            affine.parallel (%arg11) = (0) to (4) {
              %9 = vector.extract %8[%arg11] : i8 from vector<4xi8>
              affine.store %9, %alloca[%arg10 + 263, %arg11] : memref<270x4xi8, 3>
            } {affine.vector.store}
          }
          nvvm.barrier0
          %4 = affine.for %arg11 = -7 to 8 iter_args(%arg12 = %c0_i32) -> (i32) {
            %6 = affine.parallel (%arg13) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
              %9 = affine.load %alloca[%arg11 + %arg10 + 7, %arg13] : memref<270x4xi8, 3>
              %10 = vector.broadcast %9 : i8 to vector<4xi8>
              affine.yield %10 : vector<4xi8>
            } {affine.vector.load}
            %7 = llvm.bitcast %6 : vector<4xi8> to i32
            %8 = arith.addi %7, %arg12 : i32
            affine.yield %8 : i32
          }
          %5 = llvm.bitcast %4 : i32 to vector<4xi8>
          affine.parallel (%arg11) = (0) to (4) {
            %6 = vector.extract %5[%arg11] : i8 from vector<4xi8>
            affine.store %6, %1[%arg9, %arg10, %arg11] : memref<?x256x4xi8>
          } {affine.vector.store}
        } {gpu.par.block.x}
      } {gpu.par.grid.x}
      llvm.return
    }
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 256)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0 * 256 + 256)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (4)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (7)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (263)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0 * 256 + 263)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (270)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_9:.+]] = affine_set<(d0) : (-d0 + 6 >= 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_set<(d0, d1) : (-(d0 * 256 + d1) + 6 >= 0)>

// CHECK-LABEL:   gpu.module @__mlir_gpu_module [#[[?]]<chip = "sm_80">]  {
// CHECK:           llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %[[VAL_8:.*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #[[?]]<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #[[?]]<["+ptx84", "+sm_52"]>} {
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_10:.*]] = "memref.ataddr"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?x4xi8>
// CHECK:             %[[VAL_11:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?x256x4xi8>
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK:             affine.parallel (%[[VAL_13:.*]]) = (0) to (symbol(%[[VAL_12]])) {
// CHECK:               %[[VAL_14:.*]] = memref.alloca() : memref<270x4xi8, 3>
// CHECK:               affine.parallel (%[[VAL_15:.*]]) = (0) to (256) {
// CHECK:                 %[[VAL_16:.*]] = affine.max #[[$ATTR_0]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_17:.*]] = affine.min #[[$ATTR_1]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_18:.*]] = affine.max #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_19:.*]] = affine.min #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_20:.*]] = affine.max #[[$ATTR_4]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_21:.*]] = affine.min #[[$ATTR_5]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_22:.*]] = affine.max #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 %[[VAL_23:.*]] = affine.min #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                 affine.if #[[$ATTR_8]](%[[VAL_15]]) {
// CHECK:                   affine.for %[[VAL_24:.*]] = #[[$ATTR_4]](%[[VAL_13]], %[[VAL_12]]) to #[[$ATTR_5]](%[[VAL_13]], %[[VAL_12]]) {
// CHECK:                     %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_20]] : index
// CHECK:                     %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_16]] : index
// CHECK:                     affine.for %[[VAL_27:.*]] = #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]]) to #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]]) step 4 {
// CHECK:                       %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_22]] : index
// CHECK:                       %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_18]] : index
// CHECK:                       %[[VAL_30:.*]] = nvgpu.device_async_copy %[[VAL_10]]{{\[}}%[[VAL_26]], %[[VAL_29]]], %[[VAL_14]]{{\[}}%[[VAL_24]], %[[VAL_27]]], 4 : memref<?x4xi8> to memref<270x4xi8, 3>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 affine.if #[[$ATTR_9]](%[[VAL_15]]) {
// CHECK:                   %[[VAL_31:.*]] = affine.if #[[$ATTR_10]](%[[VAL_13]], %[[VAL_15]]) -> i32 {
// CHECK:                     affine.yield %[[VAL_9]] : i32
// CHECK:                   } else {
// CHECK:                     %[[VAL_32:.*]] = affine.parallel (%[[VAL_33:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                       %[[VAL_34:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 256 + %[[VAL_15]] - 7, %[[VAL_33]]] : memref<?x4xi8>
// CHECK:                       %[[VAL_35:.*]] = vector.broadcast %[[VAL_34]] : i8 to vector<4xi8>
// CHECK:                       affine.yield %[[VAL_35]] : vector<4xi8>
// CHECK:                     } {affine.vector.load}
// CHECK:                     %[[VAL_36:.*]] = llvm.bitcast %[[VAL_32]] : vector<4xi8> to i32
// CHECK:                     affine.yield %[[VAL_36]] : i32
// CHECK:                   }
// CHECK:                   %[[VAL_37:.*]] = llvm.bitcast %[[VAL_31]] : i32 to vector<4xi8>
// CHECK:                   affine.parallel (%[[VAL_38:.*]]) = (0) to (4) {
// CHECK:                     %[[VAL_39:.*]] = vector.extract %[[VAL_37]]{{\[}}%[[VAL_38]]] : i8 from vector<4xi8>
// CHECK:                     affine.store %[[VAL_39]], %[[VAL_14]]{{\[}}%[[VAL_15]], %[[VAL_38]]] : memref<270x4xi8, 3>
// CHECK:                   } {affine.vector.store}
// CHECK:                   %[[VAL_40:.*]] = affine.max #[[$ATTR_1]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_41:.*]] = affine.min #[[$ATTR_6]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_42:.*]] = affine.max #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_43:.*]] = affine.min #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_44:.*]] = affine.max #[[$ATTR_5]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_45:.*]] = affine.min #[[$ATTR_7]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_46:.*]] = affine.max #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   %[[VAL_47:.*]] = affine.min #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]])
// CHECK:                   affine.if #[[$ATTR_8]](%[[VAL_15]]) {
// CHECK:                     affine.for %[[VAL_48:.*]] = #[[$ATTR_5]](%[[VAL_13]], %[[VAL_12]]) to #[[$ATTR_7]](%[[VAL_13]], %[[VAL_12]]) {
// CHECK:                       %[[VAL_49:.*]] = arith.subi %[[VAL_48]], %[[VAL_44]] : index
// CHECK:                       %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_40]] : index
// CHECK:                       affine.for %[[VAL_51:.*]] = #[[$ATTR_2]](%[[VAL_13]], %[[VAL_12]]) to #[[$ATTR_3]](%[[VAL_13]], %[[VAL_12]]) step 4 {
// CHECK:                         %[[VAL_52:.*]] = arith.subi %[[VAL_51]], %[[VAL_46]] : index
// CHECK:                         %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_42]] : index
// CHECK:                         %[[VAL_54:.*]] = nvgpu.device_async_copy %[[VAL_10]]{{\[}}%[[VAL_50]], %[[VAL_53]]], %[[VAL_14]]{{\[}}%[[VAL_48]], %[[VAL_51]]], 4 : memref<?x4xi8> to memref<270x4xi8, 3>
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 %[[VAL_55:.*]] = nvgpu.device_async_create_group
// CHECK:                 nvgpu.device_async_wait %[[VAL_55]]
// CHECK:                 nvvm.barrier0
// CHECK:                 %[[VAL_56:.*]] = affine.for %[[VAL_57:.*]] = -7 to 8 iter_args(%[[VAL_58:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:                   %[[VAL_59:.*]] = affine.parallel (%[[VAL_60:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_61:.*]] = affine.load %[[VAL_14]]{{\[}}%[[VAL_57]] + %[[VAL_15]] + 7, %[[VAL_60]]] : memref<270x4xi8, 3>
// CHECK:                     %[[VAL_62:.*]] = vector.broadcast %[[VAL_61]] : i8 to vector<4xi8>
// CHECK:                     affine.yield %[[VAL_62]] : vector<4xi8>
// CHECK:                   } {affine.vector.load}
// CHECK:                   %[[VAL_63:.*]] = llvm.bitcast %[[VAL_59]] : vector<4xi8> to i32
// CHECK:                   %[[VAL_64:.*]] = arith.addi %[[VAL_63]], %[[VAL_58]] : i32
// CHECK:                   affine.yield %[[VAL_64]] : i32
// CHECK:                 }
// CHECK:                 %[[VAL_65:.*]] = llvm.bitcast %[[VAL_56]] : i32 to vector<4xi8>
// CHECK:                 affine.parallel (%[[VAL_66:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_67:.*]] = vector.extract %[[VAL_65]]{{\[}}%[[VAL_66]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_67]], %[[VAL_11]]{{\[}}%[[VAL_13]], %[[VAL_15]], %[[VAL_66]]] : memref<?x256x4xi8>
// CHECK:                 } {affine.vector.store}
// CHECK:               } {gpu.par.block.x}
// CHECK:             } {gpu.par.grid.x}
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }

