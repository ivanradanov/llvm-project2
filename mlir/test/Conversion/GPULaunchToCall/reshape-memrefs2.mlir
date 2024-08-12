// RUN: mlir-opt %s --reshape-memrefs --split-input-file | FileCheck %s

#set = affine_set<(d0) : (-d0 + 6 >= 0)>
#set1 = affine_set<(d0, d1) : (-(d0 * 256 + d1) + 6 >= 0)>
module {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c0_i32 = arith.constant 0 : i32
      %0 = "memref.ataddr"(%arg7) : (!llvm.ptr) -> memref<?xi8>
      %1 = "memref.ataddr"(%arg8) : (!llvm.ptr) -> memref<?xi8>
      %2 = arith.index_cast %arg0 : i64 to index
      affine.parallel (%arg9) = (0) to (symbol(%2)) {
        %alloca = memref.alloca() : memref<1080xi8, 3>
        affine.parallel (%arg10) = (0) to (256) {
          %3 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
            %6 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11] : memref<?xi8>
            %7 = vector.broadcast %6 : i8 to vector<4xi8>
            affine.yield %7 : vector<4xi8>
          } {affine.vector.load}
          affine.parallel (%arg11) = (0) to (4) {
            %6 = vector.extract %3[%arg11] : i8 from vector<4xi8>
            affine.store %6, %alloca[%arg10 * 4 + %arg11 + 28] : memref<1080xi8, 3>
          } {affine.vector.store}
          affine.if #set(%arg10) {
            %6 = affine.if #set1(%arg9, %arg10) -> i32 {
              affine.yield %c0_i32 : i32
            } else {
              %9 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
                %11 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11 - 28] : memref<?xi8>
                %12 = vector.broadcast %11 : i8 to vector<4xi8>
                affine.yield %12 : vector<4xi8>
              } {affine.vector.load}
              %10 = llvm.bitcast %9 : vector<4xi8> to i32
              affine.yield %10 : i32
            }
            %7 = llvm.bitcast %6 : i32 to vector<4xi8>
            affine.parallel (%arg11) = (0) to (4) {
              %9 = vector.extract %7[%arg11] : i8 from vector<4xi8>
              affine.store %9, %alloca[%arg10 * 4 + %arg11] : memref<1080xi8, 3>
            } {affine.vector.store}
            %8 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
              %9 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11 + 1024] : memref<?xi8>
              %10 = vector.broadcast %9 : i8 to vector<4xi8>
              affine.yield %10 : vector<4xi8>
            } {affine.vector.load}
            affine.parallel (%arg11) = (0) to (4) {
              %9 = vector.extract %8[%arg11] : i8 from vector<4xi8>
              affine.store %9, %alloca[%arg10 * 4 + %arg11 + 1052] : memref<1080xi8, 3>
            } {affine.vector.store}
          }
          nvvm.barrier0
          %4 = affine.for %arg11 = -7 to 8 iter_args(%arg12 = %c0_i32) -> (i32) {
            %6 = affine.parallel (%arg13) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
              %9 = affine.load %alloca[%arg11 * 4 + %arg10 * 4 + %arg13 + 28] : memref<1080xi8, 3>
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
            affine.store %6, %1[%arg9 * 1024 + %arg10 * 4 + %arg11] : memref<?xi8>
          } {affine.vector.store}
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
// CHECK:             %[[VAL_10:.*]] = "memref.ataddr"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?x4xi8>
// CHECK:             %[[VAL_11:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?x256x4xi8>
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK:             affine.parallel (%[[VAL_13:.*]]) = (0) to (symbol(%[[VAL_12]])) {
// CHECK:               %[[VAL_14:.*]] = memref.alloca() : memref<270x4xi8, 3>
// CHECK:               affine.parallel (%[[VAL_15:.*]]) = (0) to (256) {
// CHECK:                 %[[VAL_16:.*]] = affine.parallel (%[[VAL_17:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                   %[[VAL_18:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 256 + %[[VAL_15]], %[[VAL_17]]] : memref<?x4xi8>
// CHECK:                   %[[VAL_19:.*]] = vector.broadcast %[[VAL_18]] : i8 to vector<4xi8>
// CHECK:                   affine.yield %[[VAL_19]] : vector<4xi8>
// CHECK:                 } {affine.vector.load}
// CHECK:                 affine.parallel (%[[VAL_20:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_21:.*]] = vector.extract %[[VAL_16]]{{\[}}%[[VAL_20]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_21]], %[[VAL_14]]{{\[}}%[[VAL_15]] + 7, %[[VAL_20]]] : memref<270x4xi8, 3>
// CHECK:                 } {affine.vector.store}
// CHECK:                 affine.if #[[$ATTR_0]](%[[VAL_15]]) {
// CHECK:                   %[[VAL_22:.*]] = affine.if #[[$ATTR_1]](%[[VAL_13]], %[[VAL_15]]) -> i32 {
// CHECK:                     affine.yield %[[VAL_9]] : i32
// CHECK:                   } else {
// CHECK:                     %[[VAL_23:.*]] = affine.parallel (%[[VAL_24:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                       %[[VAL_25:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 256 + %[[VAL_15]] - 7, %[[VAL_24]]] : memref<?x4xi8>
// CHECK:                       %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i8 to vector<4xi8>
// CHECK:                       affine.yield %[[VAL_26]] : vector<4xi8>
// CHECK:                     } {affine.vector.load}
// CHECK:                     %[[VAL_27:.*]] = llvm.bitcast %[[VAL_23]] : vector<4xi8> to i32
// CHECK:                     affine.yield %[[VAL_27]] : i32
// CHECK:                   }
// CHECK:                   %[[VAL_28:.*]] = llvm.bitcast %[[VAL_22]] : i32 to vector<4xi8>
// CHECK:                   affine.parallel (%[[VAL_29:.*]]) = (0) to (4) {
// CHECK:                     %[[VAL_30:.*]] = vector.extract %[[VAL_28]]{{\[}}%[[VAL_29]]] : i8 from vector<4xi8>
// CHECK:                     affine.store %[[VAL_30]], %[[VAL_14]]{{\[}}%[[VAL_15]], %[[VAL_29]]] : memref<270x4xi8, 3>
// CHECK:                   } {affine.vector.store}
// CHECK:                   %[[VAL_31:.*]] = affine.parallel (%[[VAL_32:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_33:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_13]] * 256 + %[[VAL_15]] + 256, %[[VAL_32]]] : memref<?x4xi8>
// CHECK:                     %[[VAL_34:.*]] = vector.broadcast %[[VAL_33]] : i8 to vector<4xi8>
// CHECK:                     affine.yield %[[VAL_34]] : vector<4xi8>
// CHECK:                   } {affine.vector.load}
// CHECK:                   affine.parallel (%[[VAL_35:.*]]) = (0) to (4) {
// CHECK:                     %[[VAL_36:.*]] = vector.extract %[[VAL_31]]{{\[}}%[[VAL_35]]] : i8 from vector<4xi8>
// CHECK:                     affine.store %[[VAL_36]], %[[VAL_14]]{{\[}}%[[VAL_15]] + 263, %[[VAL_35]]] : memref<270x4xi8, 3>
// CHECK:                   } {affine.vector.store}
// CHECK:                 }
// CHECK:                 nvvm.barrier0
// CHECK:                 %[[VAL_37:.*]] = affine.for %[[VAL_38:.*]] = -7 to 8 iter_args(%[[VAL_39:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:                   %[[VAL_40:.*]] = affine.parallel (%[[VAL_41:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_42:.*]] = affine.load %[[VAL_14]]{{\[}}%[[VAL_38]] + %[[VAL_15]] + 7, %[[VAL_41]]] : memref<270x4xi8, 3>
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
// CHECK:                   affine.store %[[VAL_48]], %[[VAL_11]]{{\[}}%[[VAL_13]], %[[VAL_15]], %[[VAL_47]]] : memref<?x256x4xi8>
// CHECK:                 } {affine.vector.store}
// CHECK:               } {gpu.par.block.x}
// CHECK:             } {gpu.par.grid.x}
// CHECK:             llvm.return
// CHECK:           }
// CHECK:         }

