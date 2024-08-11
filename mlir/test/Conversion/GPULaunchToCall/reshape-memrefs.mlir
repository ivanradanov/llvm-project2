// RUN: mlir-opt %s --reshape-memrefs --split-input-file | FileCheck %s

module {
  llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
    %c0_i32 = arith.constant 0 : i32
    %c7_i32 = arith.constant 7 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = "memref.ataddr"(%arg7) : (!llvm.ptr) -> memref<?xi8>
    %1 = "memref.ataddr"(%arg8) : (!llvm.ptr) -> memref<?xi8>
    %2 = arith.index_cast %arg0 : i64 to index
    affine.parallel (%arg9) = (0) to (symbol(%2)) {
      %alloca = memref.alloca() : memref<1080xi8, 3>
      affine.parallel (%arg10) = (0) to (256) {
        %3 = arith.index_cast %arg10 : index to i32
        %4 = arith.index_cast %arg9 : index to i32
        %5 = arith.muli %4, %c256_i32 : i32
        %6 = arith.addi %5, %3 : i32
        %7 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
          %11 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11] : memref<?xi8>
          %12 = vector.broadcast %11 : i8 to vector<4xi8>
          affine.yield %12 : vector<4xi8>
        } {affine.vector.load}
        affine.parallel (%arg11) = (0) to (4) {
          %11 = vector.extract %7[%arg11] : i8 from vector<4xi8>
          affine.store %11, %alloca[%arg10 * 4 + %arg11 + 28] : memref<1080xi8, 3>
        } {affine.vector.store}
        %8 = arith.cmpi ult, %3, %c7_i32 : i32
        scf.if %8 {
          %11 = arith.cmpi slt, %6, %c7_i32 : i32
          %12 = scf.if %11 -> (i32) {
            scf.yield %c0_i32 : i32
          } else {
            %15 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
              %17 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11 - 28] : memref<?xi8>
              %18 = vector.broadcast %17 : i8 to vector<4xi8>
              affine.yield %18 : vector<4xi8>
            } {affine.vector.load}
            %16 = llvm.bitcast %15 : vector<4xi8> to i32
            scf.yield %16 : i32
          }
          %13 = llvm.bitcast %12 : i32 to vector<4xi8>
          affine.parallel (%arg11) = (0) to (4) {
            %15 = vector.extract %13[%arg11] : i8 from vector<4xi8>
            affine.store %15, %alloca[%arg10 * 4 + %arg11] : memref<1080xi8, 3>
          } {affine.vector.store}
          %14 = affine.parallel (%arg11) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
            %15 = affine.load %0[%arg9 * 1024 + %arg10 * 4 + %arg11 + 1024] : memref<?xi8>
            %16 = vector.broadcast %15 : i8 to vector<4xi8>
            affine.yield %16 : vector<4xi8>
          } {affine.vector.load}
          affine.parallel (%arg11) = (0) to (4) {
            %15 = vector.extract %14[%arg11] : i8 from vector<4xi8>
            affine.store %15, %alloca[%arg10 * 4 + %arg11 + 1052] : memref<1080xi8, 3>
          } {affine.vector.store}
        }
        nvvm.barrier0
        %9 = affine.for %arg11 = -7 to 8 iter_args(%arg12 = %c0_i32) -> (i32) {
          %11 = affine.parallel (%arg13) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
            %14 = affine.load %alloca[%arg11 * 4 + %arg10 * 4 + %arg13 + 28] : memref<1080xi8, 3>
            %15 = vector.broadcast %14 : i8 to vector<4xi8>
            affine.yield %15 : vector<4xi8>
          } {affine.vector.load}
          %12 = llvm.bitcast %11 : vector<4xi8> to i32
          %13 = arith.addi %12, %arg12 : i32
          affine.yield %13 : i32
        }
        %10 = llvm.bitcast %9 : i32 to vector<4xi8>
        affine.parallel (%arg11) = (0) to (4) {
          %11 = vector.extract %10[%arg11] : i8 from vector<4xi8>
          affine.store %11, %1[%arg9 * 1024 + %arg10 * 4 + %arg11] : memref<?xi8>
        } {affine.vector.store}
      } {gpu.par.block.x}
    } {gpu.par.grid.x}
    llvm.return
  }
}

// CHECK-LABEL:   llvm.func private local_unnamed_addr @__mlir.par.kernel._Z10stencil_1dPKiPi(
// CHECK-SAME:                                                                                %[[VAL_0:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_1:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_2:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_3:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_4:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_5:[^:]*]]: i64,
// CHECK-SAME:                                                                                %[[VAL_6:[^:]*]]: i32,
// CHECK-SAME:                                                                                %[[VAL_7:[^:]*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly},
// CHECK-SAME:                                                                                %[[VAL_8:[^:]*]]: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #[[?]]<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #[[?]]<["+ptx84", "+sm_52"]>} {
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 7 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_12:.*]] = "memref.ataddr"(%[[VAL_7]]) : (!llvm.ptr) -> memref<?x4xi8>
// CHECK:           %[[VAL_13:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?x256x4xi8>
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_0]] : i64 to index
// CHECK:           affine.parallel (%[[VAL_15:.*]]) = (0) to (symbol(%[[VAL_14]])) {
// CHECK:             %[[VAL_16:.*]] = memref.alloca() : memref<270x4xi8, 3>
// CHECK:             affine.parallel (%[[VAL_17:.*]]) = (0) to (256) {
// CHECK:               %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : index to i32
// CHECK:               %[[VAL_19:.*]] = arith.index_cast %[[VAL_15]] : index to i32
// CHECK:               %[[VAL_20:.*]] = arith.muli %[[VAL_19]], %[[VAL_11]] : i32
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_18]] : i32
// CHECK:               %[[VAL_22:.*]] = affine.parallel (%[[VAL_23:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                 %[[VAL_24:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_15]] * 256 + %[[VAL_17]], %[[VAL_23]]] : memref<?x4xi8>
// CHECK:                 %[[VAL_25:.*]] = vector.broadcast %[[VAL_24]] : i8 to vector<4xi8>
// CHECK:                 affine.yield %[[VAL_25]] : vector<4xi8>
// CHECK:               } {affine.vector.load}
// CHECK:               affine.parallel (%[[VAL_26:.*]]) = (0) to (4) {
// CHECK:                 %[[VAL_27:.*]] = vector.extract %[[VAL_22]]{{\[}}%[[VAL_26]]] : i8 from vector<4xi8>
// CHECK:                 affine.store %[[VAL_27]], %[[VAL_16]]{{\[}}%[[VAL_17]] + 7, %[[VAL_26]]] : memref<270x4xi8, 3>
// CHECK:               } {affine.vector.store}
// CHECK:               %[[VAL_28:.*]] = arith.cmpi ult, %[[VAL_18]], %[[VAL_10]] : i32
// CHECK:               scf.if %[[VAL_28]] {
// CHECK:                 %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_10]] : i32
// CHECK:                 %[[VAL_30:.*]] = scf.if %[[VAL_29]] -> (i32) {
// CHECK:                   scf.yield %[[VAL_9]] : i32
// CHECK:                 } else {
// CHECK:                   %[[VAL_31:.*]] = affine.parallel (%[[VAL_32:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                     %[[VAL_33:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_15]] * 256 + %[[VAL_17]] - 7, %[[VAL_32]]] : memref<?x4xi8>
// CHECK:                     %[[VAL_34:.*]] = vector.broadcast %[[VAL_33]] : i8 to vector<4xi8>
// CHECK:                     affine.yield %[[VAL_34]] : vector<4xi8>
// CHECK:                   } {affine.vector.load}
// CHECK:                   %[[VAL_35:.*]] = llvm.bitcast %[[VAL_31]] : vector<4xi8> to i32
// CHECK:                   scf.yield %[[VAL_35]] : i32
// CHECK:                 }
// CHECK:                 %[[VAL_36:.*]] = llvm.bitcast %[[VAL_30]] : i32 to vector<4xi8>
// CHECK:                 affine.parallel (%[[VAL_37:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_38:.*]] = vector.extract %[[VAL_36]]{{\[}}%[[VAL_37]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_38]], %[[VAL_16]]{{\[}}%[[VAL_17]], %[[VAL_37]]] : memref<270x4xi8, 3>
// CHECK:                 } {affine.vector.store}
// CHECK:                 %[[VAL_39:.*]] = affine.parallel (%[[VAL_40:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                   %[[VAL_41:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_15]] * 256 + %[[VAL_17]] + 256, %[[VAL_40]]] : memref<?x4xi8>
// CHECK:                   %[[VAL_42:.*]] = vector.broadcast %[[VAL_41]] : i8 to vector<4xi8>
// CHECK:                   affine.yield %[[VAL_42]] : vector<4xi8>
// CHECK:                 } {affine.vector.load}
// CHECK:                 affine.parallel (%[[VAL_43:.*]]) = (0) to (4) {
// CHECK:                   %[[VAL_44:.*]] = vector.extract %[[VAL_39]]{{\[}}%[[VAL_43]]] : i8 from vector<4xi8>
// CHECK:                   affine.store %[[VAL_44]], %[[VAL_16]]{{\[}}%[[VAL_17]] + 263, %[[VAL_43]]] : memref<270x4xi8, 3>
// CHECK:                 } {affine.vector.store}
// CHECK:               }
// CHECK:               nvvm.barrier0
// CHECK:               %[[VAL_45:.*]] = affine.for %[[VAL_46:.*]] = -7 to 8 iter_args(%[[VAL_47:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:                 %[[VAL_48:.*]] = affine.parallel (%[[VAL_49:.*]]) = (0) to (4) reduce ("vector_insert") -> (vector<4xi8>) {
// CHECK:                   %[[VAL_50:.*]] = affine.load %[[VAL_16]]{{\[}}%[[VAL_46]] + %[[VAL_17]] + 7, %[[VAL_49]]] : memref<270x4xi8, 3>
// CHECK:                   %[[VAL_51:.*]] = vector.broadcast %[[VAL_50]] : i8 to vector<4xi8>
// CHECK:                   affine.yield %[[VAL_51]] : vector<4xi8>
// CHECK:                 } {affine.vector.load}
// CHECK:                 %[[VAL_52:.*]] = llvm.bitcast %[[VAL_48]] : vector<4xi8> to i32
// CHECK:                 %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_47]] : i32
// CHECK:                 affine.yield %[[VAL_53]] : i32
// CHECK:               }
// CHECK:               %[[VAL_54:.*]] = llvm.bitcast %[[VAL_45]] : i32 to vector<4xi8>
// CHECK:               affine.parallel (%[[VAL_55:.*]]) = (0) to (4) {
// CHECK:                 %[[VAL_56:.*]] = vector.extract %[[VAL_54]]{{\[}}%[[VAL_55]]] : i8 from vector<4xi8>
// CHECK:                 affine.store %[[VAL_56]], %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_17]], %[[VAL_55]]] : memref<?x256x4xi8>
// CHECK:               } {affine.vector.store}
// CHECK:             } {gpu.par.block.x}
// CHECK:           } {gpu.par.grid.x}
// CHECK:           llvm.return
// CHECK:         }

