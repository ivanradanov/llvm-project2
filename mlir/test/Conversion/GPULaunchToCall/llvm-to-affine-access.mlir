// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

// CHECK-LABEL:   llvm.func @basic_struct() -> i32 {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
// CHECK:           %[[VAL_4:.*]] = affine.vector_load %[[VAL_2]][16] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_5]] : i32
// CHECK:         }
llvm.func @basic_struct() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL:   llvm.func @multi_level_direct_two_applications() -> i32 {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][1, 2, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_4:.*]] = affine.vector_load %[[VAL_2]][100] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_5]] : i32
// CHECK:         }
llvm.func @multi_level_direct_two_applications() -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[1, 2, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL:   llvm.func @multi_level_direct_two_applications(
// CHECK-SAME:                                                   %[[VAL_0:[^:]*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:[^:]*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = "affine.scope"(%[[VAL_0]], %[[VAL_1]]) ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32):
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:             %[[VAL_6:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:             %[[VAL_7:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:             %[[VAL_9:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_0]], 2, %[[VAL_1]]] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:             %[[VAL_11:.*]] = affine.vector_load %[[VAL_9]][((symbol(%[[VAL_6]]) * 8 + 1) * 2 + 2) * 4 + symbol(%[[VAL_5]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:             %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : vector<4xi8> to i32
// CHECK:             affine.yield %[[VAL_12]] : i32
// CHECK:           }) : (i32, i32) -> i32
// CHECK:           llvm.return %[[VAL_2]] : i32
// CHECK:         }
llvm.func @multi_level_direct_two_applications(%i1 : i32, %i2 : i32) -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[%i1, 2, %i2] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL:   llvm.func @func_scope(
// CHECK-SAME:                          %[[VAL_0:[^:]*]]: !llvm.ptr) -> i32 {
// CHECK:           %[[VAL_1:.*]] = "memref.ataddr"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_3:.*]] = affine.vector_load %[[VAL_1]][28] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_4:.*]] = llvm.bitcast %[[VAL_3]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_4]] : i32
// CHECK:         }
llvm.func @func_scope(%arg : !llvm.ptr) -> i32 {
  %2 = llvm.getelementptr inbounds %arg[0, 2, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}


// -----

// CHECK-LABEL:   llvm.func @multi_level_direct_two_applications(
// CHECK-SAME:                                                   %[[VAL_0:[^:]*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:[^:]*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = "affine.scope"(%[[VAL_1]], %[[VAL_0]]) ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32):
// CHECK:             %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:             %[[VAL_6:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:             %[[VAL_7:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:             %[[VAL_9:.*]] = "memref.ataddr"(%[[VAL_8]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:             %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:             %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:             %[[VAL_13:.*]] = affine.vector_load %[[VAL_9]][symbol(%[[VAL_5]]) * 64 + symbol(%[[VAL_6]]) * 4 + 16] : memref<?xi8>, vector<4xi8>
// CHECK:             %[[VAL_14:.*]] = llvm.bitcast %[[VAL_13]] : vector<4xi8> to i32
// CHECK:             affine.yield %[[VAL_14]] : i32
// CHECK:           }) : (i32, i32) -> i32
// CHECK:           llvm.return %[[VAL_2]] : i32
// CHECK:         }
llvm.func @multi_level_direct_two_applications(%i1 : i32, %i2 : i32) -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[%i1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %22 = llvm.getelementptr inbounds %2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.getelementptr inbounds %22[%i2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
  %4 = llvm.load %3 : !llvm.ptr -> i32
  llvm.return %4 : i32
}

// -----

// CHECK-LABEL:   llvm.func @multi_level_direct_two_applications(
// CHECK-SAME:                                                   %[[VAL_0:[^:]*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:[^:]*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = "affine.scope"(%[[VAL_0]]) ({
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32):
// CHECK:             %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:             %[[VAL_5:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:             %[[VAL_7:.*]] = "memref.ataddr"(%[[VAL_6]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:             %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_6]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:             %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_0]], 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:             %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:             %[[VAL_11:.*]] = affine.vector_load %[[VAL_7]][symbol(%[[VAL_4]]) * 64 + ((symbol(%[[VAL_4]]) * 8 + 1) * 2 + 2) * 4 + symbol(%[[VAL_4]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:             %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : vector<4xi8> to i32
// CHECK:             affine.yield %[[VAL_12]] : i32
// CHECK:           }) : (i32) -> i32
// CHECK:           llvm.return %[[VAL_2]] : i32
// CHECK:         }
llvm.func @multi_level_direct_two_applications(%i1 : i32, %i2 : i32) -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[%i1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %22 = llvm.getelementptr inbounds %2[%i1, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.getelementptr inbounds %22[%i1] : (!llvm.ptr, i32) -> !llvm.ptr, i32
  %4 = llvm.load %3 : !llvm.ptr -> i32
  llvm.return %4 : i32
}

// -----

// CHECK-LABEL:   llvm.func @ptr_store(
// CHECK-SAME:                         %[[VAL_0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                         %[[VAL_1:[^:]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_2:.*]] = "memref.ataddr"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr to i64
// CHECK:           %[[VAL_4:.*]] = llvm.bitcast %[[VAL_3]] : i64 to vector<8xi8>
// CHECK:           affine.vector_store %[[VAL_4]], %[[VAL_2]][0] : memref<?xi8>, vector<8xi8>
// CHECK:           llvm.return
// CHECK:         }
llvm.func @ptr_store(%3 : !llvm.ptr, %s : !llvm.ptr) {
  llvm.store %s, %3 : !llvm.ptr, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @ptr_load(
// CHECK-SAME:                        %[[VAL_0:[^:]*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           %[[VAL_1:.*]] = "memref.ataddr"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_2:.*]] = affine.vector_load %[[VAL_1]][0] : memref<?xi8>, vector<8xi8>
// CHECK:           %[[VAL_3:.*]] = llvm.bitcast %[[VAL_2]] : vector<8xi8> to i64
// CHECK:           %[[VAL_4:.*]] = llvm.inttoptr %[[VAL_3]] : i64 to !llvm.ptr
// CHECK:           llvm.return %[[VAL_4]] : !llvm.ptr
// CHECK:         }
llvm.func @ptr_load(%3 : !llvm.ptr) -> !llvm.ptr {
  %4 = llvm.load %3 : !llvm.ptr -> !llvm.ptr
  llvm.return %4 : !llvm.ptr
}

// -----

// CHECK-LABEL:   llvm.func @vector_load(
// CHECK-SAME:                           %[[VAL_0:[^:]*]]: !llvm.ptr) -> vector<4xi32> {
// CHECK:           %[[VAL_1:.*]] = "memref.ataddr"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_2:.*]] = affine.vector_load %[[VAL_1]][0] : memref<?xi8>, vector<16xi8>
// CHECK:           %[[VAL_3:.*]] = llvm.bitcast %[[VAL_2]] : vector<16xi8> to vector<4xi32>
// CHECK:           llvm.return %[[VAL_3]] : vector<4xi32>
// CHECK:         }
llvm.func @vector_load(%3 : !llvm.ptr) -> vector<4xi32> {
  %4 = llvm.load %3 : !llvm.ptr -> vector<4xi32>
  llvm.return %4 : vector<4xi32>
}

// -----

// CHECK-LABEL:   llvm.func @nested_symbol_op(
// CHECK-SAME:                                %[[VAL_0:[^:]*]]: i1,
// CHECK-SAME:                                %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_2:[^:]*]]: i32) {
// CHECK:           %[[VAL_3:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           scf.if %[[VAL_0]] {
// CHECK:             "affine.scope"() ({
// CHECK:               %[[VAL_4:.*]] = affine.vector_load %[[VAL_3]][0] : memref<?xi8>, vector<4xi8>
// CHECK:               %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : vector<4xi8> to i32
// CHECK:               %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:               affine.for %[[VAL_7:.*]] = 5 to 100 {
// CHECK:                 %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i32
// CHECK:                 %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_5]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_10:.*]] = affine.vector_load %[[VAL_3]][symbol(%[[VAL_6]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:                 %[[VAL_11:.*]] = llvm.bitcast %[[VAL_10]] : vector<4xi8> to i32
// CHECK:               }
// CHECK:               affine.yield
// CHECK:             }) : () -> ()
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }
llvm.func @nested_symbol_op(%cond : i1, %argptr : !llvm.ptr, %offset : i32) {
  scf.if %cond {
    %ub = llvm.load %argptr : !llvm.ptr -> i32
    affine.for %i = 5 to 100 {
      %ic = arith.index_cast %i : index to i32
      %ptr = llvm.getelementptr %argptr[%ub] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %a = llvm.load %ptr : !llvm.ptr -> i32
    }
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @nested_symbol_op(
// CHECK-SAME:                                %[[VAL_0:[^:]*]]: i1,
// CHECK-SAME:                                %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_2:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_3:[^:]*]]: i32) {
// CHECK:           %[[VAL_4:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_5:.*]] = "memref.ataddr"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_7]] {
// CHECK:             "affine.scope"(%[[VAL_9]]) ({
// CHECK:             ^bb0(%[[VAL_10:.*]]: index):
// CHECK:               %[[VAL_11:.*]] = affine.vector_load %[[VAL_4]][0] : memref<?xi8>, vector<4xi8>
// CHECK:               %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : vector<4xi8> to i32
// CHECK:               affine.for %[[VAL_13:.*]] = 5 to 100 {
// CHECK:                 %[[VAL_14:.*]] = arith.index_cast %[[VAL_9]] : index to i32
// CHECK:                 %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_14]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_14]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_17:.*]] = affine.vector_load %[[VAL_4]][symbol(%[[VAL_10]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:                 %[[VAL_18:.*]] = llvm.bitcast %[[VAL_17]] : vector<4xi8> to i32
// CHECK:                 %[[VAL_19:.*]] = llvm.bitcast %[[VAL_18]] : i32 to vector<4xi8>
// CHECK:                 affine.vector_store %[[VAL_19]], %[[VAL_5]][symbol(%[[VAL_10]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:               }
// CHECK:               affine.yield
// CHECK:             }) : (index) -> ()
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }
llvm.func @nested_symbol_op(%cond : i1, %argptr : !llvm.ptr, %argptrs : !llvm.ptr, %offset : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %k = %c0 to %c10 step %c1 {
    %ub = llvm.load %argptr : !llvm.ptr -> i32
    affine.for %i = 5 to 100 {
      %ic = arith.index_cast %k : index to i32
      %ptr = llvm.getelementptr %argptr[%ic] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %ptrs = llvm.getelementptr %argptrs[%ic] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %a = llvm.load %ptr : !llvm.ptr -> i32
      llvm.store %a, %ptrs : i32, !llvm.ptr
    }
    scf.yield
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @nested_symbol_op(
// CHECK-SAME:                                %[[VAL_0:[^:]*]]: i1,
// CHECK-SAME:                                %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_2:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_3:[^:]*]]: i32) {
// CHECK:           %[[VAL_4:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:           affine.for %[[VAL_8:.*]] = 5 to 100 {
// CHECK:             "affine.scope"() ({
// CHECK:               %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : index to i32
// CHECK:               %[[VAL_10:.*]] = "test.test1"() : () -> i32
// CHECK:               %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:               affine.for %[[VAL_12:.*]] = 5 to 100 {
// CHECK:                 %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i32
// CHECK:                 %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_9]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_13]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_15]]{{\[}}%[[VAL_10]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:                 %[[VAL_17:.*]] = affine.vector_load %[[VAL_4]]{{\[}}%[[VAL_8]] * 4 + %[[VAL_12]] * 4 + symbol(%[[VAL_11]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:                 %[[VAL_18:.*]] = llvm.bitcast %[[VAL_17]] : vector<4xi8> to i32
// CHECK:               }
// CHECK:               affine.yield
// CHECK:             }) : () -> ()
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }
llvm.func @nested_symbol_op(%cond : i1, %argptr : !llvm.ptr, %argptrs : !llvm.ptr, %offset : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  affine.for %i = 5 to 100 {
    %ic = arith.index_cast %i : index to i32
    %sym = "test.test1"() : () -> i32
    affine.for %j = 5 to 100 {
      %jc = arith.index_cast %j : index to i32
      %ptr1 = llvm.getelementptr %argptr[%ic] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %ptr2 = llvm.getelementptr %ptr1[%jc] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %ptr3 = llvm.getelementptr %ptr2[%sym] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %a = llvm.load %ptr3 : !llvm.ptr -> i32
    }
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @use_outside_affine_scope(
// CHECK-SAME:                                        %[[VAL_0:[^:]*]]: i1,
// CHECK-SAME:                                        %[[VAL_1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:                                        %[[VAL_2:[^:]*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = "memref.ataddr"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_4:.*]] = scf.if %[[VAL_0]] -> (i32) {
// CHECK:             %[[VAL_5:.*]] = "affine.scope"() ({
// CHECK:               %[[VAL_6:.*]] = "test.test1"() : () -> i32
// CHECK:               %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:               %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_6]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:               %[[VAL_9:.*]] = affine.vector_load %[[VAL_3]][symbol(%[[VAL_7]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:               %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : vector<4xi8> to i32
// CHECK:               affine.yield %[[VAL_10]] : i32
// CHECK:             }) : () -> i32
// CHECK:             scf.yield %[[VAL_5]] : i32
// CHECK:           } else {
// CHECK:             %[[VAL_11:.*]] = affine.vector_load %[[VAL_3]][0] : memref<?xi8>, vector<4xi8>
// CHECK:             %[[VAL_12:.*]] = llvm.bitcast %[[VAL_11]] : vector<4xi8> to i32
// CHECK:             scf.yield %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           llvm.return %[[VAL_4]] : i32
// CHECK:         }
llvm.func @use_outside_affine_scope(%cond : i1, %argptr : !llvm.ptr, %offset : i32) -> i32 {
  %y = scf.if %cond -> i32{
    %sym = "test.test1"() : () -> i32
    %ptr = llvm.getelementptr %argptr[%sym] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %ub = llvm.load %ptr : !llvm.ptr -> i32
    scf.yield %ub : i32
  } else {
    %ub = llvm.load %argptr : !llvm.ptr -> i32
    scf.yield %ub : i32
  }
  llvm.return %y : i32
}
