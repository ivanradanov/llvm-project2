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
// CHECK-SAME:                                                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = "memref.ataddr"(%[[VAL_5]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_5]]{{\[}}%[[VAL_0]], 2, %[[VAL_1]]] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_8:.*]] = affine.vector_load %[[VAL_6]][((symbol(%[[VAL_3]]) * 8 + 1) * 2 + 2) * 4 + symbol(%[[VAL_2]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_9:.*]] = llvm.bitcast %[[VAL_8]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_9]] : i32
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
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr) -> i32 {
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
// CHECK-SAME:                                                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = "memref.ataddr"(%[[VAL_5]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_5]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_8]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:           %[[VAL_10:.*]] = affine.vector_load %[[VAL_6]][symbol(%[[VAL_2]]) * 64 + symbol(%[[VAL_3]]) * 4 + 16] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_11:.*]] = llvm.bitcast %[[VAL_10]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_11]] : i32
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
// CHECK-SAME:                                                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = "memref.ataddr"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_4]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_6]]{{\[}}%[[VAL_0]], 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_7]]{{\[}}%[[VAL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
// CHECK:           %[[VAL_9:.*]] = affine.vector_load %[[VAL_5]][symbol(%[[VAL_2]]) * 64 + ((symbol(%[[VAL_2]]) * 8 + 1) * 2 + 2) * 4 + symbol(%[[VAL_2]]) * 4] : memref<?xi8>, vector<4xi8>
// CHECK:           %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : vector<4xi8> to i32
// CHECK:           llvm.return %[[VAL_10]] : i32
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
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                         %[[VAL_1:.*]]: !llvm.ptr) {
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
// CHECK-SAME:                        %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr {
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
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr) -> vector<4xi32> {
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

// llvm.func @nested_affine_scope(%cond : i1, %argptr : !llvm.ptr, %offset : i32) {
//   scf.if %cond {
//     %ub = llvm.load %argptr : !llvm.ptr -> i32
//     %ubc = arith.index_cast %ub : i32 to index
//     affine.for %i = 5 to %ubc {
//       %ic = arith.index_cast %i : index to i32
//       %ptr = llvm.getelementptr %argptr[%ic] : (!llvm.ptr, i32) -> !llvm.ptr, i32
//       %a = llvm.load %ptr : !llvm.ptr -> i32
//     }
//   }
//   llvm.return
// }
