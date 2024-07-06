// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

llvm.func @basic_struct() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, i32)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

llvm.func @multi_level_direct_two_applications() -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[1, 2, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

llvm.func @multi_level_direct_two_applications(%i1 : i32, %i2 : i32) -> i32 {
  %0 = llvm.mlir.constant(2 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr inbounds %1[%i1, 2, %i2] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}

// -----

llvm.func @func_scope(%arg : !llvm.ptr) -> i32 {
  %2 = llvm.getelementptr inbounds %arg[0, 2, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"foo", (i32, f64, array<10 x i32>, i8)>
  %3 = llvm.load %2 : !llvm.ptr -> i32
  llvm.return %3 : i32
}


// -----

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

llvm.func @ptr_store(%3 : !llvm.ptr, %s : !llvm.ptr) {
  llvm.store %s, %3 : !llvm.ptr, !llvm.ptr
  llvm.return
}

// -----

llvm.func @ptr_load(%3 : !llvm.ptr) -> !llvm.ptr {
  %4 = llvm.load %3 : !llvm.ptr -> !llvm.ptr
  llvm.return %4 : !llvm.ptr
}

// -----

llvm.func @vector_load(%3 : !llvm.ptr) -> vector<4xi32> {
  %4 = llvm.load %3 : !llvm.ptr -> vector<4xi32>
  llvm.return %4 : vector<4xi32>
}

// // -----
//
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
