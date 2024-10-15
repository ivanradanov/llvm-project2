#set = affine_set<(d0) : (d0 - 8 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 * 256 + d1 - 8 >= 0)>
module {
  gpu.module @m {
    llvm.func private @stencil(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr, %arg8: !llvm.ptr) attributes {gpu.par.kernel, sym_visibility = "private"} {
      %c0_i32 = arith.constant 0 : i32
      %0 = "memref.ataddr"(%arg7) : (!llvm.ptr) -> memref<?xi8>
      %1 = "memref.ataddr"(%arg8) : (!llvm.ptr) -> memref<?xi8>
      %2 = arith.index_cast %arg0 : i64 to index
      affine.parallel (%arg9, %u0, %u1) = (0, 0, 0) to (symbol(%2), 1, 1) {
        %alloca = memref.alloca() : memref<1080xi8, 3>
        affine.parallel (%arg10, %u2, %u3) = (0, 0, 0) to (256, 1, 1) {
          %3 = affine.vector_load %0[%arg9 * 1024 + %arg10 * 4] : memref<?xi8>, vector<4xi8>
          affine.vector_store %3, %alloca[%arg10 * 4 + 28] : memref<1080xi8, 3>, vector<4xi8>
          affine.if #set(%arg10) {
            %6 = affine.if #set1(%arg9, %arg10) -> i32 {
              affine.yield %c0_i32 : i32
            } else {
              %9 = affine.vector_load %0[%arg9 * 1024 + %arg10 * 4 - 28] : memref<?xi8>, vector<4xi8>
              %10 = llvm.bitcast %9 : vector<4xi8> to i32
              affine.yield %10 : i32
            }
            %7 = llvm.bitcast %6 : i32 to vector<4xi8>
            affine.vector_store %7, %alloca[%arg10 * 4] : memref<1080xi8, 3>, vector<4xi8>
            %8 = affine.vector_load %0[%arg9 * 1024 + %arg10 * 4 + 1024] : memref<?xi8>, vector<4xi8>
            affine.vector_store %8, %alloca[%arg10 * 4 + 1052] : memref<1080xi8, 3>, vector<4xi8>
          }
          "affine.barrier"(%arg10, %u2, %u3) : (index, index, index) -> ()
          %4 = affine.for %arg11 = 0 to 15 iter_args(%arg12 = %c0_i32) -> (i32) {
            %6 = affine.vector_load %alloca[%arg11 * 4 + %arg10 * 4] : memref<1080xi8, 3>, vector<4xi8>
            %7 = llvm.bitcast %6 : vector<4xi8> to i32
            %8 = arith.addi %7, %arg12 : i32
            affine.yield %8 : i32
          }
          %5 = llvm.bitcast %4 : i32 to vector<4xi8>
          affine.vector_store %5, %1[%arg9 * 1024 + %arg10 * 4] : memref<?xi8>, vector<4xi8>
        } {gpu.par.block}
      } {gpu.par.grid}
      llvm.return
    }
  }
}

