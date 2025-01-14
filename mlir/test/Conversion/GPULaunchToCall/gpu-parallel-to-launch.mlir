// RUN: mlir-opt %s --gpu-parallel-to-launch
module attributes {dlti.dl_spec = #dlti.dl_spec<f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, gpu.container_module} {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">] {
    llvm.func private local_unnamed_addr @__mlir_par_kernel__Z10stencil_1dPKiPi_0(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c256_i32 = arith.constant 256 : i32
      %0 = arith.index_cast %arg0 : i64 to index
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      scf.parallel (%arg9, %arg10, %arg11) = (%c0, %c0_0, %c0_1) to (%0, %c1, %c1_2) step (%c1_3, %c1_4, %c1_5) {
        %c0_6 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        %c0_9 = arith.constant 0 : index
        %c1_10 = arith.constant 1 : index
        %c1_11 = arith.constant 1 : index
        %c1_12 = arith.constant 1 : index
        %c1_13 = arith.constant 1 : index
        scf.parallel (%arg12, %arg13, %arg14) = (%c0_6, %c0_7, %c0_9) to (%c256, %c1_8, %c1_10) step (%c1_11, %c1_12, %c1_13) {
          %1 = arith.index_cast %arg12 : index to i32
          %2 = arith.index_cast %arg9 : index to i32
          "test.test"(%1, %2, %c256_i32) : (i32, i32, i32) -> ()
          scf.reduce 
        } {gpu.par.block}
        scf.reduce 
      } {gpu.par.grid}
      llvm.return
    }
    llvm.func private local_unnamed_addr @__mlir_par_kernel__Z10stencil_1dPKiPi_1(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, gpu.par.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %c256_i32 = arith.constant 256 : i32
      %0 = arith.index_cast %arg0 : i64 to index
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      scf.parallel (%arg9, %arg10, %arg11) = (%c0, %c0_0, %c0_1) to (%0, %c1, %c1_2) step (%c1_3, %c1_4, %c1_5) {
        %c0_6 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        %c0_9 = arith.constant 0 : index
        %c1_10 = arith.constant 1 : index
        %c1_11 = arith.constant 1 : index
        %c1_12 = arith.constant 1 : index
        %c1_13 = arith.constant 1 : index
        scf.parallel (%arg12, %arg13, %arg14) = (%c0_6, %c0_7, %c0_9) to (%c256, %c1_8, %c1_10) step (%c1_11, %c1_12, %c1_13) {
          %1 = arith.index_cast %arg12 : index to i32
          %2 = arith.index_cast %arg9 : index to i32
          "test.test"(%1, %2, %c256_i32) : (i32, i32, i32) -> ()
          scf.reduce 
        } {gpu.par.block}
        scf.reduce 
      } {gpu.par.grid}
      llvm.return
    }
    llvm.func local_unnamed_addr @_Z10stencil_1dPKiPi(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = nvvm.read.ptx.sreg.ntid.x : i32
      "test.test"(%0, %1, %2) : (i32, i32, i32) -> ()
      llvm.return
    }
  }
  llvm.func local_unnamed_addr @foo(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i32, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    "gpu.call"(%arg1, %c1_i64, %c1_i64, %c256_i64, %c1_i64, %c1_i64, %c0_i32, %arg8, %arg9, %arg0) <{kernel = @__mlir_gpu_module::@__mlir_par_kernel__Z10stencil_1dPKiPi_0, operandSegmentSizes = array<i32: 0, 9, 1>}> : (i64, i64, i64, i64, i64, i64, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @foo2(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i32, %arg8: !llvm.ptr, %arg9: !llvm.ptr) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    "gpu.call"(%arg1, %c1_i64, %c1_i64, %c256_i64, %c1_i64, %c1_i64, %c0_i32, %arg8, %arg9) <{kernel = @__mlir_gpu_module::@__mlir_par_kernel__Z10stencil_1dPKiPi_1, operandSegmentSizes = array<i32: 0, 9, 0>}> : (i64, i64, i64, i64, i64, i64, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}

