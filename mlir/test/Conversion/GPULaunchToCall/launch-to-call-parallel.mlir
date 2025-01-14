// RUN: mlir-opt %s --gpu-launch-to-parallel --split-input-file | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, gpu.container_module} {
  gpu.module @__mlir_gpu_module [#nvvm.target<chip = "sm_80">]  {
    llvm.func local_unnamed_addr @_Z10stencil_1dPKiPi(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {convergent, frame_pointer = #llvm.framePointerKind<all>, gpu.kernel, no_unwind, nvvm.kernel, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx84", "+sm_52"]>} {
      %2 = nvvm.read.ptx.sreg.tid.x : i32
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = nvvm.read.ptx.sreg.ntid.x : i32
      "test.test"(%2, %3, %4) : (i32, i32, i32) -> ()
      llvm.return
    }
  }
  llvm.func local_unnamed_addr @foo(%stream : !llvm.ptr, %70 : i64, %71 : i64, %72 : i64, %73 : i64, %74 : i64, %75 : i64, %shm : i32, %a1 : !llvm.ptr, %a2 : !llvm.ptr) -> () {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    gpu.launch_func <%stream : !llvm.ptr> @__mlir_gpu_module::@_Z10stencil_1dPKiPi blocks in (%70, %c1_i64, %c1_i64) threads in (%c256_i64, %c1_i64, %c1_i64) : i64 dynamic_shared_memory_size %c0_i32 args(%a1 : !llvm.ptr, %a2 : !llvm.ptr)
    llvm.return
  }
  llvm.func local_unnamed_addr @foo2(%stream : !llvm.ptr, %70 : i64, %71 : i64, %72 : i64, %73 : i64, %74 : i64, %75 : i64, %shm : i32, %a1 : !llvm.ptr, %a2 : !llvm.ptr) -> () {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    gpu.launch_func @__mlir_gpu_module::@_Z10stencil_1dPKiPi blocks in (%70, %c1_i64, %c1_i64) threads in (%c256_i64, %c1_i64, %c1_i64) : i64 dynamic_shared_memory_size %c0_i32 args(%a1 : !llvm.ptr, %a2 : !llvm.ptr)
    llvm.return
  }
}

// CHECK:       affine.parallel ({{.*}}) = (0, 0, 0) to (symbol(%0), 1, 1) {
// CHECK:         affine.parallel ({{.*}}) = (0, 0, 0) to (256, 1, 1) {
// CHECK:           %1 = arith.index_cast %arg12 : index to i32
// CHECK:           %2 = arith.index_cast %arg9 : index to i32
// CHECK:           "test.test"(%1, %2, %c256_i32) : (i32, i32, i32) -> ()
// CHECK:         } {gpu.par.block}
// CHECK:       } {gpu.par.grid}
// CHECK:       affine.parallel ({{.*}}) = (0, 0, 0) to (symbol(%0), 1, 1) {
// CHECK:         affine.parallel ({{.*}}) = (0, 0, 0) to (256, 1, 1) {
// CHECK:           %1 = arith.index_cast %arg12 : index to i32
// CHECK:           %2 = arith.index_cast %arg9 : index to i32
// CHECK:           "test.test"(%1, %2, %c256_i32) : (i32, i32, i32) -> ()
// CHECK:         } {gpu.par.block}
// CHECK:       } {gpu.par.grid}
// CHECK:       llvm.return
// CHECK:     }
