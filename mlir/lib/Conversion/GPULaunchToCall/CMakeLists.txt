add_mlir_conversion_library(MLIRGPULaunchToCall
  GPULaunchToCall.cpp
  OutlineGpuJitRegions.cpp
  LLVMToMemref.cpp
  PromoteWhile.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GPULaunchToCall

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRPass
  )
