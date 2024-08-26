
set(POLYMER_TRANSFORM_SRC
  IslExternalTransform.cc
)

add_mlir_conversion_library(PolymerTransforms
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  FoldSCFIf.cc
  AnnotateScop.cc
  "${POLYMER_TRANSFORM_SRC}"

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Transforms

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRAffineToStandard
  MLIRAffineTransforms

  PolymerSupport
  "${POLYMER_AVAIL_TARGETS}"
)
