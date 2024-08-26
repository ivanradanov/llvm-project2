add_mlir_library(PolymerSupport
  IslScop.cc
  ScopStmt.cc
  Utils.cc

  DEPENDS
  mlir-headers

  ADDITIONAL_HEADERS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineAnalysis

  PollyISL
  Polly
)

target_link_libraries(PolymerSupport PUBLIC ${GMP_LIBRARY})
