add_mlir_translation_library(PolymerTargetISL
  ISL/ConvertToISL.cc

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/ISL

  LINK_COMPONENTS
  Core
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRSupport

  PolymerSupport
)
