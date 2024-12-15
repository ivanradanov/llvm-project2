
set(INPUTGEN_HEADERS
  dump-input-impl.hpp
  common.hpp
  common-interface.def
)

add_custom_target(inputgen.record)
set_target_properties(inputgen.record PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(inputgen.record
  STATIC
  ARCHS x86_64
  OS linux
  SOURCES record.cpp
  ADDITIONAL_HEADERS ${INPUTGEN_HEADERS}
  OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTSanitizerCommonSymbolizer
  CFLAGS ${COMPILER_RT_COMMON_CFLAGS} -flto -ffat-lto-objects
  LINK_FLAGS ${COMPILER_RT_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
  LINK_LIBS ${STATS_LINK_LIBS}
  PARENT_TARGET inputgen_record)

add_custom_target(inputgen.replay)
set_target_properties(inputgen.replay PROPERTIES FOLDER "Compiler-RT Misc")

add_compiler_rt_runtime(inputgen.replay
  STATIC
  ARCHS x86_64
  OS linux
  SOURCES replay.cpp
  ADDITIONAL_HEADERS ${INPUTGEN_HEADERS}
  OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTSanitizerCommonSymbolizer
  CFLAGS ${COMPILER_RT_COMMON_CFLAGS} -flto -ffat-lto-objects
  LINK_FLAGS ${COMPILER_RT_COMMON_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS}
  LINK_LIBS ${STATS_LINK_LIBS}
  PARENT_TARGET inputgen_replay)
