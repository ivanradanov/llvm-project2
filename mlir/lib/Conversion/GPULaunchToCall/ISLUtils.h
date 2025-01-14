#ifndef MLIR_LIB_CONVERSION_GPULAUNCHTOCALL_ISLUTILS_H_
#define MLIR_LIB_CONVERSION_GPULAUNCHTOCALL_ISLUTILS_H_

namespace isl {
class union_map;
}

isl::union_map get_maximal_paths(isl::union_map umap);

#endif
