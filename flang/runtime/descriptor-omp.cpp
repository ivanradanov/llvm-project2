#include "flang/Runtime/descriptor.h"

#include <omp.h>

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS int Descriptor::AllocateTarget(OMPDeviceTy ompDevice) {
  std::size_t byteSize{Elements() * ElementBytes()};
  // Zero size allocation is possible in Fortran and the resulting
  // descriptor must be allocated/associated. Since std::malloc(0)
  // result is implementation defined, always allocate at least one byte.
  void *p{byteSize ? omp_target_alloc(byteSize, ompDevice)
                   : omp_target_alloc(1, ompDevice)};
  if (!p) {
    return CFI_ERROR_MEM_ALLOCATION;
  }
  // TODO: image synchronization
  raw_.base_addr = p;
  if (int dims{rank()}) {
    std::size_t stride{ElementBytes()};
    for (int j{0}; j < dims; ++j) {
      auto &dimension{GetDimension(j)};
      dimension.SetByteStride(stride);
      stride *= dimension.Extent();
    }
  }
  return 0;
}

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
