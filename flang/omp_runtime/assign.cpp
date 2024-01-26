#include "./utils.h"
#include "../runtime/freestanding-tools.h"
#include "../runtime/terminator.h"
#include "../runtime/type-info.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"

#include <omp.h>

namespace Fortran::runtime {

enum AssignFlags {
  NoAssignFlags = 0,
  MaybeReallocate = 1 << 0,
  NeedFinalization = 1 << 1,
  CanBeDefinedAssignment = 1 << 2,
  ComponentCanBeDefinedAssignment = 1 << 3,
  ExplicitLengthCharacterLHS = 1 << 4,
  PolymorphicLHS = 1 << 5,
  DeallocateLHS = 1 << 6
};

namespace omp {

RT_API_ATTRS static void Assign(Descriptor &to, const Descriptor &from,
    Terminator &terminator, int flags, OMPDeviceTy omp_device) {
  std::size_t toElementBytes{to.ElementBytes()};
  std::size_t fromElementBytes{from.ElementBytes()};
  std::size_t toElements{to.Elements()};
  std::size_t fromElements{from.Elements()};

  if (toElementBytes != fromElementBytes)
    terminator.Crash("Assign: toElementBytes != fromElementBytes");
  if (toElements != fromElements)
    terminator.Crash("Assign: toElements != fromElements");

  void *to_ptr = getDevicePtr(to.raw().base_addr, omp_device);
  void *from_ptr = getDevicePtr(from.raw().base_addr, omp_device);
  size_t length = toElements * toElementBytes;

  omp_target_memcpy(to_ptr, from_ptr, length, /*dst_offset*/ 0,
      /*src_offset*/ 0, /*dst*/ omp_device, /*src*/ omp_device);
  return;
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(Assign_omp)(Descriptor &to, const Descriptor &from,
    const char *sourceFile, int sourceLine, omp::OMPDeviceTy omp_device) {
  Terminator terminator{sourceFile, sourceLine};
  // All top-level defined assignments can be recognized in semantics and
  // will have been already been converted to calls, so don't check for
  // defined assignment apart from components.
  omp::Assign(to, from, terminator,
      MaybeReallocate | NeedFinalization | ComponentCanBeDefinedAssignment,
      omp_device);
}

} // extern "C"
} // namespace omp
} // namespace Fortran::runtime
