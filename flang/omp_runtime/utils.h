#ifndef FLANG_OMP_RUNTIME_UTILS_H_
#define FLANG_OMP_RUNTIME_UTILS_H_

#include <cstdint>
#include <cstdio>
#include <omp.h>

namespace Fortran::runtime::omp {

typedef int32_t OMPDeviceTy;

[[maybe_unused]] static void *getDevicePtr(
    void *any_ptr, OMPDeviceTy omp_device) {
  fprintf(stderr, "getDevicePtr(%p, %d) = ", any_ptr, omp_device);
  if (!omp_target_is_present(any_ptr, omp_device)) {
    // If not present on the device it should already be a device ptr
    fprintf(stderr, "%p\n", any_ptr);
    return any_ptr;
  }
  void *device_ptr = nullptr;
#pragma omp target data use_device_ptr(any_ptr) device(omp_device)
  device_ptr = any_ptr;
  fprintf(stderr, "%p\n", device_ptr);
  return device_ptr;
}
} // namespace Fortran::runtime::omp

#endif // FLANG_OMP_RUNTIME_UTILS_H_
