#ifndef FLANG_OMP_RUNTIME_UTILS_H_
#define FLANG_OMP_RUNTIME_UTILS_H_

#include <cstdint>
#include <cstdio>
#include <omp.h>

namespace Fortran::runtime::omp {

constexpr static bool FLANG_OMP_RUNTIME_DEBUG = false;

typedef int32_t OMPDeviceTy;

template <typename T> static T *getDevicePtr(T *anyPtr, OMPDeviceTy ompDevice) {
  auto voidAnyPtr = reinterpret_cast<void *>(anyPtr);
  if (FLANG_OMP_RUNTIME_DEBUG)
    fprintf(stderr, "getDevicePtr(%p, %d) = ", voidAnyPtr, ompDevice);
  if (!omp_target_is_present(voidAnyPtr, ompDevice)) {
    // If not present on the device it should already be a device ptr
    if (FLANG_OMP_RUNTIME_DEBUG)
      fprintf(stderr, "%p\n", voidAnyPtr);
    return anyPtr;
  }
  T *device_ptr = nullptr;
#pragma omp target data use_device_ptr(anyPtr) device(ompDevice)
  device_ptr = anyPtr;
  if (FLANG_OMP_RUNTIME_DEBUG)
    fprintf(stderr, "%p\n", reinterpret_cast<void *>(device_ptr));
  return device_ptr;
}
} // namespace Fortran::runtime::omp

#endif // FLANG_OMP_RUNTIME_UTILS_H_
