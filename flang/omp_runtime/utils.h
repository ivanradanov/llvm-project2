#ifndef FLANG_OMP_RUNTIME_UTILS_H_
#define FLANG_OMP_RUNTIME_UTILS_H_

#include <cstdint>
#include <cstdio>
#include <omp.h>

#ifdef FLANG_OMP_RUNTIME_TIME
#include <chrono>
#include <iostream>
#define TIMER_DEFINE(Name) \
  std::chrono::steady_clock::time_point Timer##Name##Begin;
#define TIMER_START(Name) \
  do { \
    if (true) \
      Timer##Name##Begin = std::chrono::steady_clock::now(); \
  } while (0)
#define TIMER_END(Name) \
  do { \
    if (true) { \
      std::chrono::steady_clock::time_point Timer##Name##End = \
          std::chrono::steady_clock::now(); \
      std::cout << "Time for " << #Name << ": " \
                << std::chrono::duration_cast<std::chrono::microseconds>( \
                       Timer##Name##End - Timer##Name##Begin) \
                       .count() \
                << std::endl; \
    } \
  } while (0)
#else
#define TIMER_DEFINE(Name)
#define TIMER_START(Name)
#define TIMER_END(Name)
#endif

namespace Fortran::runtime::omp {

static bool FLANG_OMP_RUNTIME_DEBUG = getenv("FLANG_OMP_RUNTIME_DEBUG");

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
