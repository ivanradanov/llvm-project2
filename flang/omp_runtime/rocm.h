#ifndef FLANG_OMP_RUNTIME_ROCM_H_
#define FLANG_OMP_RUNTIME_ROCM_H_

#include <cstdio>

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#define CHECK_HIP(X) \
  do { \
    hipError_t rstatus = X; \
    if (rstatus != hipSuccess) { \
      fprintf(stderr, "hip error: %s (%d) %s %d", hipGetErrorString(rstatus), \
          rstatus, __FILE__, __LINE__); \
      abort(); \
    } \
  } while (0)

#define CHECK_ROCBLAS(X) \
  do { \
    rocblas_status rstatus = X; \
    if (rstatus != rocblas_status_success) { \
      fprintf(stderr, "rocblas error: %s (%d) %s %d", \
          rocblas_status_to_string(rstatus), rstatus, __FILE__, __LINE__); \
      abort(); \
    } \
  } while (0)

namespace Fortran::runtime::rocm {
struct RocmContextTy {
  rocblas_handle handle;
  RocmContextTy();
  ~RocmContextTy();
};
RocmContextTy &getRocmContext();
} // namespace Fortran::runtime::rocm

#endif // ROCM_H_
