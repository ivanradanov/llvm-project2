#include "rocm.h"

namespace Fortran::runtime::rocm {

RocmContextTy::RocmContextTy() {
  CHECK_ROCBLAS(rocblas_create_handle(&handle));
  CHECK_ROCBLAS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
}

RocmContextTy::~RocmContextTy() {
  CHECK_ROCBLAS(rocblas_destroy_handle(handle));
}

RocmContextTy RocmContext;

RocmContextTy &getRocmContext() { return RocmContext; }

} // namespace Fortran::runtime::rocm
