#include "rocm.h"
#include "utils.h"
#include "../runtime/freestanding-tools.h"
#include "../runtime/terminator.h"
#include "../runtime/tools.h"
#include "../runtime/type-info.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"

#include <omp.h>

namespace Fortran::runtime {
namespace omp {

// Implements an instance of MATMUL for given argument types.
template <bool IS_ALLOCATING, TypeCategory RCAT, int RKIND, typename XT,
    typename YT>
static inline RT_API_ATTRS void DoMatmul(
    std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor> &result,
    const Descriptor &x, const Descriptor &y, Terminator &terminator,
    omp::OMPDeviceTy ompDevice) {
  int xRank{x.rank()};
  int yRank{y.rank()};
  int resRank{xRank + yRank - 2};
  if (xRank * yRank != 2 * resRank) {
    terminator.Crash("MATMUL: bad argument ranks (%d * %d)", xRank, yRank);
  }
  SubscriptValue extent[2]{
      xRank == 2 ? x.GetDimension(0).Extent() : y.GetDimension(1).Extent(),
      resRank == 2 ? y.GetDimension(1).Extent() : 0};
  if constexpr (IS_ALLOCATING) {
    // terminator.Crash("MATMUL: unsupported allocating matmul");
    // TODO need to allocate on device
    result.Establish(
        RCAT, RKIND, nullptr, resRank, extent, CFI_attribute_allocatable);
    for (int j{0}; j < resRank; ++j) {
      result.GetDimension(j).SetBounds(1, extent[j]);
    }
    if (int stat{result.AllocateTarget(ompDevice)}) {
      terminator.Crash(
          "MATMUL: could not allocate memory for result; STAT=%d", stat);
    }
  } else {
    RUNTIME_CHECK(terminator, resRank == result.rank());
    RUNTIME_CHECK(
        terminator, result.ElementBytes() == static_cast<std::size_t>(RKIND));
    RUNTIME_CHECK(terminator, result.GetDimension(0).Extent() == extent[0]);
    RUNTIME_CHECK(terminator,
        resRank == 1 || result.GetDimension(1).Extent() == extent[1]);
  }
  SubscriptValue n{x.GetDimension(xRank - 1).Extent()};
  if (n != y.GetDimension(0).Extent()) {
    terminator.Crash("MATMUL: unacceptable operand shapes (%jdx%jd, %jdx%jd)",
        static_cast<std::intmax_t>(x.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(n),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(y.GetDimension(1).Extent()));
  }
  using WriteResult =
      CppTypeFor<RCAT == TypeCategory::Logical ? TypeCategory::Integer : RCAT,
          RKIND>;
  if constexpr (RCAT != TypeCategory::Logical) {
    if (x.IsContiguous(1) && y.IsContiguous(1) &&
        (IS_ALLOCATING || result.IsContiguous())) {
      // Contiguous numeric matrices (maybe with columns
      // separated by a stride).
      std::optional<std::size_t> xColumnByteStride;
      if (!x.IsContiguous()) {
        // X's columns are strided.
        SubscriptValue xAt[2]{};
        x.GetLowerBounds(xAt);
        xAt[1]++;
        xColumnByteStride = x.SubscriptsToByteOffset(xAt);
      }
      std::optional<std::size_t> yColumnByteStride;
      if (!y.IsContiguous()) {
        // Y's columns are strided.
        SubscriptValue yAt[2]{};
        y.GetLowerBounds(yAt);
        yAt[1]++;
        yColumnByteStride = y.SubscriptsToByteOffset(yAt);
      }
      // Note that BLAS GEMM can be used for the strided
      // columns by setting proper leading dimension size.
      // This implies that the column stride is divisible
      // by the element size, which is usually true.
      if (resRank == 2) { // M*M -> M
        if (std::is_same_v<XT, YT>) {
          const rocblas_operation transA = rocblas_operation_none;
          const rocblas_operation transB = rocblas_operation_none;
          // enable passing alpha parameter from pointer to host memory
          auto ptrX = getDevicePtr(x.OffsetElement<XT>(), ompDevice);
          auto ptrY = getDevicePtr(y.OffsetElement<XT>(), ompDevice);
          auto ptrRes =
              getDevicePtr(result.template OffsetElement<XT>(), ompDevice);
          if constexpr (std::is_same_v<XT, float>) {
            float hAlpha = 1;
            float hBeta = 1;
            CHECK_ROCBLAS(rocblas_sgemm(rocm::getRocmContext().handle, transA,
                transB, extent[0], n, extent[1], &hAlpha, ptrX, extent[0], ptrY,
                extent[1], &hBeta, ptrRes, extent[0]));
          } else if constexpr (std::is_same_v<XT, double>) {
            double hAlpha = 1;
            double hBeta = 1;
            CHECK_ROCBLAS(rocblas_dgemm(rocm::getRocmContext().handle, transA,
                transB, extent[0], n, extent[1], &hAlpha, ptrX, extent[0], ptrY,
                extent[1], &hBeta, ptrRes, extent[0]));
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            terminator.Crash("MATMUL: unsupported matmul M*M %s", __func__);
            // TODO: call BLAS-3 CGEMM
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            terminator.Crash("MATMUL: unsupported matmul M*M %s", __func__);
            // TODO: call BLAS-3 ZGEMM
          }
          // TODO we would like to synchronize with finer granularity
          CHECK_HIP(hipDeviceSynchronize());
          return;
        }
        terminator.Crash("MATMUL: unsupported matmul M*M %s", __func__);
        return;
      } else if (xRank == 2) { // M*V -> V
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-2 SGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-2 DGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-2 CGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-2 ZGEMV(x,y)
          }
        }
        terminator.Crash("MATMUL: unsupported matmul M*V");
        return;
      } else { // V*M -> V
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-2 SGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-2 DGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-2 CGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-2 ZGEMV(y,x)
          }
        }
        terminator.Crash("MATMUL: unsupported matmul V*M");
        return;
      }
    }
  }
  terminator.Crash("MATMUL: unsupported matmul");
}

// Maps the dynamic type information from the arguments' descriptors
// to the right instantiation of DoMatmul() for valid combinations of
// types.
template <bool IS_ALLOCATING> struct Matmul {
  using ResultDescriptor =
      std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor>;
  template <TypeCategory XCAT, int XKIND> struct MM1 {
    template <TypeCategory YCAT, int YKIND> struct MM2 {
      RT_API_ATTRS void operator()(ResultDescriptor &result,
          const Descriptor &x, const Descriptor &y, Terminator &terminator,
          omp::OMPDeviceTy ompDevice) const {
        if constexpr (constexpr auto resultType{
                          GetResultType(XCAT, XKIND, YCAT, YKIND)}) {
          if constexpr (common::IsNumericTypeCategory(resultType->first) ||
              resultType->first == TypeCategory::Logical) {
            return DoMatmul<IS_ALLOCATING, resultType->first,
                resultType->second, CppTypeFor<XCAT, XKIND>,
                CppTypeFor<YCAT, YKIND>>(result, x, y, terminator, ompDevice);
          }
        }
        terminator.Crash("MATMUL: bad operand types (%d(%d), %d(%d))",
            static_cast<int>(XCAT), XKIND, static_cast<int>(YCAT), YKIND);
      }
    };
    RT_API_ATTRS void operator()(ResultDescriptor &result, const Descriptor &x,
        const Descriptor &y, Terminator &terminator, TypeCategory yCat,
        int yKind, omp::OMPDeviceTy ompDevice) const {
      ApplyType<MM2, void>(
          yCat, yKind, terminator, result, x, y, terminator, ompDevice);
    }
  };
  RT_API_ATTRS void operator()(ResultDescriptor &result, const Descriptor &x,
      const Descriptor &y, const char *sourceFile, int line,
      omp::OMPDeviceTy ompDevice) const {
    Terminator terminator{sourceFile, line};
    auto xCatKind{x.type().GetCategoryAndKind()};
    auto yCatKind{y.type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator, xCatKind.has_value() && yCatKind.has_value());
    ApplyType<MM1, void>(xCatKind->first, xCatKind->second, terminator, result,
        x, y, terminator, yCatKind->first, yCatKind->second, ompDevice);
  }
};

// RT_API_ATTRS static void Matmul(Descriptor &c, const Descriptor &a,
//     const Descriptor &b, Terminator &terminator, const char *sourceFile,
//     int sourceLine, omp::OMPDeviceTy omp_device) {
//   std::size_t toElementBytes{a.ElementBytes()};
//   std::size_t fromElementBytes{b.ElementBytes()};
//   std::size_t toElements{a.Elements()};
//   std::size_t fromElements{b.Elements()};

//   if (toElementBytes != fromElementBytes)
//     terminator.Crash("Assign: toElementBytes != fromElementBytes");
//   if (toElements != fromElements)
//     terminator.Crash("Assign: toElements != fromElements");

//   void *to_ptr = getDevicePtr(to.raw().base_addr, omp_device);
//   void *from_ptr = getDevicePtr(from.raw().base_addr, omp_device);
//   size_t length = toElements * toElementBytes;

//   omp_target_memcpy(to_ptr, from_ptr, length, /*dst_offset*/ 0,
//       /*src_offset*/ 0, /*dst*/ omp_device, /*src*/ omp_device);
//   return;
// }

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(Matmul_omp)(Descriptor &c, const Descriptor &a, const Descriptor &b,
    const char *sourceFile, int sourceLine, omp::OMPDeviceTy ompDevice) {
  omp::Matmul<true>{}(c, a, b, sourceFile, sourceLine, ompDevice);
}
void RTDEF(MatmulDirect_omp)(Descriptor &c, const Descriptor &a,
    const Descriptor &b, const char *sourceFile, int sourceLine,
    omp::OMPDeviceTy ompDevice) {
  omp::Matmul<false>{}(c, a, b, sourceFile, sourceLine, ompDevice);
}

} // extern "C"
} // namespace omp
} // namespace Fortran::runtime
