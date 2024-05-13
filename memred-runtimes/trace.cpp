
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>

#include <cuda_runtime.h>

#define CHECK_ERR(ans)                                                         \
  { checkErr((ans), __FILE__, __LINE__); }
static void checkErr(cudaError_t Err, const char *File, int Line) {
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s\n", File, Line, cudaGetErrorString(Err));
    // abort();
  }
}

typedef uintptr_t *VoidPtrTy;

struct ObjectAddressing {
  size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const {
    size_t Idx =
        (reinterpret_cast<intptr_t>(GlobalPtr) & ObjIdxMask) / MaxObjectSize;
    return Idx;
  }

  VoidPtrTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const {
    return reinterpret_cast<VoidPtrTy>(reinterpret_cast<intptr_t>(GlobalPtr) &
                                       PtrInObjMask);
  }

  VoidPtrTy getObjBasePtr() const {
    return reinterpret_cast<VoidPtrTy>(MaxObjectSize / 2);
  }

  intptr_t getOffsetFromObjBasePtr(VoidPtrTy Ptr) const {
    return Ptr - getObjBasePtr();
  }

  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, VoidPtrTy PtrInObj) const {
    return reinterpret_cast<VoidPtrTy>((ObjIdx * MaxObjectSize) |
                                       reinterpret_cast<intptr_t>(PtrInObj));
  }

  intptr_t PtrInObjMask;
  intptr_t ObjIdxMask;
  uintptr_t MaxObjectSize;
  uintptr_t MaxObjectNum;

  uintptr_t Size;

  unsigned int highestOne(uint64_t X) { return 63 ^ __builtin_clzll(X); }

  void setSize(uintptr_t Size) {
    this->Size = Size;

    uintptr_t HO = highestOne(Size | 1) + 1;
    uintptr_t BitsForObj = HO;
    MaxObjectSize = 1ULL << BitsForObj;
    MaxObjectNum = 1ULL << (HO - BitsForObj);
    PtrInObjMask = MaxObjectSize - 1;
    ObjIdxMask = ~(PtrInObjMask);
  }
};

static struct EventsTy {
  class EventTy {
  public:
    size_t Idx = -1;
    virtual ~EventTy() = default;
  };
  class AllocationTy : public EventTy {
  public:
    void *RealPtr = nullptr;
    void *VirtualPtr = nullptr;
    size_t Size = 0;
  };
  std::list<std::unique_ptr<EventTy>> Events;

  AllocationTy *insertNewAllocation(void *RealPtr, size_t Size) {
    size_t Idx = Events.size();
    auto A = std::make_unique<AllocationTy>();
    A->Idx = Idx;
    A->RealPtr = RealPtr;
    A->VirtualPtr = OA.localPtrToGlobalPtr(Idx, OA.getObjBasePtr());
    A->Size = Size;
    auto *Ret = A.get();
    Events.push_back(std::move(A));
    return Ret;
  }

  static constexpr uintptr_t MaxAllocationSize =
      1ULL * 160 /*GB*/ * 1024 * 1024 * 1024;

  EventsTy() { OA.setSize(MaxAllocationSize); }
  ~EventsTy() {
    auto &Log = std::cerr;
    Log << "Graph:\n";
    for (auto &Event : Events) {
      if (auto *A = dynamic_cast<EventsTy::AllocationTy *>(&*Event))
        Log << A->Idx << ": "
            << "Size " << A->Size << " Real " << A->RealPtr << " Virtual "
            << A->VirtualPtr << "\n";
    }
  }

  ObjectAddressing OA;
} Allocations;

#define MEMRED_ATTRS extern "C"

MEMRED_ATTRS cudaError_t __memred_cudaMalloc(void **p, size_t s) {
  void *Ptr;
  cudaError_t Err = cudaMalloc(&Ptr, s);
  CHECK_ERR(Err);
  auto A = Allocations.insertNewAllocation(Ptr, s);
  // TODO should return virtual ptr
  *p = Ptr;
  // *p = A.VirtualPtr;
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaLaunchKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  // TODO need to translate addresses here
  // abort();
  cudaError_t Err =
      cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  CHECK_ERR(Err);
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaMemcpyAsync(void *dst, const void *src,
                                                  size_t count,
                                                  enum cudaMemcpyKind kind,
                                                  cudaStream_t stream) {
  // TODO
  // abort();
  cudaError_t Err = cudaMemcpyAsync(dst, src, count, kind, stream);
  CHECK_ERR(Err);
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaMemcpy(void *dst, const void *src,
                                             size_t count,
                                             enum cudaMemcpyKind kind) {
  // TODO
  // abort();
  cudaError_t Err = cudaMemcpy(dst, src, count, kind);
  CHECK_ERR(Err);
  return Err;
}
