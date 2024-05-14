
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_ERR(ans)                                                         \
  { checkErr((ans), __FILE__, __LINE__); }
static void checkErr(cudaError_t Err, const char *File, int Line) {
  if (Err != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s\n", File, Line, cudaGetErrorString(Err));
    abort();
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

#define DUMP(X)                                                                \
  do {                                                                         \
    Log << " " #X " " << X;                                                    \
  } while (0)

static struct EventsTy {
  class EventTy {
  public:
    virtual void dump(std::ostream &Log) { Log << "UNKNOWN"; }
    virtual ~EventTy() = default;
  };
  class KernelCallTy : public EventTy {
  public:
    const char *Name;
    cudaStream_t Stream;
    // TODO these should be Allocation IDs and not raw pointers
    std::vector<void *> PtrArgs;
    void dump(std::ostream &Log) override {
      Log << "Kernel call:";
      DUMP(Name);
      DUMP(Stream);
      size_t Idx = 0;
      for (void *PtrArg : PtrArgs) {
        DUMP(Idx);
        DUMP(PtrArg);
        Idx++;
      }
    }
  };
  class CopyTy : public EventTy {
  public:
    enum cudaMemcpyKind Kind = cudaMemcpyDefault;
    cudaStream_t Stream = 0;
    const void *From = nullptr;
    void *To = nullptr;
    size_t Size = 0;
    bool Async = false;
    void dump(std::ostream &Log) override {
      Log << "Copy:";
      DUMP(Kind);
      DUMP(Stream);
      DUMP(From);
      DUMP(To);
      DUMP(Size);
    }
  };
  class AllocationTy : public EventTy {
  public:
    size_t Idx = 0;
    void *RealPtr = nullptr;
    void *VirtualPtr = nullptr;
    size_t Size = 0;
    void dump(std::ostream &Log) override {
      Log << "Allocation:";
      DUMP(Idx);
      DUMP(RealPtr);
      DUMP(VirtualPtr);
      DUMP(Size);
    }
  };

  std::list<std::unique_ptr<EventTy>> Events;
  size_t NumAllocations = 0;

  KernelCallTy *insertNewKernelCall(const char *Name, void **Args,
                                    cudaStream_t Stream) {
    auto K = std::make_unique<KernelCallTy>();
    K->Name = Name;

    auto Kernel = std::find_if(Kernels.begin(), Kernels.end(),
                               [&](KernelTy &K) { return K.Name == Name; });
    if (Kernel == Kernels.end()) {
      std::cerr << "Could not find kernel " << Name << std::endl;
      abort();
    }

    for (auto ArgNo : Kernel->PtrArgs)
      K->PtrArgs.push_back(*reinterpret_cast<void **>(Args[ArgNo]));

    auto *Ret = K.get();
    Events.push_back(std::move(K));
    return Ret;
  }

  CopyTy *insertNewCopy(const void *From, void *To, size_t Size,
                        enum cudaMemcpyKind Kind, cudaStream_t Stream,
                        bool Async) {
    auto C = std::make_unique<CopyTy>();
    C->From = From;
    C->To = To;
    C->Size = Size;
    C->Kind = Kind;
    C->Stream = Stream;
    C->Async = Async;
    auto *Ret = C.get();
    Events.push_back(std::move(C));
    return Ret;
  }

  AllocationTy *insertNewAllocation(void *RealPtr, size_t Size) {
    size_t Idx = NumAllocations++;
    auto A = std::make_unique<AllocationTy>();
    A->RealPtr = RealPtr;
    A->VirtualPtr = OA.localPtrToGlobalPtr(Idx, OA.getObjBasePtr());
    A->Size = Size;
    A->Idx = Idx;
    auto *Ret = A.get();
    Events.push_back(std::move(A));
    return Ret;
  }

  static constexpr uintptr_t MaxAllocationSize =
      1ULL * 160 /*GB*/ * 1024 * 1024 * 1024;

  struct KernelTy {
    std::string Name;
    std::vector<size_t> PtrArgs;
  };
  std::vector<KernelTy> Kernels;

  EventsTy() {
    OA.setSize(MaxAllocationSize);

    // TODO getenv this
    char *KernelAnalysisFile = "./.memred.memory.analysis.out";

    std::ifstream In(KernelAnalysisFile);

    std::string Ignore;
    while (In) {
      KernelTy Kernel;
      char C;
      do {
        In.read(&C, 1);
      } while (In && C != '@');
      if (!In)
        break;
      std::string FuncName;
      In >> FuncName;
      FuncName.erase(FuncName.size() - 2);
      Kernel.Name = FuncName;

      // TODO we need to read the whoel function memory effects here
      In >> Ignore >> Ignore >> Ignore; // Memory Effect: ArgMemOnly

      while (1) {
        std::string ArgKw;
        In >> ArgKw;
        if (ArgKw == "Function" || !In)
          break;
        if (ArgKw != "Arg") {
          std::cerr << ArgKw;
          abort();
        }

        In.read(&C, 1); // ' '
        In.read(&C, 1); // '#'

        size_t ArgNo;
        In >> ArgNo;
        Kernel.PtrArgs.push_back(ArgNo);

        // TODO we need to read the memory effects on the specific arguments
        // here
        In >> Ignore >> Ignore >> Ignore >> Ignore >>
            Ignore; // :  Effect: WriteOnly Capture: No
      }
      Kernels.push_back(Kernel);
    }
  }

  ~EventsTy() {
    auto &Log = std::cerr;
    Log << "Graph:\n";
    size_t Idx = 0;
    for (auto &Event : Events) {
      Log << Idx << ": ";
      Event->dump(Log);
      Log << "\n";
      Idx++;
    }
  }

  ObjectAddressing OA;
} Events;

#define MEMRED_ATTRS extern "C"

MEMRED_ATTRS cudaError_t __memred_cudaMalloc(void **p, size_t s) {
  void *Ptr;
  cudaError_t Err = cudaMalloc(&Ptr, s);
  CHECK_ERR(Err);
  auto *A = Events.insertNewAllocation(Ptr, s);
  // TODO should return virtual ptr
  // *p = A.VirtualPtr;
  *p = Ptr;
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaLaunchKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  // TODO need to translate addresses here
  cudaError_t Err =
      cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  CHECK_ERR(Err);

  const char *Name;
  CHECK_ERR(cudaFuncGetName(&Name, func));
  Events.insertNewKernelCall(Name, args, stream);
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaMemcpyAsync(void *dst, const void *src,
                                                  size_t count,
                                                  enum cudaMemcpyKind kind,
                                                  cudaStream_t stream) {
  cudaError_t Err = cudaMemcpyAsync(dst, src, count, kind, stream);
  CHECK_ERR(Err);
  Events.insertNewCopy(src, dst, count, kind, stream, true);
  return Err;
}

MEMRED_ATTRS cudaError_t __memred_cudaMemcpy(void *dst, const void *src,
                                             size_t count,
                                             enum cudaMemcpyKind kind) {
  cudaError_t Err = cudaMemcpy(dst, src, count, kind);
  CHECK_ERR(Err);
  Events.insertNewCopy(src, dst, count, kind, 0, false);
  return Err;
}
