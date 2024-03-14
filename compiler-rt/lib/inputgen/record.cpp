#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sys/resource.h>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

static void *(*RealMalloc)(size_t) = nullptr;
static void (*RealFree)(void *) = nullptr;

template <class T> struct IRAllocator {
  typedef T value_type;

  IRAllocator() = default;

  template <class U> constexpr IRAllocator(const IRAllocator<U> &) noexcept {}

  T *allocate(size_t Size) {
    T *Ptr = static_cast<T *>(RealMalloc(Size * sizeof(T)));
    if (!Ptr)
      abort();
    // INPUTGEN_DEBUG(std::cerr << "[IRAllocator] Allocated " << toVoidPtr(Ptr)
    //                          << " Size " << Size << "x" << sizeof(T)
    //                          << std::endl);
    return Ptr;
  }

  void deallocate(T *Ptr, size_t Size) noexcept {
    // INPUTGEN_DEBUG(std::cerr << "[IRAllocator] Freeing " << toVoidPtr(Ptr)
    //                          << " Size " << Size << "x" << sizeof(T)
    //                          << std::endl);
    RealFree(Ptr);
  }
};
template <class T, class U>
bool operator==(const IRAllocator<T> &, const IRAllocator<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const IRAllocator<T> &, const IRAllocator<U> &) {
  return false;
}

template <typename T> using IRVector = std::vector<T, IRAllocator<T>>;
using IRString =
    std::basic_string<char, std::char_traits<char>, IRAllocator<char>>;

#include "common.hpp"

// For some reason we get a template error if we try to template the IRVector
// type. That's why the implementation is in a different file and we define it
// before the include.
#include "dump-input-impl.hpp"

using BranchHint = llvm::inputgen::BranchHint;

template <typename T, typename... _Args>
std::unique_ptr<T> IRMakeUnique(_Args &&...Args) {
  IRAllocator<T> A;
  std::unique_ptr<T> UP(A.allocate(1));
  new (UP.get()) T(std::forward<_Args>(Args)...);
  return UP;
}

struct InputRecordConfTy {
  std::optional<IRString> InputOutName;
  bool QuitAfterRecord;
  bool ReportObjectAddressing;
  InputRecordConfTy() {
    ReportObjectAddressing = getenv("INPUT_RECORD_REPORT_OBJECT_ADDRESSING");
    QuitAfterRecord = getenv("INPUT_RECORD_QUIT_AFTER_RECORD");
    if (char *Str = getenv("INPUT_RECORD_FILENAME"))
      InputOutName = Str;
    else
      InputOutName = {};
  }
};

std::unique_ptr<IRObjectTy>
InputRecordPageObjectAddressing::makeUniqueObject(VoidPtrTy BasePtr,
                                                  size_t Size) {
  return IRMakeUnique<IRObjectTy>(BasePtr);
}

struct InputRecordRTTy {
  INPUTGEN_TIMER_DEFINE(InputRecordRecord);
  INPUTGEN_TIMER_DEFINE(InputRecordDump);

  AllocatorTy ObjectAllocator;
  InputRecordRTTy(InputRecordConfTy Conf) : Conf(Conf) {
    OutputObjIdxOffset = 0;
    ObjectAllocator.Malloc = RealMalloc;
    ObjectAllocator.Free = RealFree;
    OA.reset(new InputRecordPageObjectAddressing(ObjectAllocator));
  }
  ~InputRecordRTTy() {}

  InputRecordConfTy Conf;

  VoidPtrTy StackPtr;
  intptr_t OutputObjIdxOffset;
  IRString FuncIdent;
  IRString OutputDir;
  std::filesystem::path ExecPath;
  std::mt19937 Gen;

  struct GlobalTy {
    VoidPtrTy Ptr;
    size_t ObjIdx;
    uintptr_t Size;
  };
  IRVector<GlobalTy> Globals;
  IRVector<intptr_t> FunctionPtrs;

  uint64_t NumNewValues = 0;

  IRVector<GenValTy> GenVals;
  uint32_t NumArgs = 0;

#if defined(IR_MALLOC_TRACKING)
  // Storage for dynamic objects
  IRVector<std::unique_ptr<IRObjectTy>> Objects;
  IRVector<size_t> GlobalBundleObjects;
#else
  std::unique_ptr<InputRecordPageObjectAddressing> OA;
#endif

  IRVector<IRObjectTy *> getObjects() {
    IRVector<IRObjectTy *> Objects;
    OA->getObjects(Objects);
    return Objects;
  }

  std::optional<std::pair<IRObjectTy *, OffsetTy>>
  globalPtrToObjAndLocalPtr(VoidPtrTy GlobalPtr) {
#if defined(IR_MALLOC_TRACKING)
    // FIXME do we need to walk backwards so that we get the _last_ (currently
    // active) allocation?
    for (auto &Obj : Objects)
      if (Obj->isGlobalPtrInObject(GlobalPtr))
        return std::make_pair(&*Obj, Obj->getLocalPtr(GlobalPtr));
    return {};
#else
    abort();
#endif
  }
  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, OffsetTy PtrInObj) const {
#if defined(IR_MALLOC_TRACKING)
    return Objects[ObjIdx]->getGlobalPtr(PtrInObj);
#else
    abort();
#endif
  }

  template <typename T> T generateNewArg(BranchHint *BHs, int32_t BHSize) {
    abort();
  }

  template <typename T> void recordArg(T Val) {
    if constexpr (!std::is_same<T, __int128>::value) {
      if constexpr (std::is_pointer<T>::value)
        INPUTGEN_DEBUG(std::cerr << "Recorded arg " << toVoidPtr(Val)
                                 << std::endl);
      else
        INPUTGEN_DEBUG(std::cerr << "Recorded arg " << Val << std::endl);
    }
    GenVals.push_back(toGenValTy(Val, std::is_pointer<T>::value));
    NumArgs++;
  }

  template <typename T>
  T generateNewStubReturn(BranchHint *BHs, int32_t BHSize) {
    abort();
  }

  template <typename T> T getDefaultNewValue() { abort(); }

  template <typename T> T getNewValue(BranchHint *BHs, int32_t BHSize) {
    abort();
  }

  template <>
  VoidPtrTy getNewValue<VoidPtrTy>(BranchHint *BHs, int32_t BHSize) {
    abort();
  }

  template <>
  FunctionPtrTy getNewValue<FunctionPtrTy>(BranchHint *BHs, int32_t BHSize) {
    abort();
  }

  void memcpy(VoidPtrTy Tgt, VoidPtrTy Src, uint64_t N) {
    if (!Recording)
      return;
    OA->readArray(Src, N);
    OA->writeArray(Tgt, N);
  }

  void memset(VoidPtrTy Tgt, char C, uint64_t N) {
    if (!Recording)
      return;
    OA->writeArray(Tgt, N);
  }

  template <typename T> void write(VoidPtrTy Ptr, T Val, uint32_t Size) {
    if (!Recording)
      return;
#if defined(IR_MALLOC_TRACKING)
    assert(Ptr);
    auto Res = globalPtrToObjAndLocalPtr(Ptr);
    // FIXME need globals and stack handling
    assert(Res);
    auto [Obj, LocalPtr] = *Res;
    INPUTGEN_DEBUG(std::cerr << "Write to obj #" << Obj << " with size " << Size
                             << "\n");
    if (Obj)
      Obj->write<T>(Val, LocalPtr, Size);
#else
    OA->write<T>(Ptr, Val, Size);
#endif
  }

  template <typename T>
  void read(VoidPtrTy Ptr, VoidPtrTy Base, uint32_t Size, BranchHint *BHs,
            int32_t BHSize) {
    if (!Recording)
      return;
#if defined(IR_MALLOC_TRACKING)
    assert(Ptr);
    auto Res = globalPtrToObjAndLocalPtr(Ptr);
    // FIXME need globals and stack handling
    assert(Res);
    auto [Obj, LocalPtr] = *Res;
    INPUTGEN_DEBUG(std::cerr << "read from obj #" << Obj << " with size "
                             << Size << "\n");
    if (Obj)
      Obj->read<T>(LocalPtr, Size, BHs, BHSize);
#else
    OA->read<T>(Ptr, Base, Size);
#endif
  }

  // TODO need to think what happens when we free some memory and subsequently
  // the _same_ location is allocated (with different size for example) Also do
  // we need a hijacked fake `free` function in the replay so that we dont crash
  // when trying to free a non-freeable object?
  bool atFree(VoidPtrTy Ptr) {
#if defined(IR_MALLOC_TRACKING)
    if (Recording) {
      INPUTGEN_DEBUG(std::cerr << "Free " << toVoidPtr(Ptr)
                               << " ignored because currently recording"
                               << std::endl);
      return false;
    } else {
      INPUTGEN_DEBUG(std::cerr << "Free " << toVoidPtr(Ptr) << std::endl);
      return true;
    }
#else
    return true;
#endif
  }

  void cmpPtr(VoidPtrTy A, VoidPtrTy B, int32_t Predicate) {}

  // TODO if we dont track mallocs during recording, we may get a different
  // output state when replaying because a malloc may reuse part of an ObjectTy
  // and that memory usage may differ between recording and replaying.
  void atMalloc(VoidPtrTy Ptr, size_t Size) {
#if defined(IR_MALLOC_TRACKING)
    INPUTGEN_DEBUG(std::cerr << "Malloc " << toVoidPtr(Ptr) << " Size "
                             << Size);
    if (Recording) {
      INPUTGEN_DEBUG(std::cerr << " ignored because currently recording"
                               << std::endl);
    } else {
      size_t Idx = Objects.size();
      INPUTGEN_DEBUG(std::cerr << " -> Obj #" << Idx << std::endl);
      Objects.push_back(
          IRMakeUnique<IRObjectTy>(ObjectAllocator, Ptr, Size, 0, true));
    }
#endif
  }

  void registerGlobal(VoidPtrTy, VoidPtrTy *ReplGlobal, int32_t GlobalSize) {
    std::cerr << "register global not implemented yet\n";
    abort();
  }

  void registerFunctionPtrAccess(VoidPtrTy Ptr, uint32_t Size,
                                 VoidPtrTy *PotentialFPs, uint64_t N) {
    abort();
  }

  intptr_t registerFunctionPtrIdx(size_t N) { abort(); }

  void report() {
    if (Conf.InputOutName) {
      std::ofstream InputOutStream(Conf.InputOutName->c_str(),
                                   std::ios::out | std::ios::binary);
      INPUTGEN_TIMER_START(InputRecordDump);
      dumpInput<InputRecordRTTy, InputMode_Record_v1>(InputOutStream, *this);
      INPUTGEN_TIMER_END(InputRecordDump);
      std::cerr << "Dumped input to `" << *Conf.InputOutName << "`"
                << std::endl;
    } else {
      std::cerr << "No file specified, not dumping input." << std::endl;
    }
    if (Conf.ReportObjectAddressing) {
      OA->report();
    }
  }

  bool Recording = false;
  bool Done = false;
  void recordPush() {
    if (Done)
      return;
    if (Recording) {
      std::cerr << "Nested recording! Abort!" << std::endl;
      abort();
    }
    INPUTGEN_DEBUG(std::cerr << "Start recording\n");
    INPUTGEN_TIMER_START(InputRecordRecord);
    Recording = true;
  }
  void recordPop() {
    if (Done)
      return;
    if (!Recording) {
      std::cerr << "Pop without push? Abort!" << std::endl;
      abort();
    }
    INPUTGEN_TIMER_END(InputRecordRecord);
    std::cerr << "Completed recording" << std::endl;
    INPUTGEN_DEBUG(std::cerr << "Stop recording\n");
    Recording = false;
    report();
    Done = true;
    if (Conf.QuitAfterRecord)
      exit(14);
  }
};

static struct InputRecordRTInit {
  bool Initialized = false;
  std::unique_ptr<InputRecordRTTy> IRRT = nullptr;
  InputRecordRTInit() {
    IRAllocator<InputRecordRTTy> A;
    IRRT.reset(A.allocate(1));
    new (IRRT.get()) InputRecordRTTy(InputRecordConfTy());
    Initialized = true;
  }
} InputRecordRT;
static InputRecordRTTy &getInputRecordRT() { return *InputRecordRT.IRRT; }
static bool &isRTInitialized() { return InputRecordRT.Initialized; }

void *malloc(size_t Size) {
  static void *(*LRealMalloc)(size_t) = []() {
    RealMalloc =
        reinterpret_cast<decltype(RealMalloc)>(dlsym(RTLD_NEXT, "malloc"));
    return RealMalloc;
  }();
  void *Mem = LRealMalloc(Size);
  if (isRTInitialized())
    getInputRecordRT().atMalloc(reinterpret_cast<VoidPtrTy>(Mem), Size);
  return Mem;
}

void free(void *Ptr) {
  static void (*LRealFree)(void *) = []() {
    RealFree = reinterpret_cast<decltype(RealFree)>(dlsym(RTLD_NEXT, "free"));
    return RealFree;
  }();
  if (isRTInitialized()) {
    if (getInputRecordRT().atFree(reinterpret_cast<VoidPtrTy>(Ptr)))
      LRealFree(Ptr);
  } else {
    LRealFree(Ptr);
  }
}

// We need to run this before all other code that may use malloc or free, so
// priority is set to 101. 0-100 are reserved apparently. Even with 101 priority
// we get some malloc before we can get the RealMalloc which is why there is
// code for that in malloc() as well.
__attribute__((constructor(101))) static void hijackMallocAndFree() {
  RealMalloc =
      reinterpret_cast<decltype(RealMalloc)>(dlsym(RTLD_NEXT, "malloc"));
  RealFree = reinterpret_cast<decltype(RealFree)>(dlsym(RTLD_NEXT, "free"));
}

extern "C" {
void __record_push() { getInputRecordRT().recordPush(); }
void __record_pop() { getInputRecordRT().recordPop(); }
void __record_unreachable(int32_t No, const char *Name) {
  printf("Reached unreachable %i due to '%s'\n", No, Name ? Name : "n/a");
  exit(UnreachableExitStatus);
}
}

#define __IG_OBJ__ getInputRecordRT()
#include "common-interface.def"
extern "C" {
DEFINE_INTERFACE(record)
}
