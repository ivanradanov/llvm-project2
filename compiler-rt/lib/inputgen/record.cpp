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
  IRString InputOutName;
  bool QuitAfterRecord;
  InputRecordConfTy() {
    QuitAfterRecord = getenv("INPUT_RECORD_QUIT_AFTER_RECORD");
    if (char *Str = getenv("INPUT_RECORD_FILENAME"))
      InputOutName = Str;
    else
      // FIXME
      InputOutName = "/dev/null";
  }
};

std::unique_ptr<ObjectTy> InputRecordPageObjectAddressing::makeUniqueObject(
    ObjectAllocatorTy &ObjectAllocator, VoidPtrTy BasePtr, size_t Size) {
  return IRMakeUnique<ObjectTy>(ObjectAllocator, BasePtr, Size, 0, true, false);
}

struct InputRecordRTTy {
  ObjectAllocatorTy ObjectAllocator;
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
  IRVector<std::unique_ptr<ObjectTy>> Objects;
  IRVector<size_t> GlobalBundleObjects;
#else
  std::unique_ptr<InputRecordPageObjectAddressing> OA;
#endif

  IRVector<ObjectTy *> getObjects() {
    IRVector<ObjectTy *> Objects;
    OA->getObjects(Objects);
    return Objects;
  }

  // Returns nullptr if it is not an object managed by us - a stack pointer or
  // memory allocated by malloc
  ObjectTy *globalPtrToObj(VoidPtrTy GlobalPtr, bool AllowNull = false) {
#if defined(IR_MALLOC_TRACKING)
    for (auto &Obj : Objects)
      if (Obj->isGlobalPtrInObject(GlobalPtr))
        return &*Obj;
    return nullptr;
#else
    return OA->getObject(GlobalPtr);
#endif
  }
  std::optional<std::pair<ObjectTy *, OffsetTy>>
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
    ObjectTy *Obj = globalPtrToObj(Ptr);
    INPUTGEN_DEBUG(std::cerr << "Write to obj #" << Obj << " with size " << Size
                             << "\n");
    Obj->writeToGlobalPtr<T>(Val, Ptr, Size);
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
    ObjectTy *Obj = globalPtrToObj(Ptr);
    INPUTGEN_DEBUG(std::cerr << "read from obj #" << Obj << " with size "
                             << Size << "\n");
    Obj->readFromGlobalPtr<T>(Ptr, Size, BHs, BHSize);
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
          IRMakeUnique<ObjectTy>(ObjectAllocator, Ptr, Size, 0, true));
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
    std::ofstream InputOutStream(Conf.InputOutName.c_str(),
                                 std::ios::out | std::ios::binary);
    dumpInput<InputRecordRTTy, InputMode_Record_v1>(InputOutStream, *this);
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
    Recording = true;
  }
  void recordPop() {
    if (Done)
      return;
    if (!Recording) {
      std::cerr << "Pop without push? Abort!" << std::endl;
      abort();
    }
    INPUTGEN_DEBUG(std::cerr << "Stop recording\n");
    Recording = false;
    report();
    Done = true;

    std::cerr << "Completed recording" << std::endl;
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

template <typename T>
T ObjectTy::read(OffsetTy Ptr, uint32_t Size, BranchHint *BHs, int32_t BHSize) {
  intptr_t Offset = getOffsetFromObjMemory(Ptr);
  assert(Output.isAllocated(Offset, Size));
  Used.ensureAllocation(Offset, Size);
  Input.ensureAllocation(Offset, Size);

  T *OutputLoc = reinterpret_cast<T *>(
      advance(Output.Memory, -Output.AllocationOffset + Offset));
  T Val = *OutputLoc;
  if (allUsed(Offset, Size))
    return Val;

  // FIXME redundant store - we use the function to mark the correct memory as
  // used, etc
  storeInputValue(Val, Offset, Size);

  if constexpr (std::is_pointer<T>::value)
    Ptrs.insert(Offset);
  return Val;
}

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
