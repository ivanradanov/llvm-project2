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

#include "rt-common.hpp"

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

// For some reason we get a template error if we try to template the IRVector
// type. That's why the implementation is in a different file and we define it
// before the include.
#include "rt-dump-input.hpp"

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
  InputRecordConfTy() {
    if (char *Str = getenv("INPUT_RECORD_FILENAME"))
      InputOutName = Str;
    else
      // FIXME
      InputOutName = "/dev/null";
  }
};

// struct InputRecordPageObjectAddressing : public ObjectAddressing {
struct InputRecordPageObjectAddressing {

  ObjectTy::ObjectAllocatorTy &ObjectAllocator;

  template <unsigned NumBits, template <unsigned TopBit> typename ChildTy,
            unsigned TopBit>
  struct Node {
    template <unsigned X> struct Shift {
      // FIXME we depend on 1 << 64 == 0 is it UB?
      static const uintptr_t Value = ((uintptr_t)1) << X;
    };
    static constexpr uintptr_t NumChildren = Shift<NumBits>::Value;
    static constexpr unsigned NextBit = TopBit - NumBits;
    static constexpr unsigned RemainingBits = sizeof(uintptr_t) * 8 - NumBits;
    using SpecializedChildTy = ChildTy<NextBit>;

    static constexpr uintptr_t extractMaskedPart(VoidPtrTy Ptr) {
      // If NumBIts = 3
      // Mask = 0000111
      uintptr_t Mask = (static_cast<uintptr_t>(1) << NumBits) - 1;
      uintptr_t ShiftedPtr =
          reinterpret_cast<uintptr_t>(Ptr) >> (TopBit - NumBits);
      uintptr_t Masked = ShiftedPtr & Mask;
      return Masked;
    }
    template <typename T>
    static std::pair<size_t, T *> allocateNew(IRVector<uint8_t> &NodeStorage) {
      unsigned AllocSize = sizeof(T);
      NodeStorage.reserve(NodeStorage.size() + AllocSize);
      NodeStorage.insert(NodeStorage.end(), AllocSize, 0);
      size_t Idx = NodeStorage.size() - AllocSize;
      VoidPtrTy AllocPtr = NodeStorage.data() + Idx;
      return {Idx, reinterpret_cast<T *>(AllocPtr)};
    }
  };

  template <unsigned TopBit> struct Leaf {
    using Node = Node<TopBit, Leaf, TopBit>;

    static_assert(Node::NextBit == 0);

    std::unique_ptr<ObjectTy> Object;

    Leaf(ObjectTy::ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Created Leaf Node for Ptr" << toVoidPtr(Ptr)
                               << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << "\n");
      VoidPtrTy BasePtr = reinterpret_cast<VoidPtrTy>(
          reinterpret_cast<uintptr_t>(Ptr) & ~(Node::NumChildren - 1));
      assert(reinterpret_cast<VoidPtrTy>(reinterpret_cast<uintptr_t>(Ptr) &
                                         ~Masked) == BasePtr);
      Object = IRMakeUnique<ObjectTy>(ObjectAllocator, BasePtr,
                                      Node::NumChildren, 0, true, false);
    }

    ObjectTy *getObject(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
                        VoidPtrTy Ptr, IRVector<uint8_t> &NodeStorage) {
      assert(Object);
      return Object.get();
    }
    void getObjects(IRVector<ObjectTy *> Objects,
                    IRVector<uint8_t> &NodeStorage) {
      assert(Object);
      Objects.push_back(Object.get());
    }
  };

  template <unsigned NumBits, template <unsigned TopBit> typename ChildTy,
            unsigned TopBit>
  struct ArrayNode {
    using Node = Node<NumBits, ChildTy, TopBit>;
    using SpecializedChildTy = ChildTy<Node::NextBit>;

    std::array<size_t, Node::NumChildren> Children;

    ArrayNode(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
              VoidPtrTy Ptr = nullptr) {
      for (auto &ChildIdx : Children)
        ChildIdx = 0;
      INPUTGEN_DEBUG(std::cerr << "Created Array Node for Ptr " << toBits(Ptr)
                               << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");
    }

    ObjectTy *getObject(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
                        VoidPtrTy Ptr, IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      SpecializedChildTy *ChildPtr = reinterpret_cast<SpecializedChildTy *>(
          &NodeStorage[Children[Masked]]);
      if (!ChildPtr) {
        size_t Idx;
        std::tie(Idx, ChildPtr) =
            Node::template allocateNew<SpecializedChildTy>(NodeStorage);
        new (ChildPtr) SpecializedChildTy(ObjectAllocator, Ptr);
        Children[Masked] = Idx;
      }
      return ChildPtr->getObject(ObjectAllocator, Ptr, NodeStorage);
    }
    void getObjects(IRVector<ObjectTy *> Objects,
                    IRVector<uint8_t> &NodeStorage) {
      for (auto &ChildIdx : Children) {
        if (ChildIdx)
          reinterpret_cast<SpecializedChildTy *>(
              &NodeStorage[Children[ChildIdx]])
              ->getObjects(Objects, NodeStorage);
      }
    }
  };

  template <unsigned NumBits, template <unsigned TopBit> typename ChildTy,
            unsigned TopBit>
  struct LinkedListNode {
    using Node = Node<NumBits, ChildTy, TopBit>;
    using SpecializedChildTy = ChildTy<Node::NextBit>;

    // Need to think about alignment
    struct LLNodeTy {
      uintptr_t PtrMatch : NumBits;
      // FIXME this is _not_ enough memory always
      uintptr_t NextIndex : Node::RemainingBits;
      unsigned : 0; // new byte
      // TODO this can be a bit field too if we index the NodeStorage to get
      // `Next` instead of storing a pointer
      uint8_t ChildData[sizeof(typename Node::SpecializedChildTy)];
    } Head;

    LinkedListNode(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
                   VoidPtrTy Ptr) {
      INPUTGEN_DEBUG(std::cerr << "Created Linked List Node for Ptr "
                               << toBits(Ptr) << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");
      constructNode(ObjectAllocator, Head, Ptr);
    }

    void constructNode(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
                       LLNodeTy &LLNode, VoidPtrTy Ptr) {
      assert(Ptr != nullptr);
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      LLNode.PtrMatch = Masked;
      LLNode.NextIndex = 0;
      new (LLNode.ChildData) SpecializedChildTy(ObjectAllocator, Ptr);
    }

    ObjectTy *getObject(ObjectTy::ObjectAllocatorTy &ObjectAllocator,
                        VoidPtrTy Ptr, IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      SpecializedChildTy *ChildPtr;
      LLNodeTy *LLNode = &Head;
      while (LLNode->PtrMatch != Masked) {
        if (LLNode->NextIndex != 0) {
          LLNode =
              reinterpret_cast<LLNodeTy *>(&NodeStorage[LLNode->NextIndex]);
        } else {
          auto [Idx, NewLLNode] =
              Node::template allocateNew<LLNodeTy>(NodeStorage);
          constructNode(ObjectAllocator, *NewLLNode, Ptr);
          LLNode = NewLLNode;
          break;
        }
      }
      ChildPtr = reinterpret_cast<typename Node::SpecializedChildTy *>(
          LLNode->ChildData);
      return ChildPtr->getObject(ObjectAllocator, Ptr, NodeStorage);
    }
    void getObjects(IRVector<ObjectTy *> Objects,
                    IRVector<uint8_t> &NodeStorage) {
      LLNodeTy *LLNode = &Head;
      do {
        auto ChildPtr = reinterpret_cast<typename Node::SpecializedChildTy *>(
            LLNode->ChildData);
        ChildPtr->getObjects(Objects, NodeStorage);
      } while (LLNode->NextIndex == 0);
    }
  };

  template <unsigned NumBits, template <unsigned> typename ChildTy>
  struct AArrayNode {
    template <unsigned TopBit>
    struct SNode : ArrayNode<NumBits, ChildTy, TopBit> {
      using ArrayNode<NumBits, ChildTy, TopBit>::ArrayNode;
    };
  };
  template <unsigned NumBits, template <unsigned> typename ChildTy>
  struct ALinkedListNode {
    template <unsigned TopBit>
    struct SNode : LinkedListNode<NumBits, ChildTy, TopBit> {
      using LinkedListNode<NumBits, ChildTy, TopBit>::LinkedListNode;
    };
  };
  // clang-format off
  ArrayNode<0,
  ALinkedListNode<52,
  Leaf
  >::SNode
  , sizeof(uintptr_t) * 8> Tree;
  IRVector<uint8_t> NodeStorage;
  ObjectTy *getObject(VoidPtrTy Ptr) {
    return Tree.getObject(ObjectAllocator, Ptr, NodeStorage);
  }
  void getObjects(IRVector<ObjectTy *> &Objects) {
    Tree.getObjects(Objects, NodeStorage);
  }
  // clang-format on
  InputRecordPageObjectAddressing(ObjectTy::ObjectAllocatorTy &ObjectAllocator)
      : ObjectAllocator(ObjectAllocator), Tree(ObjectAllocator) {}
};

struct InputRecordRTTy {

  ObjectTy::ObjectAllocatorTy ObjectAllocator;
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
    dumpInput<InputRecordRTTy, InputMode_Record>(InputOutStream, *this);
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
    INPUTGEN_DEBUG(std::cout << "Start recording\n");
    Recording = true;
  }
  void recordPop() {
    if (Done)
      return;
    if (!Recording) {
      std::cerr << "Pop without push? Abort!" << std::endl;
      abort();
    }
    INPUTGEN_DEBUG(std::cout << "Stop recording\n");
    Recording = false;
    report();
    Done = true;
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
}

#define __IG_OBJ__ getInputRecordRT()
#include "rt-common-interface.def"
extern "C" {
DEFINE_INTERFACE(record)
}
