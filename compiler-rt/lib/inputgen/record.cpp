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

using IRObjectTy = RecordDumpObjectTy;

struct InputRecordPageObjectAddressing {

  struct ObjectAddressingInfoTy {
    AllocatorTy &Allocator;
    std::vector<MemoryChunkTy> Inputs;
    void addInput(VoidPtrTy Output, size_t Size) {
      Inputs.emplace_back(MemoryChunkTy{});
      auto &Input = Inputs.back();
      Input.Size = Size;
      Input.Location = Output;
      Input.Memory = toPtr(malloc(Size));
      memcpy(Input.Memory, Output, Size);
    }
    ObjectAddressingInfoTy(AllocatorTy &Allocator) : Allocator(Allocator) {}

  } ObjectAddressingInfo;

  template <size_t TopBit>
  struct __attribute__((packed)) StaticSizeObjectTy final {
    static constexpr size_t OutputSize = ((size_t)1) << TopBit;
    static constexpr size_t PointerSize = sizeof(uintptr_t);
    static constexpr size_t NumPointers = OutputSize / sizeof(uintptr_t);
    static constexpr VoidPtrTy Output = nullptr;
    static constexpr size_t PageSizeInBits = 12;
    static constexpr size_t PageSize = (size_t)1 << PageSizeInBits;

    static_assert(TopBit >= PageSizeInBits);
    static constexpr size_t BitsToBeAddressed = TopBit - PageSizeInBits;
    static constexpr size_t NumAddressingBits = (size_t)1 << BitsToBeAddressed;

    static_assert(NumAddressingBits % 8 == 0);

    // Zero initialized by default
    std::bitset<NumAddressingBits> Accessed;

    // VoidPtrTy Output;

    StaticSizeObjectTy() {}
    StaticSizeObjectTy(VoidPtrTy Output) {
      assert(reinterpret_cast<uintptr_t>(Output) % PointerSize == 0);
    }

    std::set<intptr_t> Ptrs;
    // std::unordered_map<intptr_t, uint32_t> FPtrs;

    VoidPtrTy getBasePtr() { return Output; }

    OffsetTy getLocalPtr(VoidPtrTy GlobalPtr) {
      return GlobalPtr - getBasePtr();
    }

    VoidPtrTy getGlobalPtr(OffsetTy LocalPtr) {
      return getBasePtr() + LocalPtr;
    }

    bool isGlobalPtrInObject(VoidPtrTy GlobalPtr) {
      VoidPtrTy BasePtr = getBasePtr();
      return BasePtr <= GlobalPtr && BasePtr + OutputSize > GlobalPtr;
    }

    OffsetTy getOffsetFromObjMemory(OffsetTy Ptr) { return Ptr; }

    void access(VoidPtrTy Ptr, ObjectAddressingInfoTy &Info) {
      static constexpr uintptr_t AddressingMask =
          (((uintptr_t)1) << BitsToBeAddressed) - 1;
      uintptr_t Masked = (toUint(Ptr) >> PageSizeInBits) & AddressingMask;
      if (!Accessed[Masked]) {
        static constexpr uintptr_t PageMask =
            ~((((uintptr_t)1) << PageSizeInBits) - 1);
        Info.addInput(toPtr(toUint(Ptr) & PageMask), PageSize);
        Accessed[Masked] = true;
      }
    }

    void report() { std::cerr << Accessed << std::endl; }

    struct MemoryTy {
      VoidPtrTy Memory;
      intptr_t AllocationSize;
      OffsetTy AllocationOffset;
    };
    MemoryTy getOutputMemory() { return MemoryTy{Output, OutputSize, 0}; }
  };

  static std::string o(unsigned N) { return std::string(N, ' '); }

  template <unsigned X> struct LShift {
    // FIXME we depend on 1 << 64 == 0 is it UB?
    static const uintptr_t Value = ((uintptr_t)1) << X;
  };

  template <unsigned NumBits, template <unsigned TopBit> typename ChildTy,
            unsigned TopBit>
  struct Node {
    static constexpr uintptr_t NumChildren = LShift<NumBits>::Value;
    static constexpr unsigned NextBit = TopBit - NumBits;
    static constexpr unsigned RemainingBits = NextBit;
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
    // This function may move the NodeStorage, so all pointers users have may be
    // invalid after the call. This is why it provides a way to set the Idx
    // before the reallocation happens.
    template <typename T, bool Store = false>
    static std::pair<size_t, T *> allocateNew(IRVector<uint8_t> &NodeStorage,
                                              size_t *IdxStorage = nullptr) {
      unsigned AllocSize = sizeof(T);
      size_t Idx = NodeStorage.size();
      if constexpr (Store) {
        assert(IdxStorage);
        *IdxStorage = Idx;
      } else {
        assert(!IdxStorage);
      }
      NodeStorage.reserve(NodeStorage.size() + AllocSize);
      NodeStorage.insert(NodeStorage.end(), AllocSize, 0);
      VoidPtrTy AllocPtr = NodeStorage.data() + Idx;
      return {Idx, reinterpret_cast<T *>(AllocPtr)};
    }
  };

  template <unsigned TopBit> struct __attribute__((packed)) Leaf {
    using Node = Node<TopBit, Leaf, TopBit>;
    using LocalObjectTy = StaticSizeObjectTy<TopBit>;

    static_assert(Node::NextBit == 0);

    LocalObjectTy Object;

    Leaf() {}
    Leaf(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr) { init(Info, Ptr); }
    void init(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr) {
      [[maybe_unused]]
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Created Leaf Node for Ptr" << toVoidPtr(Ptr)
                               << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << "\n");
      VoidPtrTy BasePtr = reinterpret_cast<VoidPtrTy>(
          reinterpret_cast<uintptr_t>(Ptr) & ~(Node::NumChildren - 1));
      assert(reinterpret_cast<VoidPtrTy>(reinterpret_cast<uintptr_t>(Ptr) &
                                         ~Masked) == BasePtr);
      Object = LocalObjectTy(BasePtr);
    }

    void accessObject(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr,
                      IRVector<uint8_t> &NodeStorage) {
      Object.access(Ptr, Info);
    }
    void report(IRVector<uint8_t> &NodeStorage) {
      std::cerr << o(sizeof(uintptr_t) * 8 - TopBit) << "Leaf Node\n";
      std::cerr << o(sizeof(uintptr_t) * 8 - TopBit);
      Object.report();
    }
  };

  template <unsigned NumBits, template <unsigned TopBit> typename ChildTy,
            unsigned TopBit>
  struct __attribute__((packed)) ArrayNode {
    using Node = Node<NumBits, ChildTy, TopBit>;
    using SpecializedChildTy = ChildTy<Node::NextBit>;
    using LocalObjectTy = typename SpecializedChildTy::LocalObjectTy;

    std::array<SpecializedChildTy, Node::NumChildren> Children;

    ArrayNode(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr) {
      INPUTGEN_DEBUG(std::cerr << "Created Array Node for Ptr " << toBits(Ptr)
                               << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");

      VoidPtrTy BasePtr = reinterpret_cast<VoidPtrTy>(
          reinterpret_cast<uintptr_t>(Ptr) & ~(((uintptr_t)1 << TopBit) - 1));
      uintptr_t Increment = ((uintptr_t)1) << Node::RemainingBits;
      INPUTGEN_DEBUG(std::cerr << "BasePtr " << toBits(BasePtr) << " Increment "
                               << toBits(Increment) << "\n");
      Ptr = BasePtr;
      for (auto &Child : Children) {
        Child.init(Info, Ptr);
        Ptr += Increment;
      }
    }

    void accessObject(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr,
                      IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      Children[Masked].accessObject(Info, Ptr, NodeStorage);
    }
    void report(IRVector<uint8_t> &NodeStorage) {
      std::cerr << o(sizeof(uintptr_t) * 8 - TopBit) << "Array Node" << "\n";
      uintptr_t I = 0;
      for (auto &Child : Children) {
        std::cerr << o(sizeof(uintptr_t) * 8 - TopBit)
                  << toBitsFixed<NumBits>(I++) << std::endl;
        Child.report(NodeStorage);
      }
    }
  };

  template <unsigned NumBits, unsigned NumChildrenPerNode,
            template <unsigned TopBit> typename ChildTy, unsigned TopBit>
  struct __attribute__((packed)) LinkedListNode {
    using Node = Node<NumBits, ChildTy, TopBit>;
    using SpecializedChildTy = ChildTy<Node::NextBit>;
    using LocalObjectTy = typename SpecializedChildTy::LocalObjectTy;

    // TODO something like this where we pack the index of the next node after
    // the PtrMatch would be nice but we may sometimes overflow the next index.
    // (also the allocateNew function cannot take the address of a bitfield as
    // its argument)
    struct LLNodeTyBitfield {
      static constexpr uintptr_t InvalidChildIdx =
          (1 << Node::RemainingBits) - 1;
      uintptr_t PtrMatch : NumBits;
      // this is _not_ enough memory always
      uintptr_t NextIndex : Node::RemainingBits;
      unsigned : 0; // new byte
      uint8_t ChildData[sizeof(typename Node::SpecializedChildTy) *
                        NumChildrenPerNode];
    };

    struct __attribute__((packed)) LLNodeTyUnpacked {
      static constexpr size_t StorageForArraySize =
          getStorageForSize<NumChildrenPerNode>();
      static constexpr size_t StorageForNextIdx =
          (sizeof(uintptr_t) * 8) - StorageForArraySize;
      static constexpr uintptr_t InvalidChildIdx =
          LShift<StorageForNextIdx>::Value - 1;
      using LLIdxType = uintptr_t;
      uintptr_t NumChildren : StorageForArraySize;
      uintptr_t NextIdx : StorageForNextIdx;
      uintptr_t PtrMatch[NumChildrenPerNode];
      uint8_t ChildData[sizeof(typename Node::SpecializedChildTy) *
                        NumChildrenPerNode];

      SpecializedChildTy *getChild(size_t Idx) {
        auto ChildPtr =
            reinterpret_cast<typename Node::SpecializedChildTy *>(ChildData) +
            Idx;
        return ChildPtr;
      }
    };
    using LLNodeTy = LLNodeTyUnpacked;
    static constexpr auto InvalidChildIdx = LLNodeTy::InvalidChildIdx;
    using LLIdxType = typename LLNodeTy::LLIdxType;
    LLNodeTy Head;
    // LLIdxType HeadIdx;

    LinkedListNode() { init(); }
    LinkedListNode(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr) { init(); }
    void init() {
      Head.NumChildren = 0;
      Head.NextIdx = InvalidChildIdx;
      INPUTGEN_DEBUG(std::cerr << "Created Linked List Node\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");
    }

    LLNodeTy *getAt(IRVector<uint8_t> &NodeStorage, LLIdxType Idx) {
      return reinterpret_cast<LLNodeTy *>(&NodeStorage[Idx]);
    }

    size_t getIdx(IRVector<uint8_t> &NodeStorage, LLNodeTy *Node) {
      return toPtr(Node) - NodeStorage.data();
    }

    SpecializedChildTy *addChild(ObjectAddressingInfoTy &Info,
                                 IRVector<uint8_t> &NodeStorage, size_t CurIdx,
                                 LLNodeTy *CurLLNode, VoidPtrTy Ptr) {
      assert(Ptr != nullptr);
      assert(getAt(NodeStorage, CurIdx) == CurLLNode);
      assert(CurIdx == toPtr(CurLLNode) - NodeStorage.data());
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      if (CurLLNode->NumChildren == NumChildrenPerNode) {
        auto [NewIdx, NewLLNode] =
            Node::template allocateNew<LLNodeTy, false>(NodeStorage);
        NewLLNode->PtrMatch[0] = Masked;
        NewLLNode->NextIdx = InvalidChildIdx;
        NewLLNode->NumChildren = 1;
        new (NewLLNode->getChild(0)) SpecializedChildTy(Info, Ptr);
        getAt(NodeStorage, CurIdx)->NextIdx = NewIdx;
        return NewLLNode->getChild(0);
      } else {
        new (CurLLNode->getChild(CurLLNode->NumChildren))
            SpecializedChildTy(Info, Ptr);
        CurLLNode = getAt(NodeStorage, CurIdx);
        CurLLNode->PtrMatch[CurLLNode->NumChildren] = Masked;
        CurLLNode->NumChildren++;
        return CurLLNode->getChild(CurLLNode->NumChildren - 1);
      }
    }

    void accessObject(ObjectAddressingInfoTy &Info, VoidPtrTy Ptr,
                      IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      LLIdxType Idx = getIdx(NodeStorage, &Head), LastIdx;
      LLNodeTy *LLNode = nullptr;
      do {
        LLNode = getAt(NodeStorage, Idx);
        for (unsigned I = 0; I < LLNode->NumChildren; I++) {
          if (LLNode->PtrMatch[I] == Masked)
            return LLNode->getChild(I)->accessObject(Info, Ptr, NodeStorage);
        }
        LastIdx = Idx;
        Idx = LLNode->NextIdx;
      } while (Idx != InvalidChildIdx);
      addChild(Info, NodeStorage, LastIdx, LLNode, Ptr)
          ->accessObject(Info, Ptr, NodeStorage);
    }
    void report(IRVector<uint8_t> &NodeStorage) {
      std::cerr << o(sizeof(uintptr_t) * 8 - TopBit) << "Linked List" << "\n";
      LLIdxType Idx = getIdx(NodeStorage, &Head);
      do {
        std::cerr << o(sizeof(uintptr_t) * 8 - TopBit) << "Node" << "\n";
        LLNodeTy *LLNode = getAt(NodeStorage, Idx);
        for (unsigned I = 0; I < LLNode->NumChildren; I++) {
          std::cerr << o(sizeof(uintptr_t) * 8 - TopBit)
                    << toBitsFixed<NumBits>(LLNode->PtrMatch[I]) << std::endl;
          LLNode->getChild(I)->report(NodeStorage);
        }
        Idx = LLNode->NextIdx;
      } while (Idx != InvalidChildIdx);
    }
  };

  template <unsigned NumBits, template <unsigned> typename ChildTy>
  struct AArrayNode {
    template <unsigned TopBit>
    struct SNode : ArrayNode<NumBits, ChildTy, TopBit> {
      using ArrayNode<NumBits, ChildTy, TopBit>::ArrayNode;
    };
  };
  template <unsigned NumBits, unsigned NumChildrenPerNode,
            template <unsigned> typename ChildTy>
  struct ALinkedListNode {
    template <unsigned TopBit>
    struct SNode
        : LinkedListNode<NumBits, NumChildrenPerNode, ChildTy, TopBit> {
      using LinkedListNode<NumBits, NumChildrenPerNode, ChildTy,
                           TopBit>::LinkedListNode;
    };
  };
  // clang-format off
  using TreeType =
    ALinkedListNode<41, 3,
      Leaf
      >::SNode<
    sizeof(uintptr_t) * 8>;
  // clang-format on
  using LocalObjectTy = typename TreeType::LocalObjectTy;

  std::vector<std::pair<bool, VoidPtrTy>> Ptrs;
  IRVector<uint8_t> NodeStorage;

  TreeType *getTree() {
    return reinterpret_cast<TreeType *>(NodeStorage.data());
  }

  void access(VoidPtrTy Ptr, size_t Size) {
    // TODO need to take into account Size
    getTree()->accessObject(ObjectAddressingInfo, Ptr, NodeStorage);
    INPUTGEN_DEBUG(report());
  }

  void writeArray(VoidPtrTy Ptr, uint32_t Size) {
    for (auto ItPtr = Ptr; ItPtr < Size + Ptr; ItPtr++)
      access(Ptr, 1);
  }
  void readArray(VoidPtrTy Ptr, uint32_t Size) {
    for (auto ItPtr = Ptr; ItPtr < Size + Ptr; ItPtr++)
      access(Ptr, 1);
  }
  template <typename T> void write(VoidPtrTy Ptr, T Val, uint32_t Size) {
    static constexpr size_t TySize = sizeof(T);
    assert(Size == TySize);
    access(Ptr, TySize);
    if constexpr (std::is_pointer<T>::value)
      Ptrs.push_back({true, Ptr});
  }

  template <typename T>
  void read(VoidPtrTy Ptr, VoidPtrTy Base, uint32_t Size) {
    static constexpr size_t TySize = sizeof(T);
    assert(Size == TySize);
    access(Ptr, TySize);
    if constexpr (std::is_pointer<T>::value)
      Ptrs.push_back({false, Ptr});
  }

  IRVector<std::unique_ptr<RecordDumpObjectTy>> DumpObjects;

  void getObjects(IRVector<RecordDumpObjectTy *> &Objects) {
    for (auto &Input : ObjectAddressingInfo.Inputs) {
      // FIXME need to fill in the pointers
      // FIXME use appropriate make_unique
      DumpObjects.push_back(std::make_unique<RecordDumpObjectTy>(Input));
      Objects.push_back(DumpObjects.back().get());
    }
  }

  void report() {
    std::cerr << "Object addressing data structure:\n";
    getTree()->report(NodeStorage);
    std::cerr << "Num pointer accesses: " << Ptrs.size() << "\n";
  }

  InputRecordPageObjectAddressing(AllocatorTy &Allocator)
      : ObjectAddressingInfo(Allocator) {
    NodeStorage.resize(sizeof(TreeType));
    new (getTree()) TreeType();
  }

  InputRecordPageObjectAddressing(AllocatorTy &Allocator,
                                  IRVector<uint8_t> NodeStorage)
      : ObjectAddressingInfo(Allocator), NodeStorage(NodeStorage) {}
};

// For some reason we get a template error if we try to template the IRVector
// type. That's why the implementation is in a different file and we define it
// before the include.
#include "dump-input-impl.hpp"

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

  template <typename T> T generateNewArg(RTInfo *RTI) { abort(); }

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

  template <typename T> T generateNewStubReturn(RTInfo *RTI) { abort(); }

  template <typename T> T getDefaultNewValue() { abort(); }

  template <typename T> T getNewValue(RTInfo *RTI) { abort(); }

  template <> VoidPtrTy getNewValue<VoidPtrTy>(RTInfo *RTI) { abort(); }

  template <> FunctionPtrTy getNewValue<FunctionPtrTy>(RTInfo *RTI) { abort(); }

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
  void read(VoidPtrTy Ptr, VoidPtrTy Base, uint32_t Size, RTInfo *RTI) {
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
      Obj->read<T>(LocalPtr, Size, BHs);
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
