#ifndef _INPUT_GEN_RUNTIMES_RT_COMMON_H_
#define _INPUT_GEN_RUNTIMES_RT_COMMON_H_

#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

#include <array>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../../../llvm/include/llvm/Transforms/IPO/InputGenerationTypes.h"

using BranchHint = llvm::inputgen::BranchHint;

namespace {
int VERBOSE = 0;
int TIMING = 0;
struct InitGlobalsTy {
  InitGlobalsTy() {
    VERBOSE = (bool)getenv("VERBOSE");
    TIMING = (bool)getenv("TIMING");
  }
} InitGlobals;
} // namespace

static uint64_t InputGenMagicNumber = 0x4e45475455504e49; // "INPUTGEN"
static uint32_t InputGenVersion = 0;

enum InputMode {
  InputMode_Generate = 0,
  InputMode_Record_v1 = 1,
  InputMode_Record_v2 = 2,
};
static const char *inputModeToStr(enum InputMode Mode) {
  switch (Mode) {
  case InputMode_Generate:
    return "Generate";
  case InputMode_Record_v1:
    return "Record_v1";
  case InputMode_Record_v2:
    return "Record_v2";
  }
  assert(false && "Invalid input mode");
}

#ifndef NDEBUG
#define INPUTGEN_DEBUG(X)                                                      \
  do {                                                                         \
    if (VERBOSE) {                                                             \
      X;                                                                       \
    }                                                                          \
  } while (0)
#else
#define INPUTGEN_DEBUG(X)
#endif

#define INPUTGEN_TIMER_DEFINE(Name)                                            \
  std::chrono::steady_clock::time_point Timer##Name##Begin

#define INPUTGEN_TIMER_START(Name)                                             \
  do {                                                                         \
    if (TIMING)                                                                \
      Timer##Name##Begin = std::chrono::steady_clock::now();                   \
  } while (0)

#define INPUTGEN_TIMER_END(Name)                                               \
  do {                                                                         \
    if (TIMING) {                                                              \
      std::chrono::steady_clock::time_point Timer##Name##End =                 \
          std::chrono::steady_clock::now();                                    \
      std::cout << "Time for " << #Name << ": "                                \
                << std::chrono::duration_cast<std::chrono::nanoseconds>(       \
                       Timer##Name##End - Timer##Name##Begin)                  \
                       .count()                                                \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

static constexpr intptr_t ObjAlignment = 16;
static constexpr intptr_t MaxPrimitiveTypeSize = 16;

static constexpr int UnreachableExitStatus = 111;

typedef uint8_t *VoidPtrTy;
typedef struct {
} *FunctionPtrTy;

template <typename T> static char *ccast(T *Ptr) {
  return reinterpret_cast<char *>(Ptr);
}

template <typename T> static void *toVoidPtr(T Ptr) {
  return static_cast<void *>(Ptr);
}

template <typename T> static std::bitset<sizeof(T) * 8> toBits(T Ptr) {
  return std::bitset<sizeof(T) * 8>((uintptr_t)Ptr);
}

template <typename T> static T readV(std::ifstream &Input) {
  T El;
  Input.read(ccast(&El), sizeof(El));
  return El;
}

template <typename T> static void writeV(std::ofstream &Output, T El) {
  Output.write(ccast(&El), sizeof(El));
}

struct ObjectTy;
using OffsetTy = intptr_t;
using SizeTy = uintptr_t;

using MallocFuncTy = void *(*)(size_t);
using FreeFuncTy = void (*)(void *);

struct ObjectAllocatorTy {
  MallocFuncTy Malloc;
  FreeFuncTy Free;
};

struct ObjectAddressing {
  virtual size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const = 0;
  virtual OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const = 0;
  virtual OffsetTy getObjBaseOffset() const = 0;
  OffsetTy getOffsetFromObjBasePtr(OffsetTy Ptr) const {
    return Ptr - getObjBaseOffset();
  }
  virtual VoidPtrTy getLowestObjPtr() const = 0;
  virtual uintptr_t getMaxObjectSize() const = 0;
  virtual ~ObjectAddressing(){};
};

template <typename RTTy>
struct InputRecordObjectAddressing : public ObjectAddressing {
  OffsetTy getObjBaseOffset() const override { return 0; }
  VoidPtrTy getLowestObjPtr() const override { abort(); }
  uintptr_t getMaxObjectSize() const override { abort(); }

  RTTy &RT;
  InputRecordObjectAddressing(RTTy &RT) : RT(RT) {}

  OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const override {
    auto Res = RT.globalPtrToObjAndLocalPtr(GlobalPtr);
    assert(Res);
    return Res->second;
  }
  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, OffsetTy PtrInObj) const {
    return RT.localPtrToGlobalPtr(ObjIdx, PtrInObj);
  }

  size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const override { abort(); }

  uintptr_t getSize() { abort(); };
};

struct InputRecordPageObjectAddressing {

  ObjectAllocatorTy &ObjectAllocator;

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

  static std::unique_ptr<ObjectTy>
  makeUniqueObject(ObjectAllocatorTy &, VoidPtrTy BasePtr, size_t Size);

  template <unsigned TopBit> struct Leaf {
    using Node = Node<TopBit, Leaf, TopBit>;

    static_assert(Node::NextBit == 0);

    std::unique_ptr<ObjectTy> Object;

    Leaf(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr) {
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
      Object = makeUniqueObject(ObjectAllocator, BasePtr, Node::NumChildren);
    }

    ObjectTy *getObject(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr,
                        IRVector<uint8_t> &NodeStorage) {
      assert(Object);
      return Object.get();
    }
    void getObjects(IRVector<ObjectTy *> &Objects,
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
    static constexpr size_t InvalidChildIdx =
        std::numeric_limits<size_t>::max();

    std::array<size_t, Node::NumChildren> Children;

    ArrayNode(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr = nullptr) {
      for (auto &ChildIdx : Children)
        // FIXME This should _not_ be one
        ChildIdx = InvalidChildIdx;
      INPUTGEN_DEBUG(std::cerr << "Created Array Node for Ptr " << toBits(Ptr)
                               << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");
    }

    ObjectTy *getObject(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr,
                        IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      SpecializedChildTy *ChildPtr;
      if (Children[Masked] == InvalidChildIdx) {
        size_t Idx;
        std::tie(Idx, ChildPtr) =
            Node::template allocateNew<SpecializedChildTy>(NodeStorage);
        new (ChildPtr) SpecializedChildTy(ObjectAllocator, Ptr);
        Children[Masked] = Idx;
      } else {
        ChildPtr = reinterpret_cast<SpecializedChildTy *>(
            &NodeStorage[Children[Masked]]);
      }
      return ChildPtr->getObject(ObjectAllocator, Ptr, NodeStorage);
    }
    void getObjects(IRVector<ObjectTy *> &Objects,
                    IRVector<uint8_t> &NodeStorage) {
      for (auto &ChildIdx : Children) {
        if (ChildIdx != InvalidChildIdx)
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
    static constexpr uintptr_t InvalidChildIdx = (1 << Node::RemainingBits) - 1;

    // TODO something like this where we pack the index of the next node after
    // the PtrMatch would be nice but we may sometimes overflow the next index.
    // (also the allocateNew function cannot take the address of a bitfield as
    // its argument)
    struct LLNodeTyBitfield {
      uintptr_t PtrMatch : NumBits;
      // this is _not_ enough memory always
      uintptr_t NextIndex : Node::RemainingBits;
      unsigned : 0; // new byte
      uint8_t ChildData[sizeof(typename Node::SpecializedChildTy)];
    };

    struct LLNodeTy {
      uintptr_t PtrMatch;
      size_t NextIndex;
      uint8_t ChildData[sizeof(typename Node::SpecializedChildTy)];

      SpecializedChildTy *getChild() {
        auto ChildPtr =
            reinterpret_cast<typename Node::SpecializedChildTy *>(ChildData);
        return ChildPtr;
      }
    } Head;

    LinkedListNode(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr) {
      INPUTGEN_DEBUG(std::cerr << "Created Linked List Node for Ptr "
                               << toBits(Ptr) << "\n");
      INPUTGEN_DEBUG(std::cerr << "TopBit " << TopBit << " NumChildren "
                               << Node::NumChildren << " NumBits " << NumBits
                               << " NextBit " << Node::NextBit << "\n");
      constructNode(ObjectAllocator, Head, Ptr);
    }

    void constructNode(ObjectAllocatorTy &ObjectAllocator, LLNodeTy &LLNode,
                       VoidPtrTy Ptr) {
      assert(Ptr != nullptr);
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      LLNode.PtrMatch = Masked;
      LLNode.NextIndex = InvalidChildIdx;
      new (LLNode.ChildData) SpecializedChildTy(ObjectAllocator, Ptr);
    }

    ObjectTy *getObject(ObjectAllocatorTy &ObjectAllocator, VoidPtrTy Ptr,
                        IRVector<uint8_t> &NodeStorage) {
      uintptr_t Masked = Node::extractMaskedPart(Ptr);
      INPUTGEN_DEBUG(std::cerr << "Getting object for " << toBits(Ptr)
                               << " Masked " << toBits(Masked) << "\n");
      SpecializedChildTy *ChildPtr;
      LLNodeTy *LLNode = &Head;
      while (LLNode->PtrMatch != Masked) {
        if (LLNode->NextIndex != InvalidChildIdx) {
          LLNode =
              reinterpret_cast<LLNodeTy *>(&NodeStorage[LLNode->NextIndex]);
        } else {
          auto [Idx, NewLLNode] = Node::template allocateNew<LLNodeTy, true>(
              NodeStorage, &LLNode->NextIndex);
          constructNode(ObjectAllocator, *NewLLNode, Ptr);
          LLNode = NewLLNode;
          break;
        }
      }
      ChildPtr = LLNode->getChild();
      return ChildPtr->getObject(ObjectAllocator, Ptr, NodeStorage);
    }
    void getObjects(IRVector<ObjectTy *> &Objects,
                    IRVector<uint8_t> &NodeStorage) {
      LLNodeTy *LLNode = &Head;
      LLNode->getChild()->getObjects(Objects, NodeStorage);
      while (LLNode->NextIndex != InvalidChildIdx) {
        LLNode = reinterpret_cast<LLNodeTy *>(&NodeStorage[LLNode->NextIndex]);
        LLNode->getChild()->getObjects(Objects, NodeStorage);
      }
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
  using TreeType =
    ArrayNode<0,
      ALinkedListNode<44,
      Leaf
      >::SNode,
    sizeof(uintptr_t) * 8>;
  // clang-format on
  TreeType Tree;
  IRVector<uint8_t> NodeStorage;
  ObjectTy *getObject(VoidPtrTy Ptr) {
    return Tree.getObject(ObjectAllocator, Ptr, NodeStorage);
  }
  void getObjects(IRVector<ObjectTy *> &Objects) {
    Tree.getObjects(Objects, NodeStorage);
  }
  InputRecordPageObjectAddressing(ObjectAllocatorTy &ObjectAllocator)
      : ObjectAllocator(ObjectAllocator), Tree(ObjectAllocator) {}
  InputRecordPageObjectAddressing(ObjectAllocatorTy &ObjectAllocator,
                                  TreeType Tree, IRVector<uint8_t> NodeStorage)
      : ObjectAllocator(ObjectAllocator), Tree(Tree), NodeStorage(NodeStorage) {
  }
};

struct InputGenObjectAddressing : public ObjectAddressing {
  ~InputGenObjectAddressing(){};
  size_t globalPtrToObjIdx(VoidPtrTy GlobalPtr) const override {
    size_t Idx =
        (reinterpret_cast<intptr_t>(GlobalPtr) & ObjIdxMask) / MaxObjectSize;
    return Idx - ObjIdxOffset;
  }

  OffsetTy globalPtrToLocalPtr(VoidPtrTy GlobalPtr) const override {
    return reinterpret_cast<intptr_t>(GlobalPtr) & PtrInObjMask;
  }

  OffsetTy getObjBaseOffset() const override { return MaxObjectSize / 2; }

  VoidPtrTy localPtrToGlobalPtr(size_t ObjIdx, OffsetTy PtrInObj) const {
    return reinterpret_cast<VoidPtrTy>(
        ((ObjIdxOffset + ObjIdx) * MaxObjectSize) |
        reinterpret_cast<intptr_t>(PtrInObj));
  }

  uintptr_t getMaxObjectSize() const override { return MaxObjectSize; }
  VoidPtrTy getLowestObjPtr() const override { return nullptr; }
  uintptr_t getSize() { return Size; };
  uintptr_t getMaxObjectNum() { return MaxObjectNum; }
  uintptr_t getMaxObjectSize() { return MaxObjectSize; }

  void setObjIdxOffset(uintptr_t Offset) { ObjIdxOffset = Offset; }

  void setSize(uintptr_t Size) {
    this->Size = Size;

    uintptr_t HO = highestOne(Size | 1);
    uintptr_t BitsForObj = HO * 70 / 100;
    uintptr_t BitsForObjIndexing = HO - BitsForObj;
    MaxObjectSize = 1ULL << BitsForObj;
    MaxObjectNum = 1ULL << (BitsForObjIndexing);
    PtrInObjMask = MaxObjectSize - 1;
    ObjIdxMask = ~(PtrInObjMask);
    INPUTGEN_DEBUG(std::cerr << "OA " << BitsForObj
                             << " bits for in-object addressing and "
                             << BitsForObjIndexing << " for object indexing\n");
  }

private:
  intptr_t PtrInObjMask;
  intptr_t ObjIdxMask;
  uintptr_t MaxObjectSize;
  uintptr_t MaxObjectNum;

  uintptr_t ObjIdxOffset = 0;
  uintptr_t Size = 0;

  unsigned int highestOne(uint64_t X) { return 63 ^ __builtin_clzll(X); }
};

static std::string getFunctionNameFromFile(std::string FileName,
                                           std::string FuncIdent) {
  std::string OriginalFuncName;
  std::ifstream In(FileName);
  std::string Id;
  while (std::getline(In, Id, '\0') &&
         std::getline(In, OriginalFuncName, '\0') && Id != FuncIdent)
    ;
  if (Id != FuncIdent) {
    std::cerr << "Could not find function with ID " << FuncIdent << " in "
              << FileName << std::endl;
    abort();
  }
  return OriginalFuncName;
}

static void useValue(VoidPtrTy Ptr, uint32_t Size) {
  if (getenv("___INPUT_GEN_USE___"))
    for (unsigned I = 0; I < Size; I++)
      printf("%c\n", *(Ptr + Size));
}

static constexpr intptr_t MinObjAllocation = 64;
static constexpr unsigned NullPtrProbability = 75;
static constexpr int CmpPtrRetryProbability = 10;
static constexpr int MaxDeviationFromBranchHint = 10;

template <typename T> static T divFloor(T A, T B) {
  assert(B > 0);
  T Res = A / B;
  T Rem = A % B;
  if (Rem == 0)
    return Res;
  if (Rem < 0) {
    assert(A < 0);
    return Res - 1;
  }
  assert(A > 0);
  return Res;
}

template <typename T> static T divCeil(T A, T B) {
  assert(B > 0);
  T Res = A / B;
  T Rem = A % B;
  if (Rem == 0)
    return Res;
  if (Rem > 0) {
    assert(A > 0);
    return Res + 1;
  }
  assert(A < 0);
  return Res;
}

template <typename T> static T alignStart(T Ptr, intptr_t Alignment) {
  intptr_t IPtr = reinterpret_cast<intptr_t>(Ptr);
  return reinterpret_cast<T>(divFloor(IPtr, Alignment) * Alignment);
}

template <typename T> static T alignEnd(T Ptr, intptr_t Alignment) {
  intptr_t IPtr = reinterpret_cast<intptr_t>(Ptr);
  return reinterpret_cast<T>(divCeil(IPtr, Alignment) * Alignment);
}

static VoidPtrTy advance(VoidPtrTy Ptr, uint64_t Bytes) {
  return reinterpret_cast<uint8_t *>(Ptr) + Bytes;
}

// TODO There should also be a template argument to the class which says whether
// the Input and Used should expand dynamically
struct ObjectTy {
  ObjectTy(ObjectAllocatorTy &Allocator, VoidPtrTy Output, size_t Size,
           OffsetTy Offset, bool AllocateInputUsed,
           bool KnownSizeObjBundle = false)
      : KnownSizeObjBundle(KnownSizeObjBundle), Output(Allocator),
        Input(Allocator), Used(Allocator) {
    INPUTGEN_DEBUG(std::cerr << "Creating Object at #" << this
                             << " with memory " << (void *)Output << " size "
                             << Size << " offset " << Offset << "\n");
    this->Output.Memory = Output;
    this->Output.AllocationSize = Size;
    this->Output.AllocationOffset = Offset;

    if (AllocateInputUsed) {
      this->Input.ensureAllocation<true>(0, Size);
      // TODO include a mode where we do not tracked Used on per-byte basis, and
      // only the range
      this->Used.ensureAllocation<true>(0, Size);
    }
    if (KnownSizeObjBundle) {
      CurrentStaticObjEnd = Offset;
    }
  }
  ~ObjectTy() {}

  struct AlignedMemoryChunk {
    VoidPtrTy Ptr;
    intptr_t InputSize;
    intptr_t InputOffset;
    intptr_t OutputSize;
    intptr_t OutputOffset;
    intptr_t CmpSize;
    intptr_t CmpOffset;
  };

  bool KnownSizeObjBundle;
  OffsetTy CurrentStaticObjEnd;

  const auto &getOutputMemory() { return Output; }

  VoidPtrTy getBasePtr() { return Output.Memory - Output.AllocationOffset; }

  OffsetTy getLocalPtr(VoidPtrTy GlobalPtr) { return GlobalPtr - getBasePtr(); }

  VoidPtrTy getGlobalPtr(OffsetTy LocalPtr) { return getBasePtr() + LocalPtr; }

  bool isGlobalPtrInObject(VoidPtrTy GlobalPtr) {
    VoidPtrTy BasePtr = getBasePtr();
    return BasePtr <= GlobalPtr && BasePtr + Output.AllocationSize > GlobalPtr;
  }

  OffsetTy getOffsetFromObjMemory(OffsetTy Ptr) {
    return Ptr - Output.AllocationOffset;
  }

#if 0
  VoidPtrTy addKnownSizeObject(uintptr_t Size) {
    assert(KnownSizeObjBundle);
    // Make sure zero-sized objects have their own address
    if (Size == 0)
      Size = 1;
    if (Size + CurrentStaticObjEnd >
        OA.getLowestObjPtr() + OA.getMaxObjectSize())
      return nullptr;
    VoidPtrTy ObjPtr = CurrentStaticObjEnd;
    CurrentStaticObjEnd = alignEnd(CurrentStaticObjEnd + Size, ObjAlignment);
    return ObjPtr;
  }

  struct KnownSizeObjInputMem {
    VoidPtrTy Start;
    uintptr_t Size;
  };
  KnownSizeObjInputMem getKnownSizeObjectInputMemory(VoidPtrTy LocalPtr,
                                                     uintptr_t Size) {
    assert(KnownSizeObjBundle);
    KnownSizeObjInputMem Mem;
    Mem.Start = std::min(
        LocalPtr + Size,
        std::max(LocalPtr, OA.getObjBaseOffset() + InputLimits.LowestOffset));
    VoidPtrTy End = std::max(
        LocalPtr, std::min(LocalPtr + Size,
                           OA.getObjBaseOffset() + InputLimits.HighestOffset));
    Mem.Size = End - Mem.Start;
    assert(Mem.Start <= End);
    return Mem;
  }

  void comparedAt(VoidPtrTy GlobalPtr) {
    OffsetTy Offset = getLocalPtr(GlobalPtr);
    CmpLimits.update(Offset, 1);
  }
#endif

  AlignedMemoryChunk getAlignedInputMemory() {
    // If we compare the pointer at some offset we need to make sure the output
    // allocation will contain those locations, otherwise comparisons may differ
    // in input-gen and input-run as we would compare against an offset in a
    // different object
    if (!OutputLimits.isEmpty()) {
      if (!CmpLimits.isEmpty())
        OutputLimits.update(CmpLimits.LowestOffset, CmpLimits.getSize());
      // We no longer need the CmpLimits, reset it
      CmpLimits = Limits();
    }

    VoidPtrTy InputStart =
        InputLimits.LowestOffset + Input.Memory - Input.AllocationOffset;
    VoidPtrTy InputEnd =
        InputLimits.HighestOffset + Input.Memory - Input.AllocationOffset;
    intptr_t OutputStart = alignStart(OutputLimits.LowestOffset, ObjAlignment);
    intptr_t OutputEnd = alignEnd(OutputLimits.HighestOffset, ObjAlignment);
    return {InputStart,
            InputEnd - InputStart,
            InputLimits.LowestOffset,
            OutputEnd - OutputStart,
            OutputStart,
            CmpLimits.getSize(),
            CmpLimits.LowestOffset};
  }

  template <typename T>
  T read(OffsetTy Ptr, uint32_t Size, BranchHint *BHs, int32_t BHSize);
  template <typename T>
  T readFromGlobalPtr(VoidPtrTy Ptr, uint32_t Size, BranchHint *BHs,
                      int32_t BHSize) {
    return read<T>(getLocalPtr(Ptr), Size, BHs, BHSize);
  }

  template <typename T> void write(T Val, OffsetTy Ptr, uint32_t Size) {
    intptr_t Offset = getOffsetFromObjMemory(Ptr);
    assert(Output.isAllocated(Offset, Size));
    Used.ensureAllocation(Offset, Size);
    markUsed(Offset, Size);
    OutputLimits.update(Offset, Size);
  }
  template <typename T>
  void writeToGlobalPtr(T Val, VoidPtrTy Ptr, uint32_t Size) {
    write<T>(Val, getLocalPtr(Ptr), Size);
  }

  void setFunctionPtrIdx(OffsetTy Ptr, uint32_t Size, VoidPtrTy FPtr,
                         uint32_t FIdx) {
    intptr_t Offset = getOffsetFromObjMemory(Ptr);
    storeInputValue(FPtr, Offset, Size);
    FPtrs.insert({Offset, FIdx});
  }

  std::set<intptr_t> Ptrs;
  std::unordered_map<intptr_t, uint32_t> FPtrs;

public:
  struct MemoryTy {
    ObjectAllocatorTy &Allocator;
    MemoryTy(ObjectAllocatorTy &Allocator) : Allocator(Allocator) {}
    VoidPtrTy Memory = nullptr;
    intptr_t AllocationSize = 0;
    OffsetTy AllocationOffset = 0;
    bool isAllocated(intptr_t Offset, uint32_t Size) {
      intptr_t AllocatedMemoryStartOffset = AllocationOffset;
      intptr_t AllocatedMemoryEndOffset =
          AllocatedMemoryStartOffset + AllocationSize;
      return (AllocatedMemoryStartOffset <= Offset &&
              AllocatedMemoryEndOffset >= Offset + Size);
    }

    /// Returns true if it was already allocated
    template <bool Exact = false>
    bool ensureAllocation(intptr_t Offset, uint32_t Size) {
      if (isAllocated(Offset, Size))
        return true;
      reallocateData<Exact>(Offset, Size);
      return false;
    }

    template <typename T>
    void extendMemory(T *&OldMemory, intptr_t NewAllocationSize,
                      intptr_t NewAllocationOffset) {
      T *NewMemory = reinterpret_cast<T *>(Allocator.Malloc(NewAllocationSize));
      assert(NewMemory && "Malloc failed");
      memset(NewMemory, 0, NewAllocationSize);
      memcpy(advance(NewMemory, AllocationOffset - NewAllocationOffset),
             OldMemory, AllocationSize);
      Allocator.Free(OldMemory);
      OldMemory = NewMemory;
    };

    /// Reallocates the data so as to make the memory at `Offset` with length
    /// `Size` available
    template <bool Exact = false>
    void reallocateData(intptr_t Offset, uint32_t Size) {
      assert(!isAllocated(Offset, Size));

      intptr_t AllocatedMemoryStartOffset = AllocationOffset;
      intptr_t AllocatedMemoryEndOffset =
          AllocatedMemoryStartOffset + AllocationSize;
      intptr_t NewAllocatedMemoryStartOffset = AllocatedMemoryStartOffset;
      intptr_t NewAllocatedMemoryEndOffset = AllocatedMemoryEndOffset;

      intptr_t AccessStartOffset = Offset;
      intptr_t AccessEndOffset = AccessStartOffset + Size;

      if (AccessStartOffset < AllocatedMemoryStartOffset) {
        // Extend the allocation in the negative direction
        if constexpr (Exact) {
          NewAllocatedMemoryStartOffset =
              alignStart(AccessStartOffset, ObjAlignment);
        } else {
          NewAllocatedMemoryStartOffset = alignStart(
              std::min(2 * AccessStartOffset, -MinObjAllocation), ObjAlignment);
        }
      }
      if (AccessEndOffset >= AllocatedMemoryEndOffset) {
        // Extend the allocation in the positive direction
        if constexpr (Exact) {
          NewAllocatedMemoryEndOffset = alignEnd(AccessEndOffset, ObjAlignment);
        } else {
          NewAllocatedMemoryEndOffset = alignEnd(
              std::max(2 * AccessEndOffset, MinObjAllocation), ObjAlignment);
        }
      }

      intptr_t NewAllocationOffset = NewAllocatedMemoryStartOffset;
      intptr_t NewAllocationSize =
          NewAllocatedMemoryEndOffset - NewAllocatedMemoryStartOffset;

      INPUTGEN_DEBUG(
          printf("Reallocating data in Object for access at %ld with size %d "
                 "from offset "
                 "%ld, size %ld to offset %ld, size %ld.\n",
                 Offset, Size, AllocationOffset, AllocationSize,
                 NewAllocationOffset, NewAllocationSize));

      extendMemory(Memory, NewAllocationSize, NewAllocationOffset);

      AllocationSize = NewAllocationSize;
      AllocationOffset = NewAllocationOffset;
    }
  };

private:
  MemoryTy Output, Input, Used;

  struct Limits {
    bool Initialized = false;
    intptr_t LowestOffset = 0;
    intptr_t HighestOffset = 0;
    bool isEmpty() { return !Initialized; }
    intptr_t getSize() { return HighestOffset - LowestOffset; }
    void update(intptr_t Offset, uint32_t Size) {
      if (!Initialized) {
        Initialized = true;
        LowestOffset = Offset;
        HighestOffset = Offset + Size;
        return;
      }
      if (LowestOffset > Offset)
        LowestOffset = Offset;
      if (HighestOffset < Offset + Size)
        HighestOffset = Offset + Size;
    }
  };
  Limits InputLimits, OutputLimits, CmpLimits;

  bool allUsed(intptr_t Offset, uint32_t Size) {
    for (unsigned It = 0; It < Size; It++)
      if (!Used.isAllocated(Offset + It, 1) ||
          !Used.Memory[Offset + It - Used.AllocationOffset])
        return false;
    return true;
  }

  void markUsed(intptr_t Offset, uint32_t Size) {
    assert(Used.isAllocated(Offset, Size));

    for (unsigned It = 0; It < Size; It++)
      Used.Memory[Offset + It - Used.AllocationOffset] = 1;
  }

  template <typename T>
  void storeInputValue(T Val, intptr_t Offset, uint32_t Size) {
    assert(Size == sizeof(Val));

    // Only assign the bytes that were uninitialized
    uint8_t Bytes[sizeof(Val)];
    memcpy(Bytes, &Val, sizeof(Val));
    for (unsigned It = 0; It < sizeof(Val); It++) {
      if (!allUsed(Offset + It, 1)) {
        VoidPtrTy OutputLoc =
            Output.Memory - Output.AllocationOffset + Offset + It;
        VoidPtrTy InputLoc =
            Input.Memory - Input.AllocationOffset + Offset + It;
        *OutputLoc = Bytes[It];
        *InputLoc = Bytes[It];
        markUsed(Offset + It, 1);
      }
    }

    InputLimits.update(Offset, Size);
    OutputLimits.update(Offset, Size);
  }
};

struct GenValTy {
  uint8_t Content[MaxPrimitiveTypeSize] = {0};
  static_assert(sizeof(Content) == MaxPrimitiveTypeSize);
  int32_t IsPtr;
};

template <typename T> static GenValTy toGenValTy(T A, int32_t IsPtr) {
  GenValTy U;
  static_assert(sizeof(T) <= sizeof(U.Content));
  memcpy(U.Content, &A, sizeof(A));
  U.IsPtr = IsPtr;
  return U;
}

#endif // _INPUT_GEN_RUNTIMES_RT_COMMON_H_
