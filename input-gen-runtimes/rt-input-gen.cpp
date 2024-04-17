#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

static int VERBOSE = 0;

static void *advance(void *Ptr, uint64_t Bytes) {
  return reinterpret_cast<char *>(Ptr) + Bytes;
}

struct ObjectTy {
  ObjectTy(void *Data, uint64_t Size, bool Artifical = true)
      : Idx(getObjIdx()), Size(Size), Data(Data) {}
  ObjectTy(const ObjectTy &Other)
      : Idx(Other.Idx), Size(Other.Size), Data(Other.Data) {}

  uint64_t getSize() const { return Size; }
  void *begin() { return Data; }
  void *end() { return advance(Data, Size); }
  static int32_t getObjIdx() {
    static int32_t ObjIdxCnt = 0;
    return ++ObjIdxCnt;
  }

  const uint32_t Idx;

private:
  uint64_t Size = 0;
  void *Data = nullptr;
};
struct ObjectCmpTy {
  bool operator()(const ObjectTy *LHS, const ObjectTy *RHS) const {
    return LHS->Idx < RHS->Idx;
  };
};

struct ArgTy {
  uintptr_t Content;
  int32_t ObjIdx;
};

template <typename T> static T fromArgTy(ArgTy U) {
  static_assert(sizeof(T) <= sizeof(U));
  T A;
  memcpy(&A, &U.Content, sizeof(A));
  return A;
}

template <typename T> static ArgTy toArgTy(T A, int32_t ObjIdx) {
  ArgTy U;
  static_assert(sizeof(T) <= sizeof(U));
  memcpy(&U.Content, &A, sizeof(A));
  U.ObjIdx = ObjIdx;
  return U;
}

static constexpr uint64_t HeapSize = 1UL << 32;

struct HeapTy : ObjectTy {
  HeapTy(HeapTy *LastHeap = nullptr)
      : ObjectTy(malloc(HeapSize), HeapSize), LastHeap(LastHeap) {
    if (VERBOSE)
      printf("New heap [%p:%p)\n", begin(), end());
  }
  HeapTy *LastHeap = nullptr;

  std::bitset<(HeapSize / 8)> UsedSet;

  bool isUsed(void *Ptr, int64_t Size) {
    uintptr_t Offset = (uintptr_t)Ptr - (uintptr_t)begin();
    uintptr_t Idx = Offset / 8;
    bool Res = true;
    do {
      Res &= UsedSet[Idx++];
      Size -= 1;
    } while (Size > 0);
    return Res;
  }
  void markUsed(void *Ptr, int64_t Size) {
    uintptr_t Offset = (uintptr_t)Ptr - (uintptr_t)begin();
    uintptr_t Idx = Offset / 8;
    do {
      UsedSet.set(Idx++);
      Size -= 1;
    } while (Size > 0);
  }

  template <typename T> T read(void *Ptr, void *Base, uint32_t Size);

  template <typename T>
  void write(T *Ptr, T Val, uint32_t Size, bool DueToRead = false,
             int32_t ObjIdx = -1) {
    if (begin() <= Ptr && advance(Ptr, Size) < end()) {
      if (isUsed(Ptr, Size))
        return;
      markUsed(Ptr, Size);
      if (DueToRead)
        memcpy(Ptr, &Val, Size);
      if constexpr (std::is_pointer<T>::value) {
        if (DueToRead && ObjIdx != -1)
          PtrMap[Ptr] = ObjIdx;
      }
      ValMap[Ptr] = {(uintptr_t)Val, Size};
    } else if (LastHeap) {
      LastHeap->write(Ptr, Val, Size, DueToRead);
    } else {
      if (VERBOSE)
        printf("Out of bound write at %p\n", (void *)Ptr);
      // exit(1);
    }
  }

  std::map<void *, int32_t> PtrMap;
  std::map<void *, std::pair<uintptr_t, uint32_t>> ValMap;
};

std::vector<int32_t> GetObjects;

struct InputGenRTTy {
  InputGenRTTy(const char *ExecPath, const char *OutputDir, int Seed)
      : Seed(Seed), OutputDir(OutputDir), ExecPath(ExecPath),
        Heap(new HeapTy()) {
    Gen.seed(Seed);
    SeedStub = rand(false);
    GenStub.seed(SeedStub);
  }
  ~InputGenRTTy() { report(); }

  int32_t Seed, SeedStub;
  std::string OutputDir;
  std::filesystem::path ExecPath;
  std::mt19937 Gen, GenStub;
  std::uniform_int_distribution<> Rand;
  std::vector<char> Conds;

  int rand(bool Stub) { return Stub ? Rand(GenStub) : Rand(Gen); }

  static ObjectTy *getNewObjImpl(uint64_t Size, bool Artifical,
                                 ObjectTy *&LastObj, HeapTy *&Heap) {
    int64_t AlignedSize = Size + (-Size & (16 - 1));
    if (VERBOSE)
      printf("Get new obj %lu :: %lu\n", Size, AlignedSize);
    void *Loc;
    if (LastObj && advance(LastObj->end(), AlignedSize) < Heap->end()) {
      Loc = LastObj->end();
    } else {
      if (LastObj)
        Heap = new HeapTy(Heap);
      Loc = Heap->begin();
    }
    LastObj = new ObjectTy(Loc, AlignedSize, Artifical);
    if (VERBOSE)
      printf("Obj %p :: %p\n", LastObj->begin(), LastObj->end());
    return LastObj;
  }

  ObjectTy *getNewObj(uint64_t Size, bool Artifical) {
    auto *Obj = getNewObjImpl(Size, Artifical, LastObj, Heap);
    Objects.insert(Obj);
    return Obj;
  }

  template <typename T>
  T getNewValue(int32_t *ObjIdx = nullptr, bool Stub = false, int Max = 10) {
    NumNewValues++;
    T V = rand(Stub) % Max;
    return V;
  }

  template <> void *getNewValue<void *>(int32_t *ObjIdx, bool Stub, int Max) {
    NumNewValues++;
    memset(Storage, 0, 64);
    if (rand(Stub) % 50) {
      ObjectTy *Obj = getNewObj(1024 * 1024, true);
      if (ObjIdx)
        *ObjIdx = Obj->Idx;
      *reinterpret_cast<void **>(Storage) = Obj->begin();
    }
    void *V = *((void **)(Storage));
    return V;
  }

  void registerGlobal(void *Global, void **ReplGlobal, int32_t GlobalSize) {
    auto *Obj = getNewObj(GlobalSize, false);
    Globals.push_back(Obj);
    if (VERBOSE)
      printf("Global %p replaced with %p @ %p\n", Global, Obj->begin(),
             ReplGlobal);
    *ReplGlobal = Obj->begin();
  }

  std::vector<ObjectTy *> Globals;

  uint64_t NumNewValues = 0;
  char Storage[64];

  std::vector<ArgTy> Args;

  ObjectTy *LastObj = 0;
  std::set<ObjectTy *, ObjectCmpTy> Objects;

  HeapTy *Heap;

  void report() {
    if (OutputDir == "-") {
      // TODO cross platform
      std::ofstream Null("/dev/null");
      report(stdout, Null);
    } else {
      auto FileName = ExecPath.filename().string();
      std::string ReportOutName(OutputDir + "/" + FileName + ".report." +
                                std::to_string(Seed) + ".txt");
      std::string InputOutName(OutputDir + "/" + FileName + ".input." +
                               std::to_string(Seed) + ".bin");
      std::ofstream InputOutStream(InputOutName,
                                   std::ios::out | std::ios::binary);
      FILE *ReportOutFD = fopen(ReportOutName.c_str(), "w");
      if (!ReportOutFD) {
        fprintf(stderr, "Could not open %s\n", ReportOutName.c_str());
        return;
      }
      report(ReportOutFD, InputOutStream);
      fclose(ReportOutFD);
    }
  }

  template <typename T> const char *ccast(T *Ptr) {
    return reinterpret_cast<const char *>(Ptr);
  }
  template <typename T> T writeSingleEl(std::ofstream &Output, T El) {
    Output.write(ccast(&El), sizeof(El));
    return El;
  }

  void report(FILE *ReportOut, std::ofstream &InputOut) {
    fprintf(ReportOut, "Args (%zu total)\n", Args.size());
    for (size_t I = 0; I < Args.size(); ++I)
      fprintf(ReportOut, "Arg %zu: %p\n", I, (void *)Args[I].Content);
    fprintf(ReportOut, "Num new values: %lu\n", NumNewValues);
    fprintf(ReportOut, "Heap PtrMap: %lu\n", Heap->PtrMap.size());
    fprintf(ReportOut, "Heap ValMap: %lu\n", Heap->ValMap.size());
    fprintf(ReportOut, "Objects (%zu total)\n", Objects.size());

    writeSingleEl(InputOut, SeedStub);

    auto BeforeTotalSize = InputOut.tellp();
    uint64_t TotalSize = 0;
    writeSingleEl(InputOut, TotalSize);

    uint32_t NumObjects = Objects.size();
    writeSingleEl(InputOut, NumObjects);

    std::map<void *, ObjectTy *> TrimmedObjs;
    std::map<uint64_t, void *> Remap;
    for (auto &It : Objects) {
      auto *ObjLIt = It->begin();
      auto *ObjRIt = It->end();
      if (VERBOSE)
        printf("%p : %p :: %lu\n", ObjLIt, ObjRIt, It->getSize());
      while (ObjRIt != ObjLIt) {
        if (Heap->isUsed(ObjLIt, 1))
          break;
        ObjLIt = advance(ObjLIt, 1);
      }
      if (VERBOSE)
        printf("%p : %p\n", ObjLIt, ObjRIt);
      while (ObjRIt != ObjLIt) {
        if (Heap->isUsed(advance(ObjRIt, -1), 1))
          break;
        ObjRIt = advance(ObjRIt, -1);
      }
      uint64_t Size =
          reinterpret_cast<char *>(ObjRIt) - reinterpret_cast<char *>(ObjLIt);
      if (VERBOSE)
        printf("Size %lu\n", Size);

      writeSingleEl(InputOut, It->Idx);
      writeSingleEl(InputOut, TotalSize);

      TotalSize += Size;
      if (ObjLIt != ObjRIt)
        TrimmedObjs[ObjLIt] = It;
    }

    if (VERBOSE)
      printf("TotalSize %lu\n", TotalSize);
    auto BeforeNumGlobals = InputOut.tellp();
    InputOut.seekp(BeforeTotalSize);
    writeSingleEl(InputOut, TotalSize);
    InputOut.seekp(BeforeNumGlobals);

    uint32_t NumGlobals = Globals.size();
    writeSingleEl(InputOut, NumGlobals);

    for (uint32_t I = 0; I < NumGlobals; ++I) {
      writeSingleEl(InputOut, Globals[I]->Idx);
    }

    // std::map<void *, std::pair<uintptr_t, uint32_t>> ValMap;
    auto End = TrimmedObjs.end();
    if (TrimmedObjs.empty() && !Heap->ValMap.empty()) {
      printf("Problem, no objects!");
      exit(2);
    }

    uint32_t NumVals = Heap->ValMap.size();
    writeSingleEl(InputOut, NumVals);

    for (auto &ValIt : Heap->ValMap) {
      auto It = TrimmedObjs.upper_bound(ValIt.first);
      if (It == TrimmedObjs.begin()) {
        printf("Problem, it is begin()");
        exit(3);
      }
      --It;
      ptrdiff_t Offset = reinterpret_cast<char *>(ValIt.first) -
                         reinterpret_cast<char *>(It->second->begin());
      assert(Offset >= 0);
      writeSingleEl(InputOut, It->second->Idx);
      writeSingleEl(InputOut, Offset);
      auto PtrIt = Heap->PtrMap.find(ValIt.first);

      // Write the obj idx next if its a pointer or the value
      enum Kind : uint32_t {
        IDX = 0,
        CONTENT = 1,
      };
      uintptr_t Content;
      if (PtrIt != Heap->PtrMap.end()) {
        writeSingleEl(InputOut, /* Enum */ Kind::IDX);
        Content = PtrIt->second;
      } else {
        writeSingleEl(InputOut, /* Enum */ Kind::CONTENT);
        Content = ValIt.second.first;
      }
      if (VERBOSE)
        printf("%lu ---> %lu [%i]\n", Offset, Content,
               (PtrIt == Heap->PtrMap.end()));
      writeSingleEl(InputOut, Content);
      // Write the size
      writeSingleEl(InputOut, ValIt.second.second);
    }

    uint32_t NumArgs = Args.size();
    writeSingleEl(InputOut, NumArgs);
    for (auto &Arg : Args) {
      writeSingleEl(InputOut, Arg.Content);
      writeSingleEl(InputOut, Arg.ObjIdx);
    }

    uint32_t NumGetObjects = GetObjects.size();
    writeSingleEl(InputOut, NumGetObjects);
    for (auto ObjIdx : GetObjects) {
      writeSingleEl(InputOut, ObjIdx);
    }
  }
};

static InputGenRTTy *InputGenRT;
#pragma omp threadprivate(InputGenRT)

static InputGenRTTy &getInputGenRT() { return *InputGenRT; }

template <typename T> T HeapTy::read(void *Ptr, void *Base, uint32_t Size) {
  if (begin() > Ptr || advance(Ptr, Size) >= end()) {
    if (LastHeap)
      return LastHeap->read<T>(Ptr, Base, Size);
    if (VERBOSE)
      printf("Out of bound read at %p < %p:%p < %p\n", begin(), Ptr,
             advance(Ptr, sizeof(T)), end());
    return *reinterpret_cast<T *>(Ptr);
  }
  if (!isUsed(Ptr, Size)) {
    int32_t ObjIdx = -1;
    T V = getInputGenRT().getNewValue<T>(&ObjIdx);
    write((T *)Ptr, V, Size, true, ObjIdx);
    assert(isUsed(Ptr, Size));
  }
  assert(begin() <= Ptr && advance(Ptr, Size) < end());
  return *reinterpret_cast<T *>(Ptr);
}

extern "C" {
void __inputgen_version_mismatch_check_v1() {}

void __inputgen_init() {
  // getInputGenRT().init();
}
void __inputgen_deinit() {
  // getInputGenRT().init();
}

void __inputgen_global(int32_t NumGlobals, void *Global, void **ReplGlobal,
                       int32_t GlobalSize) {
  getInputGenRT().registerGlobal(Global, ReplGlobal, GlobalSize);
}

void *__inputgen_memmove(void *Tgt, void *Src, uint64_t N) {
  char *SrcIt = (char *)Src;
  char *TgtIt = (char *)Tgt;
  for (uintptr_t I = 0; I < N; ++I, ++SrcIt, ++TgtIt) {
    auto V = getInputGenRT().Heap->read<char>(SrcIt, Src, sizeof(char));
    getInputGenRT().Heap->write<char>(TgtIt, V, sizeof(char));
  }
  return TgtIt;
}
void *__inputgen_memcpy(void *Tgt, void *Src, uint64_t N) {
  return __inputgen_memmove(Tgt, Src, N);
}

void *__inputgen_memset(void *Tgt, char C, uint64_t N) {
  char *TgtIt = (char *)Tgt;
  for (uintptr_t I = 0; I < N; ++I, ++TgtIt) {
    getInputGenRT().Heap->write<char>(TgtIt, C, sizeof(char));
  }
  return TgtIt;
}

#define RW(TY, NAME)                                                           \
  TY __inputgen_get_##NAME() {                                                 \
    int32_t ObjIdx = -1;                                                       \
    TY V = getInputGenRT().getNewValue<TY>(&ObjIdx, true);                     \
    if constexpr (std::is_pointer<TY>::value)                                  \
      GetObjects.push_back(ObjIdx);                                            \
    return V;                                                                  \
  }                                                                            \
  void __inputgen_read_##NAME(void *Ptr, int64_t Val, int32_t Size,            \
                              void *Base) {                                    \
    getInputGenRT().Heap->read<TY>(Ptr, Base, Size);                           \
  }                                                                            \
  void __inputgen_write_##NAME(void *Ptr, int64_t Val, int32_t Size,           \
                               void *Base) {                                   \
    getInputGenRT().Heap->write<TY>((TY *)Ptr, (TY)Val, Size);                 \
  }                                                                            \
  void __inputgen_read_write_##NAME(void *Ptr, int64_t Val, int32_t Size,      \
                                    void *Base) {                              \
    abort();                                                                   \
  }                                                                            \
  void __record_read_##NAME(void *Ptr, int64_t Val, int32_t Size,              \
                            void *Base) {                                      \
    getInputGenRT().Heap->read<TY>(Ptr, Base, Size);                           \
  }                                                                            \
  void __record_write_##NAME(void *Ptr, int64_t Val, int32_t Size,             \
                             void *Base) {                                     \
    getInputGenRT().Heap->write<TY>((TY *)Ptr, (TY)Val, Size);                 \
  }                                                                            \
  void __record_read_write_##NAME(void *Ptr, int64_t Val, int32_t Size,        \
                                  void *Base) {                                \
    abort();                                                                   \
  }

RW(bool, i1)
RW(char, i8)
RW(short, i16)
RW(int32_t, i32)
RW(int64_t, i64)
RW(float, float)
RW(double, double)
RW(void *, ptr)
#undef RW

#define ARG(TY, NAME)                                                          \
  TY __inputgen_arg_##NAME(TY Arg) {                                           \
    int32_t ObjIdx = -1;                                                       \
    getInputGenRT().Args.push_back(                                            \
        toArgTy<TY>(getInputGenRT().getNewValue<TY>(&ObjIdx), ObjIdx));        \
    return fromArgTy<TY>(getInputGenRT().Args.back());                         \
  }

ARG(bool, i1)
ARG(char, i8)
ARG(short, i16)
ARG(int32_t, i32)
ARG(int64_t, i64)
ARG(float, float)
ARG(double, double)
ARG(void *, ptr)
#undef ARG

void free(void *) {}
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Wrong usage." << std::endl;
    return 1;
  }

  const char *OutputDir = argv[1];
  int Start = std::stoi(argv[2]);
  int End = std::stoi(argv[3]);
  const char *FuncName = argv[4];

  VERBOSE = (bool)getenv("VERBOSE");

  int Size = End - Start;
  if (Size <= 0)
    return 1;

  std::cout << "Will generate " << Size << " inputs." << std::endl;

  void *Handle = dlopen(NULL, RTLD_NOW);
  if (!Handle) {
    std::cout << "Could not dyn load binary" << std::endl;
    std::cout << dlerror() << std::endl;
    return 11;
  }
  typedef void (*EntryFnType)(int, char **);
  EntryFnType EntryFn = (EntryFnType)dlsym(
      Handle, (std::string("__inputgen_entry") + FuncName).c_str());

  if (!EntryFn) {
    std::cout << "Function " << FuncName << " not found in binary."
              << std::endl;
    return 12;
  }

  // #pragma omp parallel for schedule(dynamic, 5)
  for (int I = Start; I < End; I++) {
    InputGenRTTy LocalInputGenRT(argv[0], OutputDir, I);
    InputGenRT = &LocalInputGenRT;
    EntryFn(argc, argv);
  }

  dlclose(Handle);

  return 0;
}
