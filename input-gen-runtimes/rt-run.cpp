#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

static int VERBOSE = 0;

template <typename T> static char *ccast(T *Ptr) {
  return reinterpret_cast<char *>(Ptr);
}

template <typename T> static T readSingleEl(std::ifstream &Input) {
  T El;
  Input.read(ccast(&El), sizeof(El));
  return El;
}

// TODO

static std::vector<void *> GetObjectPtrs;
static unsigned GetObjectIdx = 0;

static std::vector<char *> Globals;
static size_t GlobalsIt = 0;

static std::mt19937 Gen;
static std::uniform_int_distribution<> Rand;
int rand() { return Rand(Gen); }

template <typename T> T getNewValue() {
  T V = rand() % 10;
  return V;
}

template <> void *getNewValue<void *>() {
  if (!(rand() % 50))
    return nullptr;
  return GetObjectPtrs[GetObjectIdx++];
}

extern "C" {
void __inputgen_global(int32_t NumGlobals, void *Global, void **ReplGlobal,
                       int32_t GlobalSize) {
  assert(Globals.size() > GlobalsIt);
  memcpy(Global, Globals[GlobalsIt], GlobalSize);
}

#define RW(TY, NAME)                                                           \
  TY __inputrun_get_##NAME() { return getNewValue<TY>(); }

RW(bool, i1)
RW(char, i8)
RW(short, i16)
RW(int32_t, i32)
RW(int64_t, i64)
RW(float, float)
RW(double, double)
RW(void *, ptr)

#undef RW

void free(void *) {}
}

int main(int argc, char **argv) {
  if (argc != 3 && argc != 2) {
    std::cerr << "Wrong usage." << std::endl;
    return 1;
  }

  char *InputName = argv[1];
  std::string FuncName = ("__inputrun_entry");
  if (argc == 3) {
    FuncName += "_";
    FuncName += argv[2];
  }

  VERBOSE = (bool)getenv("VERBOSE");

  printf("Replay %s\n", InputName);

  std::ifstream Input(InputName, std::ios::in | std::ios::binary);

  auto Seed = readSingleEl<uint32_t>(Input);
  Gen.seed(Seed);

  auto MemSize = readSingleEl<uint64_t>(Input);
  char *Memory = ccast(malloc(MemSize));
  if (VERBOSE)
    printf("MemSize %lu : %p\n", MemSize, (void *)Memory);

  std::map<uint64_t, char *> ObjMap;
  auto NumObjects = readSingleEl<uint32_t>(Input);
  if (VERBOSE)
    printf("NO %u\n", NumObjects);
  for (uint32_t I = 0; I < NumObjects; I++) {
    auto ObjIdx = readSingleEl<uint32_t>(Input);
    auto Offset = readSingleEl<uint64_t>(Input);
    if (VERBOSE)
      printf("O %u -> %lu\n", ObjIdx, Offset);

    assert(Offset <= MemSize);
    ObjMap[ObjIdx] = Memory + Offset;
  }

  auto NumGlobals = readSingleEl<uint32_t>(Input);
  for (uint32_t I = 0; I < NumGlobals; I++) {
    auto ObjIdx = readSingleEl<uint32_t>(Input);
    if (VERBOSE)
      printf("G %u -> %u\n", I, ObjIdx);
    assert(ObjMap.count(ObjIdx));
    Globals.push_back(ObjMap[ObjIdx]);
  }

  auto NumVals = readSingleEl<uint32_t>(Input);
  for (uint32_t I = 0; I < NumVals; ++I) {
    uint32_t ObjIdx = readSingleEl<uint32_t>(Input);
    ptrdiff_t Offset = readSingleEl<ptrdiff_t>(Input);
    uint32_t ContentOrIdxKind = readSingleEl<int32_t>(Input);
    uintptr_t ContentOrIdx = readSingleEl<uintptr_t>(Input);
    uint32_t Size = readSingleEl<uint32_t>(Input);
    if (VERBOSE)
      printf("ObjIdx %u Kind %i COI %lu\n", ObjIdx, ContentOrIdxKind,
             ContentOrIdx);
    assert(ObjMap.count(ObjIdx));
    char *Tgt = ObjMap[ObjIdx] + Offset;
    char *Src;
    if (ContentOrIdxKind == /* Idx */ 0) {
      assert(ObjMap.count(ContentOrIdx));
      ContentOrIdx = (uintptr_t)ObjMap[ContentOrIdx];
    } else if (ContentOrIdxKind == /* Content */ 1) {
    } else {
      printf("Error\n");
      exit(4);
    }
    Src = ccast(&ContentOrIdx);
    memcpy(Tgt, Src, Size);
  }

  auto NumArgs = readSingleEl<uint32_t>(Input);
  char *ArgsMemory = ccast(malloc(NumArgs * sizeof(uintptr_t)));
  if (VERBOSE)
    printf("Args %u : %p\n", NumArgs, (void *)ArgsMemory);
  for (uint32_t I = 0; I < NumArgs; ++I) {
    auto Content = readSingleEl<uintptr_t>(Input);
    auto ObjIdx = readSingleEl<int32_t>(Input);
    if (ObjIdx != -1) {
      assert(ObjMap.count(ObjIdx));
      Content = (uintptr_t)ObjMap[ObjIdx];
    }
    memcpy(ArgsMemory + I * sizeof(void *), &Content, sizeof(void *));
  }

  auto NumGetObjects = readSingleEl<uint32_t>(Input);
  for (uint32_t I = 0; I < NumGetObjects; ++I) {
    auto ObjIdx = readSingleEl<int32_t>(Input);
    assert(ObjMap.count(ObjIdx));
    GetObjectPtrs.push_back(ObjMap[ObjIdx]);
  }

  void *Handle = dlopen(NULL, RTLD_NOW);
  if (!Handle) {
    std::cout << "Could not dyn load binary" << std::endl;
    std::cout << dlerror() << std::endl;
    return 11;
  }
  typedef void (*EntryFnType)(char *);
  EntryFnType EntryFn = (EntryFnType)dlsym(Handle, FuncName.c_str());

  if (!EntryFn) {
    std::cout << "Function " << FuncName << " not found in binary."
              << std::endl;
    return 12;
  }

  printf("Run\n");
  EntryFn(ArgsMemory);

  dlclose(Handle);

  // TODO: we intercept free
  free(ArgsMemory);
  free(Memory);
}
