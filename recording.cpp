#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <map>
#include <stdio.h>
#include <type_traits>
#include <vector>

template <typename Ty = void *> static Ty advance(void *Ptr, uint64_t Bytes) {
  return reinterpret_cast<Ty>(reinterpret_cast<char *>(Ptr) + Bytes);
}

extern "C" {

static std::map<void *, size_t> *ObjMap;
static int32_t RecordingLevel = 0;
static void *Buffer, *InitialBuffer;
static const size_t BufferSize = 6L * 1024 * 1024 * 1024;
static const size_t TrackingOffset = BufferSize / 3;
static const size_t ShadowOffset = TrackingOffset + BufferSize / 3;
static const size_t UserSize = TrackingOffset;

static void *(*real_malloc)(size_t) = NULL;
static void (*real_free)(void *) = NULL;

static void mtrace_init(void) {
  real_malloc = (decltype(real_malloc))dlsym(RTLD_NEXT, "malloc");
  real_free = (decltype(real_free))dlsym(RTLD_NEXT, "free");
  if (!real_malloc || !real_free)
    exit(2);
  InitialBuffer = Buffer = real_malloc(BufferSize);
  memset(advance(Buffer, UserSize), 0, BufferSize - UserSize);
  ++RecordingLevel;
  ObjMap = new std::map<void *, size_t>;
  --RecordingLevel;
}

void *malloc(size_t Size) {
  if (!real_malloc)
    mtrace_init();

  void *P = NULL;
  if (RecordingLevel) {
    P = real_malloc(Size);
  } else {
    P = Buffer;
    Size += Size % 16;
    Buffer = advance(Buffer, Size);
    ++RecordingLevel;
    (*ObjMap)[P] = Size;
    --RecordingLevel;
  }
  // fprintf(stderr, "malloc(%zu): %p [%i]\n", Size, P, RecordingLevel);
  return P;
}
void free(void *P) {
  // fprintf(stderr, "free(%p)\n", P);
  auto It = ObjMap->find(P);
  if (It == ObjMap->end()) {
    real_free(P);
    return;
  }
  if (Buffer == advance(P, It->second))
    Buffer = advance(Buffer, -It->second);
  ObjMap->erase(It);
}

void __record_version_mismatch_check_v1() {}

void __record_init() {
  printf("Init: %p : %p : %p : %p\n", InitialBuffer,
         advance(InitialBuffer, TrackingOffset),
         advance(InitialBuffer, ShadowOffset),
         advance(InitialBuffer, BufferSize));
  srand(time(NULL));
}

void __record_deinit() {
  printf("Deinit: %zu\n", ObjMap->size());
  for (auto &It : *ObjMap) {
    printf("%p : %p [%zu]\n", It.first, advance(It.first, It.second),
           It.second);
  }
  printf("Used %lu bytes\n", (char *)Buffer - (char *)InitialBuffer);
  uint32_t *TPtr = advance<uint32_t *>(InitialBuffer, TrackingOffset);
  for (size_t I = 0; I < TrackingOffset; I += sizeof(*TPtr)) {
    if (*advance<uint32_t *>(TPtr, I) != 17)
      continue;
    void *P = advance(TPtr, I - TrackingOffset);
    auto It = ObjMap->lower_bound(P);
    if (It == ObjMap->end() || It->first > P) {
      --It;
    }
    if (It->first <= P && advance(It->first, It->second) > P) {
      printf("Got %p at %zu: Obj(%zu): [", P, I, It->second);
      uint32_t *ItTPtr = advance<uint32_t *>(It->first, TrackingOffset);
      uint32_t *ItSPtr = advance<uint32_t *>(It->first, ShadowOffset);
      //      for (int J = 0; J < It->second; J += sizeof(*TPtr))
      //        if (*advance<uint32_t *>(ItTPtr, J) == 17)
      //          printf("%d", *advance<uint32_t *>(ItSPtr, J));
      //        else
      //          printf("-");
      printf("]\n");
      I += It->second - sizeof(*TPtr);
      continue;
    }
    printf("Got %p at %zu: Unknown\n", P, I);
  }
}

void __record_push() { ++RecordingLevel; }

void __record_pop() { --RecordingLevel; }

void mark(void *Ptr, int32_t Size) {
  if (Ptr < InitialBuffer || Ptr >= advance(InitialBuffer, UserSize))
    return;
  uint32_t *TPtr = advance<uint32_t *>(Ptr, TrackingOffset);
  for (int I = 0; I < Size; I += sizeof(*TPtr)) {
    if (!TPtr[I]) {
      *advance<uint32_t *>(TPtr, I) = 1;
    }
  }
}

void checkAndRemember(void *Ptr, int32_t Size) {
  if (Ptr < InitialBuffer || Ptr >= advance(InitialBuffer, UserSize))
    return;
  uint32_t *TPtr = advance<uint32_t *>(Ptr, TrackingOffset);
  uint32_t *SPtr = advance<uint32_t *>(Ptr, ShadowOffset);
  for (int I = 0; I < Size; I += sizeof(*TPtr)) {
    if (TPtr[I])
      continue;
    *advance<uint32_t *>(TPtr, I) = 17;
    *advance<uint32_t *>(SPtr, I) = *advance<uint32_t *>(Ptr, I);
  }
}

#define RW(TY, NAME)                                                           \
  __attribute__((always_inline)) void __record_read_##NAME(                    \
      void *Ptr, int64_t Val, int32_t Size, void *Base) {                      \
    checkAndRemember(Ptr, Size);                                               \
  }                                                                            \
  __attribute__((always_inline)) void __record_write_##NAME(                   \
      void *Ptr, int64_t Val, int32_t Size, void *Base) {                      \
    mark(Ptr, Size);                                                           \
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
  TY __record_arg_##NAME(TY Arg) { return Arg; }

ARG(bool, i1)
ARG(char, i8)
ARG(short, 16)
ARG(int32_t, i32)
ARG(int64_t, i64)
ARG(float, float)
ARG(double, double)
ARG(void *, ptr)
#undef ARG
}
