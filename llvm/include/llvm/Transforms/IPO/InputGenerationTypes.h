#ifndef LLVM_TRANSFORMS_IPO_INPUTGENERATIONTYPES_H_
#define LLVM_TRANSFORMS_IPO_INPUTGENERATIONTYPES_H_

#include <stdint.h>

namespace llvm {
class Value;
namespace inputgen {

struct BranchHint {
  enum KindTy {
    EQ = 1,
    NE,
    LT,
    GT,
    LE,
    GE,
    Invalid = 0,
  } Kind;
  bool Signed;
  Value *Val;
  uint64_t Frequency;
  int32_t Dominator;
};

} // namespace inputgen
} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INPUTGENERATIONTYPES_H_
