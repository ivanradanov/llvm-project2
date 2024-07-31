//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lower omp workshare construct.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Passes.h"

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKSHARE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workshare"

namespace {
class LowerWorksharePass
    : public flangomp::impl::LowerWorkshareBase<LowerWorksharePass> {
public:
  void runOnOperation() override {}
};
} // namespace
