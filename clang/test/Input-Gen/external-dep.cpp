// RUN: inputgen-minimize %s Inputs/external_func.cpp

#include "Inputs/external_func.h"

__attribute__((inputgen_entry))
int foo(int a, int b) {
    return external_foo(a, b);
}

int main() {
    foo(4, 5);
}
