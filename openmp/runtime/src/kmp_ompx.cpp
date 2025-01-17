#include "kmp.h"

extern "C" {
#define _TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C(NAME, VALUE) \
  int ompx_##NAME(int Dim) { return VALUE; }
_TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C(thread_id,
                                      __kmp_get_ancestor_thread_num(__kmp_entry_gtid(),
                                                                    Dim + 1))
_TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C(block_dim,
                                      __kmp_get_team_size(__kmp_entry_gtid(),
                                                          Dim + 1))
_TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C(block_id, 0)
_TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C(grid_dim, 1)
#undef _TGT_KERNEL_LANGUAGE_HOST_IMPL_GRID_C

}
