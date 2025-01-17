! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - | FileCheck %s

module mod1
contains
  subroutine sub1()
    use omp_lib
    integer :: tmp
    tmp = ompx_thread_id(0)
    tmp = ompx_block_id(0)
    tmp = ompx_block_dim(0)
    tmp = ompx_grid_dim(0)
  end subroutine sub1

  subroutine host()

    !$omp target teams ompx_bare num_teams(1, 2, 3) thread_limit(4, 5, 6)
    call sub1()
    !$omp end target teams

    ! TODO we would like to have this:
    !!$omp target teams parallel num_teams(1, 2, 3) num_threads(4, 5, 6)
    !call sub1()
    !!$omp end target teams parallel

  end subroutine host
end module

! CHECK:    omp.target thread_limit(%c4_i32, %c5_i32, %c6_i32 : i32, i32, i32) num_teams(%c1_i32_0, %c2_i32_1, %c3_i32_2 : i32, i32, i32 to %c1_i32_0, %c2_i32_1, %c3_i32_2 : i32, i32, i32)
