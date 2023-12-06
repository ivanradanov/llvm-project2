!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

! TODO checks
!
subroutine coexecute_add(a, b, c, n)
  integer :: n
  real, dimension(n) :: a, b, c
  !$omp teams
  !$omp coexecute
    c = a + b
  !$omp end coexecute
  !$omp end teams
end subroutine coexecute_add

function coexecute_a(x, y, tmp, n) result(any_less)
  integer :: n
  logical :: any_less
  real, dimension(n, n) :: x, y, tmp
  !$omp teams
  !$omp coexecute
      y(1:n/2,1:n) = 1.0
      y = y + x
      tmp = a * matmul(x, y + 1.0)
      any_less = any(tmp(1:n/2,1:n/3) < 1.0)
  !$omp end coexecute
  !$omp end teams
end function coexecute_a
