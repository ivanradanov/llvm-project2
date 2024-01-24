
! subroutine coexecute_assign(a, c, n)
!   integer :: n
!   real, dimension(n, n) :: a, c
!   !$omp target
!   !$omp teams
!   !$omp coexecute
!     c = a
!   !$omp end coexecute
!   !$omp end teams
!   !$omp end target
! end subroutine coexecute_assign

subroutine coexecute_add2(a, b, c, n)
  integer :: n
  real, dimension(n, n) :: a, b, c
  !$omp target
  !$omp teams
  !$omp coexecute
    c = a + b
  !$omp end coexecute
  !$omp end teams
  !$omp end target
end subroutine coexecute_add2
