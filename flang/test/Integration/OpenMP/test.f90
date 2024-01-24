
program test
    implicit none

    integer, parameter :: N = 20480
    ! integer, parameter :: N = 16
    real :: a
    real, dimension(:, :), allocatable :: x
    real, dimension(:, :), allocatable :: y
    real, dimension(:, :), allocatable :: z

    allocate(x(N, N))
    allocate(y(N, N))
    allocate(z(N, N))

    x = (3, 1)
    y = (2, -1)
    z = 0

    write (*, '(A)') 'calling axpy'
    a = abs(coexecute_a(x, y, z, N))

    write (*,'(A,F4.2)') 'checksum ', a

    deallocate(x)
    deallocate(y)
    deallocate(z)

contains
function coexecute_a(x, y, z, n) result(sum_less)
  use omp_lib
  integer :: n
  real :: sum_less
  real, dimension(n, n) :: x, y, z
  double precision :: ostart, oend

  ostart = omp_get_wtime()
  !$omp target
  !$omp teams
  !$omp coexecute
      y(1:n/2,1:n) = 0.1
      y = sqrt(y * x) * x * y
      !z = matmul(x, y - 0.000002)
      z = y
  !$omp end coexecute
  !$omp end teams
  !$omp end target
  oend = omp_get_wtime()
  print *, 'Time: ', oend-ostart, 'seconds.'
  sum_less = sum(z(1:n/2,1:n/3) - 2) / ( n * n)
end function coexecute_a
end program test
