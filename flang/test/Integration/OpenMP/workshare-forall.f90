!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!XFAIL: *
!RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix FIR

subroutine sb1(a, x, y, z)
  integer :: x(:,:)
  !$omp parallel
  !$omp workshare
  forall (i = 1:10)
    x(i,:) = x(i,:) + 1
  end forall
  !$omp end workshare
  !$omp end parallel
end subroutine

! HLFIR:  func.func @_QPsb1
! HLFIR:    omp.parallel {
! HLFIR:      omp.workshare {
! HLFIR:        hlfir.forall

! TODO This is currently unimplemented, thus we get no `omp.wsloop` in the FIR
! output.

! FIR:  func.func @_QPsb1
! FIR:    omp.parallel {
! FIR:      omp.wsloop {
