! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 6.0 Preview 2
! 12.5 coexecute Construct
! The coexecute construct must be a closely nested construct inside a teams construct.

subroutine coexecute()
  !$omp coexecute
  call f1()
  !$omp end coexecute
end subroutine coexecute
