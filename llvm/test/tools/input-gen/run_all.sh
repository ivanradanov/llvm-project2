#!/bin/bash
set -e

for func in $(cat "$(1)/available_functions"); do
  gen="$(1)/input-gen.module.generate.a.out"
  "$gen" "$(1)"  0 2 "$func" 2> "${func}.err.out" > "${func}.log.out"

  run="$(1)/input-gen.module.run.a.out"
  for i in "$gen".input."$func".*.bin; do
    "$run" "${i}" "$func" 2> "${i}".err.out > "${i}".log.out
  done;

done;
