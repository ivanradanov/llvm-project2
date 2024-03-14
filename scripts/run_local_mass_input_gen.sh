#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

### Default values. These can be overridden in the user config file below
DATASET="/p/vast1/LExperts/ComPile-Public-V2"
OUTDIR="/l/ssd/$USER/compile-input-gen-out/"
LLVM_INSTALL_DIR="/usr/WS1/$USER/opt/input-gen-release"
JUGDIR="$SCRIPT_DIR/jugfile.jugdata"
OUTDIR="/p/vast1/doerfert/llvm-project/compile-input-gen-out/"
LLVM_INSTALL_DIR="/p/vast1/doerfert/install"
INPUT_GEN_NUM=${INPUT_GEN_NUM:=1}
START=${START:=0}
END=${END:=7000}
NUM_CPU=${NUM_CPU:="$(nproc --all)"}
INPUT_GEN_RUNTIME="${INPUT_GEN_RUNTIME:=$(readlink -f "$SCRIPT_DIR/../input-gen-runtimes/rt-input-gen.cpp")}"
INPUT_RUN_RUNTIME="${INPUT_RUN_RUNTIME:=$(readlink -f "$SCRIPT_DIR/../input-gen-runtimes/rt-run.cpp")}"
### Default values

# User provided configuration
USER_CONFIG="$SCRIPT_DIR/configuration.sh"
if [ -f "$USER_CONFIG" ]; then
    . "$USER_CONFIG"
fi

if [ "$SINGLE" != "" ]; then
    START="$SINGLE"
    END="$(("$SINGLE" + 1))"
fi

SCRIPT="$SCRIPT_DIR/mass_input_gen.py"

if [ "$JUG" == "run" ]; then
    JUG_RUN="jug-execute --jugdir $JUGDIR --will-cite"
    SCRIPT="$SCRIPT_DIR/jugfile.py"
    DASHDASH=--
elif [ "$JUG" == "status" ]; then
    JUG_RUN="jug status --jugdir $JUGDIR"
    SCRIPT="$SCRIPT_DIR/jugfile.py"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --get-jug-results"
    DASHDASH=--
elif [ "$JUG" == "results" ]; then
    export JUGDIR
    JUG_RUN=
    SCRIPT="$SCRIPT_DIR/print_mig_jug_results.py"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --get-jug-results"
elif [ "$JUG" != "" ]; then
    echo Invalid JUG option 1>&2
    exit 1
fi

if [ "$ENABLE_TIMING" != "" ]; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --timing"
fi

if [ "$NOCLEANUP" == "" ]; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --cleanup"
fi

if [ "$INPUT_GEN_DISABLE_BRANCH_HINTS" == "" ]; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --branch-hints"
fi

if [ "$INPUT_GEN_DISABLE_FUNC_PTR" !=  "" ]; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --disable-fp-handling"
fi

. "$SCRIPT_DIR/enable.sh" "$LLVM_INSTALL_DIR"
PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

if [ "$LANGUAGE" == "" ]; then
    echo Specify LANGUAGE
    exit 1
fi

function run() {
    $JUG_RUN "$SCRIPT" $DASHDASH \
        --dataset "$DATASET" \
        --language "$LANGUAGE" \
        --outdir "$OUTDIR" \
        --start "$START" \
        --end "$END" \
        --precompile-rts \
        --input-gen-runtime "$INPUT_GEN_RUNTIME" \
        --input-run-runtime "$INPUT_RUN_RUNTIME" \
        --input-gen-num "$INPUT_GEN_NUM" \
        --input-gen-timeout 5 \
        --input-run-timeout 5 \
        --num-procs="$NUM_CPU" \
        --coverage-statistics \
        --coverage-runtime "$(readlink -f "$LLVM_INSTALL_DIR/lib/clang/19/lib/x86_64-unknown-linux-gnu/libclang_rt.profile.a")" \
        $ADDITIONAL_FLAGS
}
run
