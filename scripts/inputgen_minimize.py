#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import re

embedded_input_name = '__inputgen_embedded_input'

replay_runtime = os.path.dirname(os.path.realpath(__file__)) + '/../compiler-rt/lib/inputgen/replay.cpp'
replay_common = os.path.dirname(os.path.realpath(__file__)) + '/../compiler-rt/lib/inputgen/common.hpp'

def get_minimized_func(f, func_name):
    stdout = subprocess.PIPE
    args = [
        'inputgen-minimize',
        '--ast-print',
        '--ast-dump-filter',
        func_name,
        f,
    ]

    proc = subprocess.Popen(
        args,
        stdout=stdout,
        text=True)

    out, err = proc.communicate()

    if proc.returncode != 0:
        print(args)
        print('inputgen-minimize failed')
        exit(1)

    # FIXME we are using a quick hack to just pripnt out the function we are
    # interesetd in using the ASTPrinter, which prints "Printing <func_name>"
    # before printing the function, remove this once we don't do that any more
    out = re.sub(r'^Printing .*:\n', '', out)
    return out

def get_c_array_for_file(f):

    stdout = subprocess.PIPE
    args = [
        'xxd',
         '-i',
        '-n',
        embedded_input_name,
         f,
    ]

    proc = subprocess.Popen(
        args,
        stdout=stdout,
        text=True)

    out, err = proc.communicate()

    if proc.returncode != 0:
        print(args)
        print('xxd failed')
        exit(1)
    return out

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source-file', required=True)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--func-name', required=True)
    parser.add_argument('--output-file', required=True)

    args = parser.parse_args()

    print(args)

    embedded_input = get_c_array_for_file(args.input_file)
    minimized_func = get_minimized_func(args.source_file, args.func_name)

    with open(replay_runtime, 'r') as f:
        rt = f.read()
    with open(replay_common, 'r') as f:
        rt_common = f.read()

    rt_split = rt.split('#include "common.hpp"')
    rt = rt_split[0] + rt_common + rt_split[1]

    rt_split = rt.split('// DO_NOT_REMOVE_INPUT_REPLAY_AFTER_INCLUDES')
    assert(len(rt_split) == 2)
    rt_pre = rt_split[0]
    rt_post = rt_split[1]

    with open(args.output_file, 'w') as f:

        print(rt_pre, file=f)
        print('\n\n// =========== RECORDED FUNCTION BEGIN ===========\n', file=f)
        print(minimized_func, file=f)
        print('\n// =========== RECORDED FUNCTION END ===========\n\n', file=f)
        print(embedded_input, file=f)
        print('#define INPUT_REPLAY_EMBEDDED_INPUT\n', file=f)
        print(rt_post, file=f)


if __name__ == '__main__':
    main()
