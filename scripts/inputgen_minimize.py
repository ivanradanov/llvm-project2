#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import re

embedded_input_name = '__inputgen_embedded_input'

replay_runtime = os.path.dirname(os.path.realpath(__file__)) + '/../compiler-rt/lib/inputgen/replay.cpp'
replay_common = os.path.dirname(os.path.realpath(__file__)) + '/../compiler-rt/lib/inputgen/common.hpp'

def get_minimized_func(f):
    stdout = subprocess.PIPE
    args = [
        'inputgen-minimize',
        *f,
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

def construct_minimized_file(output_file, rt, embedded_input, minimized_func):

    # FIXME Need to recursively preprocess local includes in the rt. currently
    # we only have common.hpp and we hardcode that.
    with open(replay_common, 'r') as f:
        rt_common = f.read()
    rt_split = rt.split('#include "common.hpp"')
    rt = rt_split[0] + rt_common + rt_split[1]

    rt_split = rt.split('// DO_NOT_REMOVE_INPUT_REPLAY_AFTER_INCLUDES')
    assert(len(rt_split) == 2)
    rt_pre = rt_split[0]
    rt_post = rt_split[1]

    print(rt_pre, file=output_file)
    print(minimized_func, file=output_file)
    print('\n', file=output_file)
    if embedded_input is not None:
        print('// =========== EMBEDDED INPUT BEGIN ===========\n', file=output_file)
        print(embedded_input, file=output_file)
        print('// =========== EMBEDDED INPUT END ===========\n', file=output_file)
        print('#define INPUT_REPLAY_EMBEDDED_INPUT\n', file=output_file)
    print('#define INPUT_REPLAY_EMBEDDED_SOURCE\n', file=output_file)
    print(rt_post, file=output_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--output-file', '-o')
    parser.add_argument('--embed-input-file')

    args = parser.parse_args()

    print(args)

    if args.embed_input_file is not None:
        embedded_input = get_c_array_for_file(args.embed_input_file)
    else:
        embedded_input = None

    minimized_func = get_minimized_func(args.filenames)

    with open(replay_runtime, 'r') as f:
        rt = f.read()

    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            construct_minimized_file(f, rt, embedded_input, minimized_func)
    else:
        construct_minimized_file(sys.stdout, rt, embedded_input, minimized_func)



if __name__ == '__main__':
    main()
