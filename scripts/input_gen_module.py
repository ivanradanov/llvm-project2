#!/usr/bin/env python3

import subprocess
import argparse
import os
import sys
import time
import json

def add_option_args(parser):
    parser.add_argument('--input-gen-num', type=int, default=1)
    parser.add_argument('--input-gen-timeout', type=int, default=5)
    parser.add_argument('--input-run-timeout', type=int, default=5)
    # TODO Pass in an object files here so as not to compile the cpp every time
    parser.add_argument('--input-gen-runtime', default='./input-gen-runtimes/rt-input-gen.cpp')
    parser.add_argument('--input-run-runtime', default='./input-gen-runtimes/rt-run.cpp')

class Function:
    def __init__(self, name):
        self.name = name
        self.input_gen_executable = None
        self.input_run_executable = None
        self.inputs_dir = None
        self.inputs = []
        self.times = {}

    def run_all_inputs(self, timeout):
        for input in self.inputs:
            self.run_input(input, timeout)

    def run_input(self, input, timeout):
        print('Running executables for', self.input_run_executable)

        try:
            start_time = time.time()

            proc = subprocess.Popen(
                [
                    self.input_run_executable,
                    input,
                    self.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            out, err = proc.communicate(timeout=timeout)

            end_time = time.time()
            elapsed_time = end_time - start_time

            if proc.returncode != 0:
                print('Input run process failed (%i): %s' % (proc.returncode, input))
            else:
                if input not in self.times:
                    self.times[input] = []
                self.times[input].append(elapsed_time)

        except subprocess.CalledProcessError as e:
            print('Input run process failed: %s' % input)
        except subprocess.TimeoutExpired as e:
            print("Input run timed out! Terminating... %s" % input)
            proc.terminate()
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired as e:
                print("Termination timed out! Killing... %s" % input)
                proc.kill()
                proc.communicate()
                print("Killed.")

class InputGenModule:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def generate_inputs(self):

        os.makedirs(self.outdir, exist_ok=True)

        print('Generating inputgen executables for', self.input_module, 'in', self.outdir)

        try:
            subprocess.run([
                'input-gen',
                '--input-gen-runtime', self.input_gen_runtime,
                '--input-run-runtime', self.input_run_runtime,
                '--output-dir', self.outdir,
                self.input_module,
                '--compile-input-gen-executables',
            ], check=True)
        except Exception:
            print('Failed to instrument')

        available_functions_file_name = os.path.join(self.outdir, 'available_functions')
        with open(available_functions_file_name, 'r') as available_functions_file:
            self.functions = []

            for fname in available_functions_file.read().splitlines():
                func = Function(fname)
                self.functions.append(func)

                input_gen_executable = os.path.join(
                    self.outdir, 'input-gen.module.generate.a.out')
                input_run_executable = os.path.join(
                    self.outdir, 'input-gen.module.run.a.out')
                inputs_dir = os.path.join(self.outdir, 'input-gen.' + fname + '.inputs')

                if not os.path.isfile(input_gen_executable):
                    continue

                # Only attach these to the Function if they exist
                os.makedirs(inputs_dir, exist_ok=True)
                func.input_run_executable = input_run_executable
                func.input_gen_executable = input_gen_executable
                func.inputs_dir = inputs_dir

                print('Generating inputs for function @{}'.format(fname))

                try:
                    start = 0
                    end = self.input_gen_num
                    proc = subprocess.Popen(
                        [
                            input_gen_executable,
                            inputs_dir,
                            str(start), str(end),
                            fname,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

                    # TODO we could accidentally kill the input gen while it
                    # writes the results to the file, find a better way to
                    # orchestrate this.
                    #
                    # TODO also with the current implementation one of the input
                    # gens timing out would mean we lose even the completed
                    # ones.
                    #
                    # Perhaps multi-process input-gen into a tmp dir and move
                    # completed inputs into the correct dir only if we succeeded
                    out, err = proc.communicate(timeout=self.input_gen_timeout)

                    if proc.returncode != 0:
                        print('Input gen process failed: @{}'.format(fname))
                    else:
                        print('Input gen process succeeded: @{}'.format(fname))
                        # Populate the generated inputs
                        func.inputs = [
                            os.path.join(inputs_dir,
                                        '{}.input.{}.{}.bin'.format(os.path.basename(input_gen_executable), fname, str(i)))
                            for i in range(start, end)]
                        for inpt in func.inputs:
                            # If the input gen process exited successfully these
                            # _must_ be here
                            assert(os.path.isfile(inpt))

                except subprocess.CalledProcessError as e:
                    print('Input gen process failed: @{}'.format(fname))
                except subprocess.TimeoutExpired as e:
                    print('Input gen timed out! Terminating...: @{}'.format(fname))
                    proc.terminate()
                    try:
                        proc.communicate(timeout=1)
                    except subprocess.TimeoutExpired as e:
                        print("Termination timed out! Killing...")
                        proc.kill()
                        proc.communicate()
                        print("Killed.")

        available_functions_pickle_file_name = os.path.join(self.outdir, 'available_functions.json')
        with open(available_functions_pickle_file_name, 'w') as available_functions_pickle_file:
            available_functions_pickle_file.write(json.dumps(self.functions, default=vars))

    def run_all_inputs(self):
        for func in self.functions:
            func.run_all_inputs(self.input_run_timeout)

    def get_empty_statistics(self):
        stats = {
            'num_funcs': 0,
            'num_instrumented_funcs': 0,
            'num_input_generated_funcs': 0,
            'num_input_ran_funcs': 0,
        }
        return stats

    def get_statistics(self):
        stats = self.get_empty_statistics()
        self.update_statistics(stats)
        return stats

    def update_statistics(self, stats):
        for func in self.functions:
            stats['num_funcs'] += 1
            if func.input_gen_executable:
                stats['num_instrumented_funcs'] += 1
            if len(func.inputs) != 0:
                stats['num_input_generated_funcs'] += 1
            if len(func.times) != 0:
                stats['num_input_ran_funcs'] += 1

    def aggregate_statistics(self, stats, to_add):
        for k, v in stats.items():
            stats[k] += to_add[k]

    def add_statistics(self, a, b):
        c = self.get_empty_statistics()
        self.aggregate_statistics(c, a)
        self.aggregate_statistics(c, b)
        return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser('InputGenModule')

    parser.add_argument('--outdir', required=True)
    parser.add_argument('--input-module', required=True)

    add_option_args(parser)

    args = parser.parse_args()

    IGM = InputGenModule(**vars(args))
    IGM.generate_inputs()
    IGM.run_all_inputs()
    print(IGM.get_statistics())
