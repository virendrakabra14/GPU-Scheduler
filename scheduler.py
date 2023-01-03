import subprocess as sp
import argparse
import sys
import os
import time
import shlex
# import signal     # pause-resume, but GPU memory is still used by paused processes
import pathlib
from scheduler_files.utils import *

def modify_process_list(peak_mems):
    """
    Remove terminated processes from process list,
    and adds up their peak memories (NOTE: pre-computed)
    to available_memory
    """
    global procs_with_indices
    global available_memory

    modified = False
    indices_to_keep = []
    for i in range(len(procs_with_indices)):        # no lock needed, as this is the only one
                                                    # handling this list
        if procs_with_indices[i][0].poll() is not None:
            modified = True
            available_memory += peak_mems[procs_with_indices[i][1]]
            print(f"id {procs_with_indices[i][1]}, return-code {procs_with_indices[i][0].returncode}, ", end="")
        else:
            indices_to_keep.append(i)

    # https://stackoverflow.com/questions/1207406
    # list-slicing [:] prevents copying everything
    # can also use filterfalse from itertools
    procs_with_indices[:] = [procs_with_indices[i] for i in indices_to_keep]
    return modified


def get_sorted_data(args, remove_from_csv):
    """
    Returns top args.num rows of args.csv,
    sorted by 'Priority'
    """
    df_files = top_n_lines_csv(filename=args.csv, n=args.num, header=args.header, remove_from_csv=remove_from_csv)
    df_files['Priority'] = df_files['Priority'].astype(int)         # Priority must be integral
    df_files['Executable'] = df_files['Executable'].astype(str)     # Executable must be string (default is int if all are -1)
    df_files.sort_values(by='Priority', inplace=True)               # smaller value, higher priority
    df_files.reset_index(inplace=True)
    return df_files


def get_peak_mems(data, gpu_script_path, start_counter, num_epochs=2, sleep_period=1, tmp_dir='./tmp', log_dir='./logs', conversion_factors=None):
    """
    Initial run: train for a few epochs to get per-process/file peak memory usage stats.
    This is sequential (to avoid GPU overload).
    Assumption: There is enough GPU memory to train each process.
                Can't wait for GPU to be free, coz then we've the same 
                problem that we're trying to solve!

    sleep_period    : seconds; reduce to get finer granularity
    """

    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    peak_mems = []

    if conversion_factors is None:
        conversion_factors = get_conversion_factors()

    for i in range(data.shape[0]):

        # PIDs can be re-used, so need separate mem-usage-file for each train_file
        # storing data for all processes in single file might give incorrect results
        # (We rewrite same file. Since this is sequential, it's correct.)

        executable = sys.executable if data['Executable'][i]=='-1' else data['Executable'][i]
        filename = data['Filename'][i]

        out_file = f'{tmp_dir}/tmp_output.txt'          # NOTE: this will be first removed by the script
        bash_command = shlex.split(f'bash {gpu_script_path} {out_file} {sleep_period}')
        bash_p = sp.Popen(bash_command, cwd='.')        # tmp, logs are relative to current directory

        command = shlex.split(f"{executable} {filename} --epochs={num_epochs} --id={start_counter+i}")
        p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, cwd=data['Directory'][i])
        pid_p = p.pid
        # p.wait()
        out, err = p.communicate()
        out = out.decode('utf-8')
        err = err.decode('utf-8')
        print(f"return-code for {filename} (priority {data['Priority'][i]}) {start_counter+i}:", p.returncode)
        if p.returncode>0:
            print('out:', out)
            print('err:', err)

        time.sleep(sleep_period)
        bash_p.kill()               # bash process killed

        with open(out_file, 'r') as tmp_file:
            expected_useful_mem_data = False        # expected mem data for this process
                                                    # in the current line
            peak_mem = 0
            for line in tmp_file:
                line_list = [x.strip() for x in line.split(':')]
                if 'Process ID' in line_list:
                    expected_useful_mem_data = (int(line_list[1])==pid_p)
                else:
                    if expected_useful_mem_data:
                        unit = line_list[1].split()[1].lower()
                        peak_mem = max(peak_mem, conversion_factors[unit]*int(line_list[1].split()[0]))
                        expected_useful_mem_data = False        # reset
            peak_mems.append(peak_mem)

    return peak_mems


def run_processes(data, peak_mems, start_counter, buffer=2**29, log_dir='./logs'):
    """
    Run as many processes as possible, in parallel.
    Limited by GPU memory.
    Assumption: No other entity is running any _new_ processes on GPU.
                Some reasoning in below comments.

    buffer (bytes)  : GPU memory buffer. Default: 0.5GB
    """
    global available_memory
    global procs_with_indices

    procs = []

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    wait_kwargs = {'peak_mems': peak_mems}

    for i in range(data.shape[0]):
        
        # wait until sufficient memory is available
        while available_memory < peak_mems[i]+buffer:           # see main: IMPORTANT comments on available_memory
            wait_until(modify_process_list, sleep_period=1.0, **wait_kwargs)
                                                                # if needed, modify_process_list will modify available_memory

        available_memory -= peak_mems[i]

        executable = sys.executable if data['Executable'][i]=='-1' else data['Executable'][i]
        filename = data['Filename'][i]
        num_epochs = data['Epochs'][i]

        command = shlex.split(f"{executable} {filename} --epochs={num_epochs} --id={start_counter+i}")
        p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, cwd=data['Directory'][i])

        procs.append(p)
        procs_with_indices.append([p,i])

    for i in range(len(procs)):
        out, err = procs[i].communicate()      # if not terminated, waits
        print(f"return-code for {data['Filename'][i]} (priority {data['Priority'][i]}) {start_counter+i}:", procs[i].returncode)
        print(out.decode('utf-8'))
        print(err.decode('utf-8'))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', help='Path to CSV file', type=str, required=False)
    parser.add_argument('-n', '--num', help='Number of tasks to be picked from the file in a single go', type=int, required=False)
    parser.add_argument('--header', help='Row number of header of CSV. Write \'None\' (without quotes) if there is no header', type=str, required=False)
    parser.add_argument('-m', '--mps', help='Specify whether to use MPS', type=bool, required=False)
    parser.add_argument('--threads_percentage', help='Limit on per-process percentage of threads. Supported via an MPS option', type=int, required=False)
    args = parser.parse_args()

    scheduler_files_dir = './scheduler_files'
    default_args = {
        'csv': os.path.join(scheduler_files_dir, 'commands_1.csv'),
        'num': 20,
        'header': '0',
        'mps': True
    }
    args = init_args(args, default_args)       # fill defaults if None

    if args.mps:
        mps(action='start')
        if args.threads_percentage is not None:
            args.threads_percentage = max(0, min(100, args.threads_percentage))     # keep %age in bounds
            env_var = 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'
            previous_value = None
            if env_var in os.environ:
                previous_value = os.environ[env_var]
            os.environ[env_var] = str(args.threads_percentage)
            print("env_value:", os.environ[env_var])
    
    counter_file = os.path.join(scheduler_files_dir, 'counter.txt')
    start_counter = None
    with open(counter_file, 'r') as f:
        start_counter = int(f.read().strip())

    conversion_factors = get_conversion_factors()

    while True:

        data = get_sorted_data(args, remove_from_csv=True)

        # if there are no tasks, end scheduler
        if data.shape[0]==0:
            break

        print("Number of files:", data.shape[0])

        peak_mems = get_peak_mems(data=data, gpu_script_path=os.path.join(scheduler_files_dir, 'gpu_script.sh'), start_counter=start_counter)
                                                        # doesn't rely on output from train files anymore
        print("peak_mems:", peak_mems)
        # nvidia-smi (converted MiB to bytes): 1907359744, 2041577472

        # rm *.pt [or, continue training them: supported (only) by train_n_epochs]

        #####

        available_memory = get_gpu_memory(conversion_factors)
        # only computed once [assumption: no process on GPU, other than those started by this script]
        # dynamically checking won't always work, as the processes increase mem usage *gradually*

        procs_with_indices = []     # this will be modified
                                    # so, can use procs[] for final printing

        run_processes(data=data, peak_mems=peak_mems, start_counter=start_counter)

        start_counter += data.shape[0]          # increment start counter for later use
    
    if args.mps:
        mps(action='stop')
        if args.threads_percentage is not None:
            if previous_value is not None:
                os.environ[env_var] = previous_value
            else:
                del os.environ[env_var]

    with open (counter_file, 'w') as f:
        f.write(str(start_counter))