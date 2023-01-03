import pandas as pd
import time
import subprocess as sp
import shlex
import os

def top_n_lines_csv(filename, n, header=0, remove_from_csv=False):
    """
    Adapted from https://stackoverflow.com/a/73316287/17786040.
    Returns pd.Dataframe consisting of top `n` lines of the CSV file.
    Optionally, removes them from the file
    """
    df = pd.read_csv(filename, header=header)
    df_removed = df.head(n).copy()      # deepcopy, by default
    if remove_from_csv:
        df.drop(df.index[:n], inplace=True)
    df.to_csv(filename, index=False, header=False if header is None else True)
    return df_removed


def wait_until(func, timeout=3600.0, sleep_period=1.0, **kwargs):
    """
    https://stackoverflow.com/a/2785908/17786040
    Wait until `func` returns True, while sleeping
    intermittently for `sleep_period`, until `timeout`
    """
    max_limit = time.time() + timeout
    print("Waiting...    ", end="")
    while time.time() < max_limit:
        if func(**kwargs):
            print("DONE")
            return True
        time.sleep(sleep_period)
    return False


def init_args(args, default_args):
    """
    Initializes `args` with default values,
    if values are None
    """
    args.csv = default_args['csv'] if args.csv is None else args.csv
    args.num = default_args['num'] if args.num is None else args.num
    if args.header is None:
        args.header = int(default_args['header'])
    else:
        if args.header == 'None':
            args.header = None
        else:
            args.header = int(args.header)
    args.mps = default_args['mps'] if args.mps is None else args.mps
    return args


def run_command(command, print_output=False):
    """
    Run `command` as a subprocess.
    Optionally print output.
    Return process code.
    """
    p = sp.Popen(shlex.split(command))
    out, err = p.communicate()          # waits till completion
    if print_output:
        print(out)
        print(err)
    return p.returncode


def mps(action, bash_filepath=None, scheduler_files_dir=None, print_output=False):
    """
    Helper function to start/stop MPS daemon.
    Runs the relevant script.
    """
    if scheduler_files_dir is None:
        scheduler_files_dir = './scheduler_files'
    if bash_filepath is None:
        bash_filepath = os.path.join(scheduler_files_dir, f'{action}_mps.sh')
    command = f"sudo bash {bash_filepath}"
    return run_command(command=command, print_output=print_output)


def get_gpu_memory(conversion_factors=None):
    """
    Returns available GPU memory via `nvidia-smi`.
    """
    command = shlex.split('nvidia-smi --query-gpu=memory.free --format=csv')
    p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate()
    if err.decode('utf-8') != '':
        print("ERROR in get_gpu_memory:", err)
        return -1
    out = out.decode('utf-8')
    value = int(out.split('\n')[1].split()[0])
    units = out.split('\n')[1].split()[1].lower()
    if conversion_factors is None:
        conversion_factors = get_conversion_factors()
    if units not in conversion_factors:
        print("ERROR in get_gpu_memory: units absent from dict")
        return -1
    return value*conversion_factors[units]


def get_conversion_factors():
    """
    Returns dictionary of factors
    for conversion to bytes
    """
    conversion_factors = {
        'kib': 2**10,
        'mib': 2**20,
        'gib': 2**30,
    }
    return conversion_factors