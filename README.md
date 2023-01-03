# Priority-based GPU Scheduler for ML Tasks

### Usage
- Populate a CSV with the following fields: Directory, Executable, Filename, Epochs, Priority.
    - Directory: This will be the working directory for execution. Possibly different for each file.
    - Executable: Python executable. Can be of a virtual environment.
    - The file should support these arguments: `epochs` and `id`.
    - Priority: Lower number implies higher priority.
- Run `python scheduler.py` with appropriate arguments
    - `--csv`: Path to CSV file
    - `--num`: Number of tasks to be picked from the file in a single go
    - `--header`: Row number of header of CSV. Write `None` if there is no header
    - `--mps`: Specify whether to use Nvidia MPS (Multi-Process Service)
    - `--thread_percentage`: Limit on per-process percentage of threads. Supported via an MPS option.
- Suppose there are 50 tasks in the CSV, and we take 20 tasks (set by `--num`) in a single go. The scheduler sequentially takes batches of 20 (2 times, and a batch of 10 once), and runs the commands.
- To run with default tasks: Copy `commands.csv` to `commands_1.csv` in `scheduler_files/`. Run `python scheduler.py`.

### Working and Features
- Implementation:
    - Given a batch of commands, training is done (sequentially) for a limited number of epochs (default: 2) for knowing per-process peak memory usage.
    - Then, the batch is trained in parallel. Number of processes running parallelly is limited by GPU memory. If enough GPU memory is not available, the scheduler sleep-waits (function `wait_until`).
- Sorting by priority is done within a batch to prevent [starvation](https://en.wikipedia.org/wiki/Starvation_(computer_science)).
- GPU utilization is quite high, compared to sequential training. Reaches close to 100%.
- Effect of MPS is visible on smaller loads (on larger ones, parallelization has a greater effect).

### Assumptions
- In the main function, `available_memory` is computed only once per iteration.
    - It is assumed that no process, other than those started by this script, runs on the GPU. No process that consumes a lot of GPU memory, at least. Other processes are fine, as a `buffer` can be set in the script (default is 0.5GB).
    - Dynamically checking for available memory won't always work, as the processes increase memory usage *gradually*.
- Observed peak memory mismatch in `get_peak_mems()` vs `run_processes()` : `code2.py` (1947 vs 2005 MiB). Again, good to keep buffer.
    - Could also add per process small buffers in `peak_mems[]`, if number of processes is not very large.