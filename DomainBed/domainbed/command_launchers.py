# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
import re
import submitit
import tqdm

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def submitit_launcher(commands):
    """ Launch commands on a slurm cluster using submitit. """

    def get_values_from_commands_str(commands, value):
        values = [c[c.index(f"--{value}") + 2:].split("--")[0].split(" ")[1:] for c in commands]
        values_flat = list(set([v for sublist in values for v in sublist if len(v) > 1]))
        print(values_flat)
        return values_flat

    alg_names = get_values_from_commands_str(commands, "algorithm")
    dataset_names = get_values_from_commands_str(commands, "dataset")
    job_name = f"{'-'.join(alg_names)}_{'-'.join(dataset_names)}"

    # Initialize Submitter (hardcoded slurm parameters: TODO: change for your cluster!)
    executor = submitit.AutoExecutor(folder='/home/s1668298/slurm_logs',  slurm_max_num_timeout=30)
    executor.update_parameters(
        name=job_name,
        timeout_min=300,
        slurm_partition='PGR-Standard',
        slurm_exclude="damnii10",
        tasks_per_node=1,       # one task per GPU
        nodes=1,
        gpus_per_node=1,
        cpus_per_task=8
    )

    n_jobs = len(commands)
    print(f'Submitting {n_jobs} jobs.')

    # Submit gridjobs as arrays to avoid overloading the scheduler.
    jobs = []
    with tqdm.tqdm(total=n_jobs) as progress_bar:
        with executor.batch():
            for cmd in commands:
                # Submit each grid-job
                cmd_as_list = re.split(r"\s+", cmd)
                func = submitit.helpers.CommandFunction(cmd_as_list, verbose=True)
                job = executor.submit(func)
                jobs.append(job)
                progress_bar.update(1)

    print("Finished scheduling!")


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'submitit': submitit_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
