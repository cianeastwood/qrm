"""
Submitit batch jobs.
"""

import os
import re
import argparse
import submitit
import tqdm
from pathlib import Path
import uuid
from submitit.helpers import RsyncSnapshot


def create_temp_dir(base_dir):
    random_dirname = str(uuid.uuid4())[:10]
    snapshot_dir = os.path.join(base_dir, random_dirname)
    os.makedirs(snapshot_dir, exist_ok=True)
    return Path(snapshot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands-file', '-c', type=str, required=True)
    parser.add_argument('--jobname', '-j', type=str)
    parser.add_argument('--partition', '-p', type=str, default='PGR-Standard')
    parser.add_argument('--root-dir', '-rd', type=str, default='./',
                        help="Root directory to copy/snapshot.")
    parser.add_argument('--temp-dir', '-td', type=str, default='/home/s1668298/tmp',
                        help="Temp directory for snapshot")
    parser.add_argument('--slurm-dir', '-sd', type=str, default='/home/s1668298/slurm_logs',
                        help="Directory for slurm logs.")
    parser.add_argument('--slurm-exclude', '-se', type=str, default="damnii10",
                        help="Names of the cluster nodes to be excluded.")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--cpus-per-task', '-nc', type=int, default=8)
    parser.add_argument('--gpus-per-node', '-ng', type=int, default=1)
    parser.add_argument('--mem-per-gpu', '-mem', type=int)
    parser.add_argument('--timeout_min', '-to', type=int, default=4000)
    parser.add_argument('--comment', type=str, help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument('--signal-delay-s', type=int, default=120,
                        help='Delay between the kill signal and the actual kill of the slurm job.')

    # Extract info
    args = parser.parse_args()
    argparse_defaults = {k: parser.get_default(k) for k in vars(args).keys()}
    if args.jobname is None:
        args.jobname = args.commands_file.split("/")[-1].split(".")[0]
    print(args.commands_file, args.jobname)

    # Load commands from rgs.commands_file file with one command per line
    f = open(args.commands_file, "r")
    cmds = f.read().split("\n")
    n_jobs = len(cmds)
    print(f'Submitting {n_jobs} jobs.')

    # Prep kwargs
    kwargs = {}
    if args.slurm_exclude is not None:
        kwargs['slurm_exclude'] = args.slurm_exclude
    if args.comment is not None:
        kwargs['slurm_comment'] = args.comment
    if args.mem_per_gpu is not None:
        kwargs['mem_gb'] = args.mem_per_gpu * args.gpus_per_node

    # Initialize Submitter
    executor = submitit.AutoExecutor(folder=args.slurm_dir, slurm_max_num_timeout=30)
    executor.update_parameters(
        name=args.jobname,
        timeout_min=args.timeout_min,
        slurm_partition=args.partition,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=args.gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        slurm_signal_delay_s=args.signal_delay_s,
        **kwargs
    )

    # Create temp directory with a "snapshot" of current codebase (requeue interrupted jobs while editing codebase)
    snapshot_dir = create_temp_dir(args.temp_dir)
    print("Snapshot dir is: ", snapshot_dir)

    # Submit jobs
    jobs = []
    with RsyncSnapshot(snapshot_dir=snapshot_dir, root_dir=args.root_dir):
        with tqdm.tqdm(total=n_jobs) as progress_bar:
            with executor.batch():
                for cmd in cmds:
                    # Submit job as a command string directly (as batch/grid array)
                    cmd_as_list = re.split(r"\s+", cmd)
                    func = submitit.helpers.CommandFunction(cmd_as_list, verbose=True)
                    job = executor.submit(func)
                    jobs.append(job)
                    progress_bar.update(1)

    print("Finished scheduling!")
