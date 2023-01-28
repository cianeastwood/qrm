#!/usr/bin/env python3
"""Script for generating commands/jobs."""
import argparse

if __name__ == "__main__":
    # Flags
    parser = argparse.ArgumentParser(description='Generate commands for CMNIST experiments.')
    parser.add_argument('--data_dir', type=str, required=True, help="Absolute path to data directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Absolute path to output directory.")
    parser.add_argument('--exp_name', type=str, default="reproduce")
    args = parser.parse_args()

    # Base settings
    lr = 1e-4
    batch_size = 25000
    dropout_p = 0.2
    seeds = list(range(10))

    # Algorithm settings
    erm_pretr_steps = [0, 400]
    erm_total_steps = [400, 600, 1000]
    penalties = [1000, 5000, 10000, 50000, 100000]
    sd_penalties = [1, 5, 10, 50, 100]
    group_dro_etas = [0.001, 0.01, 0.1, 0.5, 1.0]
    eqrm_alphas = [-100, -500, -1000, -5000, -10000]

    # Algorithm total steps
    algs_1 = ["groupdro", "sd", "iga"]
    algs_steps_1 = [(a, 1000) for a in algs_1]              # 1000 for GroupDRO, SD, IGA (1)

    algs_2 = ["irm", "vrex", "eqrm"]
    algs_steps_2 = [(a, 600) for a in algs_1]               # 600 for IRM, VREX, QRM (2)

    algs_steps = algs_steps_1 + algs_steps_2
    algs_settings = [(a, pretr_s, total_s) for (a, total_s) in algs_steps for pretr_s in erm_pretr_steps]

    # ERM and Oracle total steps
    train_envs = ["default", "gray"]                        # ERM and Oracle (grayscale train envs)
    erm_settings = [(e, s) for e in train_envs for s in erm_total_steps]

    # Create file of commands and base command
    output_file = open(f"job_scripts/{args.exp_name}.txt", "w")
    base_call = (
        f"python train.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--exp_name {args.exp_name} "
        f"--lr {lr} "
        f"--batch_size {batch_size} "
        f"--dropout_p {dropout_p}"
    )

    # Print command lines to file
    for seed in seeds:
        # Create job calls for erm and erm_grayscale (oracle)
        for envs, steps in erm_settings:
            erm_call = (
                f"{base_call} "
                f"--seed {seed} "
                f"--erm_pretrain_iters 0 "
                f"--algorithm erm "
                f"--train_envs {envs} "
                f"--steps {steps}"
            )
            print(erm_call, file=output_file)

        # Create job calls for all other algs
        for i, (alg, pretr_steps, total_steps) in enumerate(algs_settings):
            alg_base_call = (
                f"{base_call} "
                f"--seed {seed} "
                f"--erm_pretrain_iters {pretr_steps} "
                f"--lr_cos_sched "
                f"--algorithm {alg} "
                f"--steps {total_steps} "
                f"--save_ckpts"
            )
            if alg in ["irm", "vrex", "iga"]:
                alg_settings = [f"--penalty_weight {pen}" for pen in penalties]
            elif alg == "sd":
                alg_settings = [f"--penalty_weight {pen}" for pen in sd_penalties]
            elif alg == "eqrm":
                alg_settings = [f"--alpha {a}" for a in eqrm_alphas]
            elif alg == "groupdro":
                alg_settings = [f"--groupdro_eta {e}" for e in group_dro_etas]
            else:
                raise ValueError(f"Invalid algorithm selected {alg}.")

            for alg_setting in alg_settings:
                alg_call = (
                    f"{alg_base_call} "
                    f"{alg_setting}"
                )
                if seed == seeds[-1] and i == (len(algs_settings) - 1) and alg_setting == alg_settings[-1]:
                    # last line, no newline "\n"
                    print(alg_call.strip(), file=output_file, end="")
                else:
                    print(alg_call.strip(), file=output_file)

    output_file.close()
    output_file = open(f"job_scripts/{args.exp_name}.txt", "r")
    print(f'Total num experiments = {len(output_file.readlines())}')
