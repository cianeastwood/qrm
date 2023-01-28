import os
import argparse
import ast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Regression experiments')
    parser.add_argument('--exp_name', type=str, default="reproduce")
    parser.add_argument('--algorithms', nargs='+', default=["ERM", "Oracle", "EQRM"])
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--alphas', nargs='+', default=[str(a) for a in [0.25, 0.5, 0.75, 0.9, 0.99, int(-10**30)]])
    parser.add_argument('--datasets', nargs='+', default=["Example2"])
    parser.add_argument('--dim_inv', type=int, default=1)
    parser.add_argument('--dim_spu', type=int, default=1)
    parser.add_argument('--n_envs', type=int, default=1000)
    parser.add_argument('--test_env_type', type=str, default="quantiles", choices=["quantiles", "shuffled"])
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--n_data_seeds', type=int, default=1)
    parser.add_argument('--n_alg_seeds', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="results")
    args = parser.parse_args()

    # Gather experiment settings
    eqrm_alphas = [ast.literal_eval(a) for a in args.alphas]
    exp_output_dir = os.path.join(args.output_dir, args.exp_name)

    # Create file of commands and base call/command
    output_file = open(f"job_scripts/{args.exp_name}.txt", "w")
    base_call = (
        f"python train.py "
        f"--n_iterations {args.n_iterations} "
        f"--output_dir {exp_output_dir} "
        f"--n_envs {args.n_envs} "
        f"--n_samples {args.n_samples} "
        f"--test_env_type {args.test_env_type} "
        f"--dim_inv {args.dim_inv} "
        f"--dim_spu {args.dim_spu}"
    )

    def is_last_call(alg, dset, d_seed, m_seed):
        return alg == args.algorithms[-1] and dset == args.datasets[-1] \
               and (d_seed == args.n_data_seeds - 1) and m_seed == (args.n_alg_seeds - 1)

    for dataset in args.datasets:
        for alg in args.algorithms:
            for data_seed in range(args.n_data_seeds):
                for alg_seed in range(args.n_alg_seeds):
                    hparams = "random" if alg_seed else "default"
                    alg_base_call = (
                        f"{base_call} "
                        f"--alg {alg} "
                        f"--dataset {dataset} "
                        f"--data_seed {data_seed} "
                        f"--alg_seed {alg_seed} "
                        f"--hparams {hparams}"
                    )
                    if "QRM" in alg:
                        alg_settings = [f"--hparams_fixed alpha={a}" for a in eqrm_alphas]
                    else:
                        alg_settings = [""]         # add any algorithm-specific settings/hparams here

                    for alg_setting in alg_settings:
                        alg_call = (
                            f"{alg_base_call} "
                            f"{alg_setting}"
                        )
                        if alg_setting == alg_settings[-1] and is_last_call(alg, dataset, data_seed, alg_seed):
                            # last line, no newline "\n"
                            print(alg_call.strip(), file=output_file, end="")
                        else:
                            print(alg_call.strip(), file=output_file)

    output_file.close()
    output_file = open(f"job_scripts/{args.exp_name}.txt", "r")
    print(f'Total num experiments = {len(output_file.readlines())}')
