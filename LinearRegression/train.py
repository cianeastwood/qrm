import argparse
import hashlib
import pprint
import json
import os
import ast

import datasets
import algorithms
import utils

TEST_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]


def edit_alg_name_w_fixed_hparam(alg_name, hparam_name, hparam_value):
    if alg_name == "EQRM" and hparam_name == "alpha":
        if hparam_value > 0:
            alpha_str = f"a={hparam_value}"
        else:
            alpha_str = "a~=1"
        return f"{alg_name}_{alpha_str}"
    else:
        # If any other algorithm names should be edited to contain the hparam value (for sweeps), put it here.
        return alg_name


def fit_and_save_results(alg, envs, args, results_dirname, dataset):
    # build file name
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    results_fname = os.path.join(results_dirname, md5_fname + ".jsonl")

    # fit the dataset
    alg.fit(
        envs=envs,
        n_iterations=args["n_iterations"],
        callback=args["callback"]
    )

    # compute the train, validation and test errors
    for split in ("train", "validation", "test"):
        key = "error_" + split
        for k_env, env in zip(envs[split]["keys"], envs[split]["envs"]):
            env_error = utils.compute_error(alg, *env)
            if "p_test" in results_dirname and ("q=" not in k_env):
                # add environment standard deviation for training envs ('q=' is an indicator of test envs)
                env_sdv = dataset.envs[k_env]
                args[f"{key}_{k_env}_s={env_sdv:.1f}"] = env_error
            else:
                args[f"{key}_{k_env}"] = env_error

    # save weights and biases
    w, b = alg.network.weight, alg.network.bias
    args["w_shape"], args["b_shape"] = list(w.shape), list(b.shape)
    args["w"], args["b"] = [round(w_.item(), 3) for w_ in w.flatten()], [round(b_.item(), 3) for b_ in b.flatten()]

    # write results
    results_file = open(results_fname, "w")
    results_file.write(json.dumps(args))
    results_file.close()

    return args


def run_experiment(args):
    # build directory name
    results_dirname = os.path.join(args["output_dir"], args["test_env_type"])
    os.makedirs(results_dirname, exist_ok=True)

    # create dataset
    utils.set_seed(args["data_seed"])
    dataset = datasets.DATASETS[args["dataset"]](
        dim_inv=args["dim_inv"],
        dim_spu=args["dim_spu"],
        n_envs=args["n_envs"],
        verbose=args["verbose"],
    )

    # oracle (ERM trained on test mode, i.e. scrambled effect features X_2)
    train_split = "train" if args["alg"] != "Oracle" else "test"

    # sample the envs
    envs = {}
    for key_split, split in zip(("train", "validation", "test"), (train_split, train_split, "test")):
        envs[key_split] = {"keys": [], "envs": []}
        for env in dataset.envs:
            envs[key_split]["envs"].append(dataset.sample(
                n=args["n_samples"],
                env=env,
                split=split)
            )
            envs[key_split]["keys"].append(env)

    # sample separate test envs at different quantiles of the env distribution
    if args["test_env_type"] == "quantiles":
        if not hasattr(dataset, "p_env"):
            raise ValueError("Datasets must have a distribution over environments (dataset.p_env) in order to "
                             f"evaluate performance at different quantiles. Chosen dataset {args['dataset']} does not.")
        for q in TEST_QUANTILES:
            inputs, outputs, sdv = dataset.sample(n=args["n_samples"], env=f"test_q={q}")
            envs["test"]["envs"].append((inputs, outputs))
            envs["test"]["keys"].append(f"q={q}_s={sdv:.1f}")

    # offsetting alg seed to avoid overlap with data_seed
    utils.set_seed(args["alg_seed"] + 1000)

    # selecting alg
    alg = algorithms.ALGORITHMS[args["alg"]](
        in_features=args["dim_inv"] + args["dim_spu"],
        out_features=1,
        task=dataset.task,
        hparams=args["hparams"]
    )

    # fix certain hparams to specified values, e.g.: 'alpha=100,lr=0.0001'
    if args["hparams_fixed"] is not None:
        kvs = [kv.split("=") for kv in args["hparams_fixed"].split(",")]
        args["hparams_fixed"] = {kv[0]: ast.literal_eval(kv[1]) for kv in kvs}     # infer type of s
        for k, v in args["hparams_fixed"].items():
            alg.hparams[k] = v
            args["alg"] = edit_alg_name_w_fixed_hparam(args["alg"], k, v)

    # update this field for printing purposes
    args["hparams"] = alg.hparams

    # Fit single alg
    args_ = fit_and_save_results(alg, envs, args, results_dirname, dataset)

    return args_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--alg', type=str, default="ERM")
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--dataset', type=str, default="Example1")
    parser.add_argument('--dim_inv', type=int, default=1)
    parser.add_argument('--dim_spu', type=int, default=1)
    parser.add_argument('--n_envs', type=int, default=1000)
    parser.add_argument('--test_env_type', type=str, default="quantiles", choices=["quantiles", "shuffled"])
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--alg_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results/test_run")
    parser.add_argument('--callback', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hparams_fixed', type=str, help="Hparams to fix for grid sweeps. E.g.: alpha=0.9,lr=0.01")
    args = parser.parse_args()

    # Print results to 3 decimal places
    results = run_experiment(vars(args))
    for k, v in results.items():
        if isinstance(v, float):
            results[k] = round(v, 3)
    pprint.pprint(results)
