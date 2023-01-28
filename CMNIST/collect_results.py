import json
import argparse
import glob
import pandas as pd
import os
import ast
from functools import partial


def print_row(row, col_width=15, latex=False, col_0_width=10):
    sep = " & " if latex else "  "
    end_ = "\\\\" if latex else ""
    print(sep.join([x.ljust(col_0_width) if i == 0 else x.ljust(col_width) for i, x in enumerate(row)]), end_)


def print_table(table_data, table_headers, col_width=15, latex=False, col_0_width=10):
    print("\n")
    if latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\begin{table}")
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * len(table_headers) + "}")
        print("\\toprule")

    print_row([""] + table_headers, col_width, latex, col_0_width)

    if latex:
        print("\\midrule")

    for row_k, row_vs in sorted(table_data.items()):
        print_row([row_k] + row_vs, col_width, latex, col_0_width)

    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")
        print("\\label{main_results}")
        print("\\caption{Main results.}")
        print("\\end{table}")
        print("\\end{document}")


def print_table_hparams(table):
    print("\n")
    for alg in sorted(table.keys()):
        print(alg, table[alg])
    print("\n")


def filter_df_dict_column(row, dict_column_name, filter_dict):
    """ Filter a dataframe based on a dict-type column. """
    df_dict = dict(row[dict_column_name])
    for k, v in filter_dict.items():
        if k not in df_dict:
            raise ValueError(f"Key '{k}' not in df_dict with keys: {df_dict.keys()}")
        if df_dict[k] != v:
            return False
    return True


def build_table(dirname, test_env_ms=0.9, test_envs_print=(), ms_type="best",
                algs=None, arg_values=None, latex=False, standard_error=False):
    records = []
    for fname in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(fname, "r") as f:
            if os.path.getsize(fname) != 0:
                records.append(f.readline().strip())

    df = pd.read_json("\n".join(records), lines=True)
    if algs is not None:
        df = df.query(f"algorithm in {algs}")
    if arg_values is not None:
        filter_fn = partial(filter_df_dict_column, dict_column_name='args', filter_dict=arg_values)
        mask = df.apply(filter_fn, axis=1)
        df = df[mask]

    print(f'{len(df)} records.')

    table = {}
    table_hparams = {}
    pm = "$\\pm$" if latex else "+-"

    env_choices = "|".join([str(p) for p in test_envs_print])       # use regex OR operator '|'
    envs = sorted(list(set([c.replace(f"_acc_{ms_type}", "")
                            for c in df.filter(regex=f"({env_choices})_acc_{ms_type}").columns])))
    print(f"Envs: {envs}")

    for alg in df["algorithm"].unique():
        # filtered by model
        df_m = df[df["algorithm"] == alg]

        # find the args/hparams which led to the best mean performance over seeds
        best_args_id = df_m.groupby("args_id").mean().filter(regex=f'{test_env_ms}_acc_{ms_type}').sum(1).idxmax()

        # store the best args/hparams
        df_m_a_args = df_m[df_m["args_id"] == best_args_id].filter(regex="args")
        table_hparams[alg] = json.dumps(df_m_a_args['args'].iloc[0])

        # get the accuracies over seeds for these best args/hparams
        df_m_a = df_m[df_m["args_id"] == best_args_id].filter(regex=f"({env_choices})_acc_{ms_type}")
        n_seeds = len(df_m_a)

        ms, stds = df_m_a.mean().to_numpy(), df_m_a.std().to_numpy()
        if standard_error:
            ses = stds / n_seeds
            fmt_strs = [f"{m:.3f} {pm} {s:.3f}" for m, s in zip(ms, ses)]
        else:
            fmt_strs = [f"{m:.3f} {pm} {s:.3f}" for m, s in zip(ms, stds)]

        table[alg] = fmt_strs

    return table, envs, table_hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument('--algorithms', nargs='+', default=None)
    parser.add_argument('--arg_values', type=str, default=None)
    parser.add_argument('--model_selection_env', type=float, default=0.9)
    parser.add_argument('--model_selection_type', type=str, default="final", choices=["best", "final"])
    parser.add_argument('--test_envs_print', type=str, default="all")
    args = parser.parse_args()

    if args.arg_values is not None:
        # pass in arguments and their values, e.g.: 'penalty=100,erm_pretrain_iters=0,lr=0.0001'
        kvs = [kv.split("=") for kv in args.arg_values.split(",")]
        args.arg_values = {kv[0]: ast.literal_eval(kv[1]) for kv in kvs}     # infer type of string, e.g. float, int

    if args.test_envs_print == "all":
        test_envs_print = [i / 10. for i in range(11)]
    elif args.test_envs_print == "train_test":
        test_envs_print = [0.1, 0.2, 0.9]
    else:
        test_envs_print = [float(p) for p in args.test_envs_print.split(",")]

    table_data, table_headers, table_hparams = build_table(args.dirname, args.model_selection_env, test_envs_print,
                                                           args.model_selection_type, args.algorithms, args.arg_values,
                                                           args.latex)

    print_table(table_data, table_headers, latex=args.latex)
    print_table_hparams(table_hparams)
