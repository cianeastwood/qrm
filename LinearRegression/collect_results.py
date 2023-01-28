""" Collect and (optionally) plot results."""

import pandas as pd
import glob
import os
import json
import argparse
import utils
import plotting


def print_row(row, col_width=15, latex=False, col_0_width=20):
    sep = " & " if latex else "  "
    end_ = "\\\\" if latex else ""
    print(sep.join([x.ljust(col_0_width) if i == 0 else x.ljust(col_width) for i, x in enumerate(row)]), end_)


def print_table(table, col_width=15, latex=False, col_0_width=20):
    col_names = sorted(table[next(iter(table))].keys())
    
    print("\n")
    if latex:
        col_names = [utils.get_alpha_qrm_name_latex(n) if "QRM" in n else n for n in col_names]

        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\begin{table}")
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * len(col_names) + "}")
        print("\\toprule")

    print_row([""] + col_names, col_width, latex, col_0_width)

    if latex:
        print("\\midrule")

    for row_k, row_v in sorted(table.items()):
        row_values = [row_k]
        for col_k, col_v in sorted(row_v.items()):
            row_values.append(col_v)
        print_row(row_values, col_width, latex, col_0_width)

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
    for dataset in table.keys(): 
        print(dataset, "\n")
        for alg in table[dataset].keys():
            print(alg, table[dataset][alg])
        print("\n")


def build_table(dirname, algs=None, n_envs=None, latex=False, standard_error=False,
                split="test", p_test=False):
    records = []
    for fname in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(fname, "r") as f:
            if os.path.getsize(fname) != 0:
                records.append(f.readline().strip())

    df = pd.read_json("\n".join(records), lines=True)
    if algs is not None:
        df = df.query(f"alg in {algs}")
    if n_envs is not None:
        df = df.query(f"n_envs=={n_envs}")

    print(f'{len(df)} records.')
    pm = "$\\pm$" if latex else "+-"

    table = {}
    table_avg = {}
    table_val = {}
    table_val_avg = {
        "data": {},
        "n_envs": 0,
        "dim_inv": 0,
        "dim_spu": 0
    }
    table_hparams = {}

    for dataset in df["dataset"].unique():
        # filtered by dataset
        df_d = df[df["dataset"] == dataset]

        if p_test:
            envs = sorted(list(set(
                [c[c.index("E") + 1:]
                 for c in df_d.filter(regex="error_test_E").columns])), key=lambda s: int(s.split("_")[0]))
        else:
            envs = sorted(list(set(
                [c[-1] for c in df_d.filter(regex="error_").columns])))

        if n_envs:
            envs = envs[:n_envs]

        table_hparams[dataset] = {}
        table_val[dataset] = {}
        for key in ["n_envs", "dim_inv", "dim_spu"]:
            table_val_avg[key] = int(df[key].iloc[0])
        table_val_avg["data"][dataset] = {}

        for alg in df["alg"].unique():
            # filtered by alg
            df_d_m = df_d[df_d["alg"] == alg]

            best_alg_seed = df_d_m.groupby("alg_seed").mean().filter(
                regex='error_validation').sum(1).idxmin()

            # filtered by hparams
            df_d_m_s = df_d_m[df_d_m["alg_seed"] == best_alg_seed].filter(
                regex=f"error_{split}")

            # store the best hparams
            df_d_m_s_h = df_d_m[df_d_m["alg_seed"] == best_alg_seed].filter(
                regex="hparams")
            table_hparams[dataset][alg] = json.dumps(
                df_d_m_s_h['hparams'].iloc[0])

            table_val[dataset][alg] = {}
            for env in envs:
                errors = df_d_m_s[[f"error_{split}_E{env}"]]
                std = float(errors.std(ddof=0))
                se = std / len(errors)
                fmt_str = "{:.2f} {} {:.2f}".format(
                    float(errors.mean()), pm, std)
                if standard_error:
                    fmt_str += " {} {:.1f}".format(
                        float('/', se))

                dataset_env = dataset + ".E" + str(env)
                if dataset_env not in table:
                    table[dataset_env] = {}

                table[dataset_env][alg] = fmt_str
                table_val[dataset][alg][env] = {
                    "mean": float(errors.mean()), 
                    "std": float(errors.std(ddof=0))
                    }

            # Avg
            if dataset not in table_avg:
                table_avg[dataset] = {}
            table_test_errors = df_d_m_s[[f"error_{split}_E{env}" for env in envs]]
            mean = table_test_errors.mean(axis=0).mean(axis=0)
            std = table_test_errors.std(axis=0, ddof=0).mean(axis=0)
            table_avg[dataset][alg] = f"{float(mean):.2f} {pm} {float(std):.2f}"
            table_val_avg["data"][dataset][alg] = {
                "mean": float(mean), 
                "std": float(std),
                "hparams": table_hparams[dataset][alg]
                }

    return table, table_avg, table_hparams, table_val, table_val_avg, df


def build_table_p_test(dirname, algs=None, n_envs=None, latex=False, standard_error=False):
    records = []
    for fname in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(fname, "r") as f:
            if os.path.getsize(fname) != 0:
                records.append(f.readline().strip())

    df = pd.read_json("\n".join(records), lines=True)
    if algs is not None:
        df = df.query(f"alg in {algs}")
    if n_envs is not None:
        df = df.query(f"n_envs=={n_envs}")

    print(f'{len(df)} records.')
    pm = "$\\pm$" if latex else "+-"

    table = {}
    table_val = {}
    table_hparams = {}
    w_dict = {}

    for dataset in df["dataset"].unique():
        # filtered by dataset
        df_d = df[df["dataset"] == dataset]
        envs = sorted(list(set(
            [c[-1] for c in df_d.filter(regex="error_").columns])))

        if n_envs:
            envs = envs[:n_envs]

        table_hparams[dataset] = {}
        table_val[dataset] = {}
        w_dict[dataset] = {}

        for alg in df["alg"].unique():
            # filtered by alg
            df_d_m = df_d[df_d["alg"] == alg]

            best_alg_seed = df_d_m.groupby("alg_seed").mean().filter(
                regex='error_validation').sum(1).idxmin()

            # filtered by hparams
            df_d_m_s = df_d_m[df_d_m["alg_seed"] == best_alg_seed].filter(
                regex="error_test")

            # store the best hparams
            df_d_m_s_h = df_d_m[df_d_m["alg_seed"] == best_alg_seed].filter(
                regex="hparams")
            table_hparams[dataset][alg] = json.dumps(
                df_d_m_s_h['hparams'].iloc[0])

            # gather values for all qs (p_tests) and ss (stddev tests)
            table_val[dataset][alg] = {}
            qs = sorted([float(k[k.index("q=") + 2:k.index("s=") - 1])
                         for k in df_d_m_s.filter(regex="error_test_q").keys()])
            qs = [f"{q}" for q in qs]           # q={q:.3f}
            ss = sorted([float(k[k.index("s=") + 2:])
                         for k in df_d_m_s.filter(regex="error_test_q").keys()])
            ss = [f"{s:.1f}" for s in ss]
            for q, s in zip(qs, ss):
                errors = df_d_m_s[[f"error_test_q={q}_s={s}"]]
                std = float(errors.std(ddof=0))
                se = std / len(errors)
                fmt_str = "{:.2f} {} {:.2f}".format(float(errors.mean()), pm, std)
                if standard_error:
                    fmt_str += " {} {:.1f}".format(float('/', se))

                dataset_env = dataset + f".q={q}.s={s}"
                if dataset_env not in table:
                    table[dataset_env] = {}

                table[dataset_env][alg] = fmt_str
                table_val[dataset][alg][f"{q}_{s}"] = {
                    "mean": float(errors.mean()),
                    "std": float(errors.std(ddof=0))
                    }

            # gather alg weights
            df_d_m_w = df_d_m[df_d_m["alg_seed"] == best_alg_seed].filter(regex="w|b")
            w_dict[dataset][alg] = {}
            for key in ["w", "b", "w_shape", "b_shape"]:
                w_dict[dataset][alg][key] = df_d_m_w[key].iloc[0]

    return table, table_hparams, table_val, w_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument('--algs', nargs='+', default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument("--plot", action="store_true", help="Plot results.")
    parser.add_argument("--figs_dir", type=str, default="figs/")
    args = parser.parse_args()

    # Set width of printed-table columns
    col_width = 18 if args.latex else 14

    # Fix bug in pandas' read_json module by overwriting
    pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

    if "quantiles" in args.dirname:
        # test envs correspond to samples from a distribution over environments: evaluate quantile performance

        # Collect results from stored .json files
        table_p_test, table_hparams_p_test, table_val_p_test, w_dict = build_table_p_test(args.dirname, args.algs,
                                                                                          args.n_envs, args.latex)

        # Print table and averaged table
        print_table(table_p_test, latex=args.latex, col_width=col_width, col_0_width=24)

        # Print best hparams
        print_table_hparams(table_hparams_p_test)

        if args.plot:
            _, _, _, table_val, _, _ = build_table(args.dirname, args.algs, args.n_envs, args.latex,
                                                   p_test=True, split="validation")

            plotting.plot_all_quantile_figs(
                table_quantiles=table_val_p_test,
                table_all_envs=table_val,
                weights_dict=w_dict,
                savedir=args.figs_dir,
            )

    else:
        # test envs correspond to "shuffled" effect features X_2: evaluate average performance

        # Collect results from stored .json files
        table, table_avg, table_hparams, table_val, table_val_avg, df = build_table(
            args.dirname, args.algs, args.n_envs, args.latex)

        # Print table and averaged table
        print_table(table, latex=args.latex)
        print_table(table_avg, latex=args.latex)

        # Print best hparams
        print_table_hparams(table_hparams)
