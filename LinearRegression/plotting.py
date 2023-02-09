""" Functions for plotting. """

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from tueplots import bundles

# mpl.use('macOsX')
plt.rcParams.update(bundles.neurips2022())
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath,amsfonts}')


def get_model_name(m_name):
    if "QRM" in m_name:
        m_name_ = utils.get_alpha_qrm_name_latex(m_name, alpha_value_only=True, tight=True)
    elif m_name == "Oracle":
        m_name_ = "Causal"
    else:
        m_name_ = m_name
    return m_name_


def get_line_colour_and_style(model_name, n_settings_per_model=6, clr_iterator=None):
    if model_name in ["Oracle", "Causal"]:
        return "black", "dotted", None
    elif model_name == "ERM":
        return "gray", "dashed", None
    else:        # "QRM" in model_name:
        if clr_iterator is None:
            clr_iterator = iter(plt.cm.coolwarm(np.linspace(0, 1, n_settings_per_model)))   # + 1
            # for _ in range(1):
            #     _ = next(clr_iterator)  # avoid lightest colours, hard to see
        return next(clr_iterator), "solid", clr_iterator


def plot_all_quantile_figs(table_all_envs, table_quantiles, weights_dict, savedir="figs/"):
    """
    Plot all linear regression figures relating to quantile-based evaluation.

    :param table_all_envs: Validation-set results on all training envs. See collect_results.py.
    :param table_quantiles: Results on envs at different quantiles. See collect_results.py.
    :param weights_dict: Dict containing model weights. Expected in the structure of collect_results.py.
    :param savedir: Directory in which to save the figures.
    """

    plot_risk_pdfs(table_all_envs, savedir=savedir)
    plot_risk_cdfs(table_all_envs, savedir=savedir)
    qq_plot(table_all_envs, savedir=savedir)
    plot_quantile_performance(table_quantiles, savedir=savedir)
    plot_coefficients(weights_dict, savedir=savedir)


def plot_quantile_performance(table, savedir="figs/", fname="quantile_perf"):
    """
    Plot the quantile performance for different algorithms/models.

    Used for Fig. 3a of our paper (https://arxiv.org/pdf/2207.09944.pdf).

    Warning: many hardcoded values below! Change for your data.
    """
    # Plot params
    figsize = (4, 3)
    ylim = (0.15, 2.3)          # risk (ignoring exploding risks as we go OOD)

    # Setup
    dataset_names = sorted(table.keys())
    clr_iterator = None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    os.makedirs(savedir, exist_ok=True)

    for dataset_name in dataset_names:
        # Get data
        dataset_results = table[dataset_name]
        model_names = sorted(dataset_results.keys(), key=utils.sort_models_alpha)

        # More setup
        qs_ss = list(dataset_results[list(dataset_results.keys())[0]].keys())  # qs=quantiles,ss=stddevs. E.g. '0.5_0.1'
        qs, ss = list(zip(*[q_s.split("_") for q_s in qs_ss]))
        qs_float = [float(q) for q in qs]
        ss_float = [float(s) for s in ss]

        for m_name in model_names:
            # Get data and color
            m_means = [dataset_results[m_name][q_s]['mean'] for q_s in qs_ss]
            # m_stds = [dataset_results[m_name][q_s]['std'] for q_s in qs_ss]
            clr, style, clr_iterator = get_line_colour_and_style(m_name, clr_iterator=clr_iterator)

            # Plot
            m_name_display = get_model_name(m_name)
            ax.plot(qs_float, m_means, label=m_name_display, color=clr, linestyle=style, marker='.')
            # ax.errorbar(qs_float, m_means, yerr=m_stds, label=m_name, marker='.', color=clr, linestyle=style)

        # Plot settings and labels
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_xscale('logit')
        ax.set_xticks(qs_float)
        ax.xaxis.set_major_formatter('{x:.4g}')
        ax.set_xlabel(r"True quantile of $\mathbb{Q}(\sigma_{\text{effect}})$")
        ax.set_ylabel('Risk')
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.legend(loc='best')

        # Create second x-axis on top of the plot with the *values* at quantile, i.e. the stddev/sigma values for X_2
        ax_sigma = ax.twiny()
        ax_sigma.set_xscale('logit')
        ax_sigma.set_xlim(ax.get_xlim())
        ax_sigma.set_xticks(ax.get_xticks())
        ax_sigma.set_xticklabels(ss_float)
        ax_sigma.set_xlabel(r"$\sigma_{\text{effect}}$")

        plt.savefig(os.path.join(savedir, f"{dataset_name}_{fname}.pdf"))


def plot_risk_pdfs(table, savedir="figs/", fname="risk_pdfs"):
    """
    Plot the risk distributions (pdfs) for different algorithms/models.

    Used for Fig. 3b of our paper (https://arxiv.org/pdf/2207.09944.pdf).

    Warning: many hardcoded values below! Change for your data.
    """
    # Plot params
    figsize = (4, 3)
    xlim = (0, 2.3)             # risk (cut off large risks for better clarity)
    ylim = (-0.3, 3.5)          # density (cut off densities with large peaks)

    # Setup
    dataset_names = sorted(table.keys())
    clr_iterator = None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    os.makedirs(savedir, exist_ok=True)

    for dataset_name in dataset_names:
        dataset_results = table[dataset_name]
        model_names = sorted(dataset_results.keys(), key=utils.sort_models_alpha)
        envs = dataset_results[list(dataset_results.keys())[0]].keys()

        for m_name in model_names:
            # Get data and settings
            envs_means = [dataset_results[m_name][e]['mean'] for e in envs]
            clr, style, clr_iterator = get_line_colour_and_style(m_name, clr_iterator=clr_iterator)
            m_name_display = get_model_name(m_name)

            # Plot density using KDE
            kde = utils.KernelDensityEstimator(torch.tensor(envs_means))
            xs = np.linspace(*xlim, num=1000)
            p_xs = kde(torch.from_numpy(xs)).numpy()
            ax.plot(xs, p_xs, label=m_name_display, color=clr, linestyle=style)

        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_xlabel(r"Risk")
        ax.set_ylabel(r'Density')
        plt.legend(loc='best')
        plt.savefig(os.path.join(savedir, f"{dataset_name}_{fname}.pdf"))


def plot_risk_cdfs(table, savedir="figs/", fname="risk_cdfs"):
    """
    Plot the risk cdfs for different algorithms/models.

    Used for Fig. 6 (Appendix G) of our paper (https://arxiv.org/pdf/2207.09944.pdf).

    Warning: many hardcoded values below! Change for your data.
    """
    # Plot params
    figsize = (4, 3)
    xlim = (0, 2.3)         # risk (cut off large risks for better clarity)

    # Setup
    dataset_names = sorted(table.keys())
    clr_iterator = None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    os.makedirs(savedir, exist_ok=True)

    for dataset_name in dataset_names:
        dataset_results = table[dataset_name]
        model_names = sorted(dataset_results.keys(), key=utils.sort_models_alpha)
        envs = dataset_results[list(dataset_results.keys())[0]].keys()

        for m_name in model_names:
            # Get data and settings
            envs_means = sorted([dataset_results[m_name][e]['mean'] for e in envs])
            clr, style, clr_iterator = get_line_colour_and_style(m_name, clr_iterator=clr_iterator)
            m_name_display = get_model_name(m_name)

            # Plot density using KDE
            kde = utils.KernelDensityEstimator(torch.tensor(envs_means))
            xs = np.linspace(*xlim, num=1000)
            cum_p_xs = [kde.cdf(x).item() for x in torch.from_numpy(xs)]
            ax.plot(xs, cum_p_xs, label=m_name_display, color=clr, linestyle=style)

        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_xlabel(r"Risk")
        ax.set_ylabel(r'Probability')
        plt.legend(loc='best')
        plt.savefig(os.path.join(savedir, f"{dataset_name}_{fname}.pdf"))


def plot_coefficients(weights_dict, savedir="figs/", fname="coefficients"):
    """
    Plot the regression coefficients for the cause (X_1) and effect (X_2) features.

    Used for Fig. 3c of our paper (https://arxiv.org/pdf/2207.09944.pdf).

    Warning: many hardcoded values below! Change for your data.
    """
    # Plot params
    bar_labels = [r"$\beta_{\text{cause}}$", r"$\beta_{\text{effect}}$"]
    total_width = 0.8
    single_width = 1
    figsize = (4, 3)

    # Setup
    dataset_names = sorted(weights_dict.keys())
    xs = list(range((len(bar_labels))))
    clr_iterator = None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    os.makedirs(savedir, exist_ok=True)

    for id_d, dataset_name in enumerate(dataset_names):
        dataset_results = weights_dict[dataset_name]
        bar_data = {k: [max(w, 0) + 0.01 for w in sub_d["w"]] for k, sub_d in dataset_results.items()}
        sorted_m_names = sorted(bar_data.keys(), key=utils.sort_models_alpha)

        # Number of bars per group
        n_bars = len(bar_data)

        # The width of a single bar
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for i, m_name in enumerate(sorted_m_names):
            ws = bar_data[m_name]

            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

            # Get model settings
            clr, style, clr_iterator = get_line_colour_and_style(m_name, clr_iterator=clr_iterator)

            # Draw a bar for every value of that type
            for x, y in enumerate(ws):
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=clr)

            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])

        # Final settings and save
        display_m_names = [get_model_name(m) for m in sorted_m_names]
        ax.legend(bars, display_m_names, loc='best')
        ax.set_xlabel("Coefficient")
        ax.set_ylabel('Magnitude')
        ax.set_xticks(xs)
        ax.set_xticklabels(bar_labels, fontsize=10)
        plt.savefig(os.path.join(savedir, f"{dataset_name}_{fname}.pdf"))


def qq_plot(table, savedir="figs/", fname="qq", m_name="EQRM_a=0.9"):
    """
    Create a QQ-plot for one particular model's risk distribution (m_name).

    Used for Fig. 3d of our paper (https://arxiv.org/pdf/2207.09944.pdf).

    Warning: hardcoded values below! Change for your data (e.g. subsample_ns).
    """
    # Plot params
    figsize = (3, 3)

    # Setup
    dataset_names = sorted(table.keys())
    utils.set_seed()
    os.makedirs(savedir, exist_ok=True)

    for dataset_name in dataset_names:
        # More setup
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get data
        dataset_results = table[dataset_name]
        envs = dataset_results[list(dataset_results.keys())[0]].keys()
        envs_means = [dataset_results[m_name][e]['mean'] for e in envs]

        # Choose subsample_sizes s with: 1 < s < n_envs
        true_n = len(envs_means)
        subsample_ns = [true_n // 2, true_n // 10, true_n // 20, true_n // 100]
        subsample_ns = [n for n in subsample_ns if n > 1]

        # Build KDE
        kde = utils.KernelDensityEstimator(torch.tensor(envs_means), bw_select="silverman")
        xs = np.linspace(min(envs_means), max(envs_means), num=true_n)
        true_q_values = [kde.cdf(x).item() for x in torch.from_numpy(xs)]
        ax.plot(true_q_values, true_q_values, label=f"True ($m={{{true_n}}}$)", color="black")

        clr_iterator = iter(plt.cm.viridis(np.linspace(0, 1, len(subsample_ns) + 1)))
        next(clr_iterator)  # first yellow too bright, hard to see

        for subsample_size in subsample_ns:
            clr = next(clr_iterator)
            m_means_ss = np.random.choice(envs_means, subsample_size, replace=False)
            kde = utils.KernelDensityEstimator(torch.tensor(m_means_ss), bw_select="silverman")
            est_q_values = [kde.cdf(x).item() for x in torch.from_numpy(xs)]
            ax.plot(true_q_values, est_q_values, label=f"m={subsample_size}", color=clr)

        ax.legend()
        ax.set_xlabel(f"True q. ($m={true_n}$)")
        ax.set_ylabel(f"Data q. ($m < {true_n}$)")
        plt.savefig(os.path.join(savedir, f"{dataset_name}_{fname}.pdf"))
