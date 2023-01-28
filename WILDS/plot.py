import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tueplots import bundles
import argparse

plt.rcParams.update(bundles.neurips2022())
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath, amsfonts}')
plt.rcParams["figure.figsize"] = (6, 6)

fontsizes = {'font.size': 20,
                 'axes.labelsize': 25,
                 'legend.fontsize': 25,
                 'xtick.labelsize': 25,
                 'ytick.labelsize': 25,
                 'axes.titlesize': 25}
plt.rcParams.update(fontsizes)

DATASET_MAP = {
    'iwildcam': 'IWildCam',
    'ogb-molpcba': 'OGB-MolPCBA'
}
X_AXIS_MAP = {
    'iwildcam': [0, 6.5],
    'ogb-molpcba': [0, 0.04]
}

ALPHAS = [0.25, 0.5, 0.75, 0.9, 0.99, 31]
c_iter = iter(plt.cm.coolwarm(np.linspace(0, 1, len(ALPHAS))))
COLORS = [next(c_iter) for _ in ALPHAS]
def get_color(alpha):
    return COLORS[ALPHAS.index(alpha)]
BLACK = np.array([0, 0, 0, 1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['iwildcam', 'ogb-molpcba'], required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--roots', nargs='+', type=str, required=True)
    parser.add_argument('--algorithms', nargs='+', required=True)
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()

    assert(len(args.roots) == len(args.algorithms))

    palette = [BLACK] + [get_color(a) for a in [0.25, 0.5, 0.75, 0.9, 0.99]]
    sns.set_palette(palette)

    def group_risk(df):
        groups = sorted(df.Group.unique())
        risks = [df[df.Group == g]['Risk'].mean() for g in groups]
        return pd.DataFrame(
            data=risks,
            columns=['Risk'])

    def create_df(root, fname, alg):
        path = os.path.join(root, fname)
        df = pd.read_pickle(path)
        df.columns = ['Risk', 'Group']
        df = group_risk(df)
        df['Algorithm'] = alg

        return df

    dfs = []
    for alg_name, root_dir in zip(args.algorithms, args.roots):
        df = create_df(
            root=root_dir,
            fname=f'{args.dataset}_split:{args.split}_seed:0_epoch:{args.epoch}_risks.pd',
            alg=alg_name
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    fig, ax1 = plt.subplots()

    g = sns.kdeplot(
        data=df,
        x='Risk',
        hue='Algorithm',
        linewidth=6,
        ax=ax1)
    # ax1.set(xlim=X_AXIS_MAP[args.dataset])
    ax1.set(xlabel='Risk')
    ax1.set(title=f'{DATASET_MAP[args.dataset]} Risk Distribution')
    g.legend_.set_title(None)

    plt.tight_layout()
    # plt.show()
    plt.savefig('risk_dists.png')
    