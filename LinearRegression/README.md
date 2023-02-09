# Linear regression experiments
This sub-repo contains code for running experiments on the linear regression dataset of our paper:

<p align="center">
  <img src="https://github.com/cianeastwood/qrm/blob/main/assets/overview_lr.png?raw=true" width="700" alt="LR dataset" />
</p>

## Installation

```bash
pip install -r requirements.txt
```

## Single run

```bash
python train.py --alg EQRM --dataset Example2 --n_envs 50 --n_samples 50000 --hparams_fixed alpha=0.99
```

## Reproducing results (multiple runs)
### 1. Create sweep commands
Create a text file of commands `job_scripts/reproduce.txt`, specifying the _absolute_ path to your output directory:
```bash
python job_scripts/gen_exps.py --output_dir /my/output/dir
```

### 2. Run the commands
Run the commands in the text file. To do so on a local machine (warning: may take a while!), use:
```bash
source ./job_scripts/reproduce.txt
```

To do so via a slurm cluster, the script `job_scripts/submit_jobs.py` may provide a useful starting point, editing where necessary with the details of your cluster. After installing [submitit](https://github.com/facebookincubator/submitit), the following command will then run the commands/jobs in the text file:
```bash
python job_scripts/submit_jobs.py -c job_scripts/reproduce.txt
```

### 3. View results
Sweep results will have been saved to results/reproduce/quantiles. View the results of this sweep using:

```bash
python collect_results.py /my/output/dir/reproduce/quantiles
```

### 4. Plot results
To plot the results, and reproduce Figures 3a--3d of our paper, some dependencies may need to be installed to render latex text with matplotlib. We hope to remove these dependecies in the future. With Mac OS, it is usually sufficient to uncomment the `mpl.use('macOsX')` line at the top of `plot_results.py`. With Ubuntu, following commands should be sufficient:
```bash
sudo apt install cm-super 
sudo apt install dvipng
```

With these dependencies installed, the results can be plotted using the following command, saving figures to `figs/` by default:
```bash
python collect_results.py /my/output/dir/reproduce/quantiles --plot
```

## Acknowledgements

This repo structure was based on that of
[Linear unit-tests for invariance discovery](https://github.com/facebookresearch/InvarianceUnitTests).
