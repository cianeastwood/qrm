# Linear regression experiments
This sub-repo contains code for running experiments on the linear regression dataset of our paper:

<p align="center">
  <img src="https://github.com/cianeastwood/prob_gen/blob/clean/assets/overview_lr.png?raw=true" width="700" alt="LR dataset" />
</p>

## Installation

```bash
pip install requirements.txt
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
Run the commands in the text file. We did so by submitting them to a slurm cluster using 
[submitit](https://github.com/facebookincubator/submitit). If you are also using a cluster with slurm then 
`job_scripts/submit_jobs.py` and the command below may provide a useful starting point (after installing submitit):
```sh
python job_scripts/submit_jobs.py -c job_scripts/reproduce.txt
```

### 3. View results
Sweep results will have been saved to results/reproduce/quantiles. View the results of this sweep using:

```bash
python collect_results.py /my/output/dir/reproduce/quantiles
```

### 4. Plot results
To plot the results and reproduce Figures 3a--3d of our paper, add the `--plot` flag which will save these to figs/.
```bash
python collect_results.py /my/output/dir/reproduce/quantiles --plot
```

## Acknowledgements

This repo structure was based on the structure of
[Linear unit-tests for invariance discovery](https://github.com/facebookresearch/InvarianceUnitTests).
