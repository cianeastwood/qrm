# CMNIST experiments
This sub-repo contains code for running experiments on the ColoredMNIST dataset of 
[Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893).

<p align="center">
  <img src="https://github.com/cianeastwood/qrm/blob/main/assets/overview_cmnist.png?raw=true" width="300" alt="CMNIST images" />
</p>

## Requirements
We used Python 3.10. The requirements can then be installed using:
```
pip install -r requirements.txt
```

## Single run
```
python train.py --algorithm eqrm --alpha -1000 --save_ckpts --lr_cos_sched
```

## Reproducing results (multiple runs)
### 1. Create sweep commands
Create a text file of commands `job_scripts/reproduce.txt` with the following command, specifying the _absolute_ 
paths to your data and output directories:
```sh
python -m job_scripts.gen_exps --exp_name reproduce --data_dir /my/data/dir --output_dir /my/output/dir
```

### 2. Run the commands
Run the commands in the text file. To do so on a local machine (warning: may take a while!), use:
```sh
source ./job_scripts/reproduce.txt
```

To do so via a slurm cluster, the script `job_scripts/submit_jobs.py` may provide a useful starting point, editing where necessary with the details of your cluster. After installing [submitit](https://github.com/facebookincubator/submitit), the following command will then run the commands in the text file:
```sh
python -m job_scripts.submit_jobs -c job_scripts/reproduce.txt
```

### 3. View results
Results will have been saved to /my/output/dir (due to `--output_dir` and `--exp_name` flags in step 1). View
with:

```bash
python collect_results.py /my/output/dir/results/reproduce
```

## Filtering results for analysis
You can use the flags of `collect_results.py` to filter results, e.g. to view runs that did not use ERM pretraining 
and had a penalty weight of 10: 
```bash
python collect_results.py /my/output/dir/results/reproduce --arg_values erm_pretrain_iters=0,penalty_weight=10
```
