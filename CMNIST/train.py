import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json
import time
import copy
import os
import hashlib
import sys
import random
import math
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from lib import misc
import algorithms as algorithms
import networks as networks

from datasets import get_cmnist_datasets


if __name__ == "__main__":
    # -------- FLAGS --------
    parser = argparse.ArgumentParser(description='Colored MNIST')
    # Datasets
    parser.add_argument('--train_envs', type=str, default='default')
    parser.add_argument('--test_envs', type=str, default='0.1,0.2,0.5,0.9')     # test envs to log/print
    parser.add_argument('--test_env_ms', type=str, default='0.9')               # test env for selecting best model
    parser.add_argument('--full_resolution', action='store_true')

    # Network architecture
    parser.add_argument('--network', type=str, default="MLP")
    parser.add_argument('--mlp_hidden_dim', type=int, default=390)

    # Algorithms
    parser.add_argument('--algorithm', type=str, default='eqrm')
    parser.add_argument('--penalty_weight', type=float, default=1000)           # irm, vrex, etc.
    parser.add_argument('--alpha', type=float, default=-10**4)                  # qrm
    parser.add_argument('--groupdro_eta', type=float, default=1.)               # group_dro

    # General hparams
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=25000)
    parser.add_argument('--loss_fn', type=str, default='nll', choices=["nll, cross_ent"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_factor_reduction', type=float, default=1)
    parser.add_argument('--lr_cos_sched', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--erm_pretrain_iters', type=int, default=400)
    parser.add_argument('--eval_freq', type=int, default=50)

    # Directories and saving
    parser.add_argument('--data_dir', type=str, default="data/")
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--exp_name', type=str, default="reproduce")
    parser.add_argument('--save_ckpts', action='store_true')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')

    # Other
    parser.add_argument('--n_workers', type=int, default=0)

    # --------  SETUP --------
    args = parser.parse_args()
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    alg_arg_keys = ["algorithm", "penalty_weight", "alpha", "groupdro_eta",
                    "lr_factor_reduction", "lr_cos_sched", "steps", "save_ckpts"]
    if args.loss_fn == "nll":
        n_targets = 1
        loss_fn = F.binary_cross_entropy_with_logits
        int_target = False
    else:
        n_targets = 2
        loss_fn = F.cross_entropy
        int_target = True

    test_env_ps = tuple(float(e) for e in args.test_envs.split(","))
    if args.train_envs == 'default':
        train_env_ps = (0.1, 0.2)
    elif args.train_envs == 'gray':
        train_env_ps = (0.5, 0.5)
    else:
        train_env_ps = tuple(float(e) for e in args.train_envs.split(","))

    args.train_env_ps = train_env_ps
    train_env_names = [str(p) for p in train_env_ps]
    test_env_names = [str(p) for p in test_env_ps]

    # --------  LOGGING --------
    logs_dir = os.path.join(args.output_dir, "logs", args.exp_name)
    results_dir = os.path.join(args.output_dir, "results", args.exp_name)
    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    sys.stdout = misc.Tee(os.path.join(logs_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(logs_dir, 'err.txt'))
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # -------- REPRODUCIBILITY --------
    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    seed_all(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --------  DEVICE --------
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # --------  DATA LOADING --------
    envs = get_cmnist_datasets(args.data_dir, train_envs=train_env_ps, test_envs=test_env_ps,
                               cuda=(device == "cuda"), int_target=int_target, subsample=not args.full_resolution)
    train_envs, test_envs = envs[:len(train_env_ps)], envs[len(train_env_ps):]
    input_shape = train_envs[0].tensors[0].size()[1:]
    n_train_samples = train_envs[0].tensors[0].size()[0]
    steps_per_epoch = n_train_samples / args.batch_size

    train_loaders = [InfiniteDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
                     for env in train_envs]
    test_loaders = [FastDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
                    for env in test_envs]
    train_minibatches_iterator = zip(*train_loaders)

    # --------  NETWORK --------
    if args.network == "MLP":
        net = networks.MLP(np.prod(input_shape), args.mlp_hidden_dim, n_targets, dropout=args.dropout_p)
    elif args.network == "CNN":
        net = networks.CNN(input_shape)
    else:
        raise NotImplementedError

    # -------- ALGORITHM --------
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(net, vars(args), loss_fn)
    algorithm.to(device)

    # -------- LOAD ERM CHECKPOINT --------
    start_step = 1
    if args.erm_pretrain_iters > 0:
        erm_args = vars(copy.deepcopy(args))
        for k in alg_arg_keys:
            del erm_args[k]
        erm_ckpt_name = hashlib.md5(str(erm_args).encode('utf-8')).hexdigest()
        erm_ckpt_pth = os.path.join(ckpt_dir, f"{erm_ckpt_name}.pkl")
        if os.path.exists(erm_ckpt_pth):
            algorithm.load_state_dict(torch.load(erm_ckpt_pth, map_location=device), strict=False)
            print(f"ERM-pretrained model loaded: {erm_ckpt_name}.")
            start_step = args.erm_pretrain_iters + 1

    # -------- LR SCHEDULING --------
    def adjust_learning_rate(optimizer, current_step, lr, total_steps):
        lr_adj = lr
        lr_adj *= 0.5 * (1. + math.cos(math.pi * current_step / total_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adj

    # -------- UPDATES --------
    results = {}
    best_acc, best_weights = 0., copy.deepcopy(algorithm.state_dict())
    start_time, step_since_eval = time.time(), 0

    for step in range(start_step, args.steps + 1):
        step_start_time = time.time()

        # -------- ADJUST LR --------
        if args.lr_cos_sched and args.algorithm.lower() != "erm":
            if args.erm_pretrain_iters == 0:
                adjust_learning_rate(algorithm.optimizer, step, args.lr, args.steps)
            elif step > args.erm_pretrain_iters > 0:
                lr_ = args.lr / args.lr_factor_reduction
                steps_ = args.steps - args.erm_pretrain_iters
                step_ = step - args.erm_pretrain_iters
                adjust_learning_rate(algorithm.optimizer, step_, lr_, steps_)

        # -------- STEP --------
        step_values = algorithm.update(next(train_minibatches_iterator))

        # -------- EVALUATION --------
        if step % args.eval_freq == 0 or step == args.steps:
            results.update({
                'step': step,
                'epoch': step / steps_per_epoch,
                'avg_step_time': (time.time() - start_time) / (step - step_since_eval),
            })

            for key, val in step_values.items():
                results[key] = val

            for env_name, env_loader in zip(test_env_names, test_loaders):
                results[env_name + '_acc'] = misc.accuracy(algorithm, env_loader, device)
                results[env_name + '_loss'] = misc.loss(algorithm, env_loader, loss_fn, device)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
            results_keys = sorted(results.keys())

            misc.print_row(results_keys, colwidth=12)
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            start_time, step_since_eval = time.time(), 0
            if results[args.test_env_ms + '_acc'] > best_acc:
                best_acc = results[args.test_env_ms + '_acc']
                best_weights = copy.deepcopy(algorithm.state_dict())

        # -------- SAVE ERM CHECKPOINT --------
        if step == args.erm_pretrain_iters > 0 and args.save_ckpts:
            torch.save(algorithm.state_dict(), erm_ckpt_pth)
            print("Saved ERM-pretrained model.")

    # -------- FINAL EVAL ON ALL ENVS AND HELD-OUT TEST SET --------
    all_ps = [i / 10. for i in range(11)]
    all_env_names = [str(p) for p in all_ps]
    all_envs = get_cmnist_datasets(args.data_dir, train_envs=[], test_envs=all_ps, cuda=(device == "cuda"),
                                   int_target=int_target, subsample=not args.full_resolution, use_test_set=True)
    loaders = [FastDataLoader(dataset=env, batch_size=512, num_workers=args.n_workers)
               for env in all_envs]

    results = {}
    for ms_name in ["final", "best"]:
        if ms_name == "best":
            algorithm.load_state_dict(best_weights)

        # -------- EVAL --------
        for env_name, env_loader in zip(all_env_names, loaders):
            results[env_name + '_acc_' + ms_name] = misc.accuracy(algorithm, env_loader, device)
            results[env_name + '_loss_' + ms_name] = misc.loss(algorithm, env_loader, loss_fn, device)

        # -------- PRINT --------
        print(f"\n{ms_name} accuracies:")
        results_print_keys = [k for k in sorted(results.keys()) if f"_acc_{ms_name}" in k]
        misc.print_row([k.replace(f"_acc_{ms_name}", "") for k in results_print_keys], colwidth=5)
        misc.print_row([round(results[k], 3) for k in results_print_keys], colwidth=5)

        # -------- SAVE CHECKPOINT --------
        if args.save_ckpts:
            ckpt_save_dict = {"args": vars(args), "model_dict": algorithm.state_dict()}
            torch.save(ckpt_save_dict, os.path.join(ckpt_dir, f"{md5_fname}_{ms_name}.pkl"))

    # -------- SAVE ALL RESULTS --------
    # Create args_id without seed to allow the mean over seeds to be easily computed in collect_results.py
    args_no_seed = copy.deepcopy(args)
    delattr(args_no_seed, "seed")
    args_id = hashlib.md5(str(args_no_seed).encode('utf-8')).hexdigest()

    if (args.train_envs == 'gray') and (args.algorithm.lower() == "erm"):
        results["algorithm"] = "oracle"
    else:
        results["algorithm"] = args.algorithm.lower()
    results["seed"] = args.seed
    results["args_id"] = args_id
    results["args"] = vars(args_no_seed)

    with open(os.path.join(results_dir, f"{md5_fname}.jsonl"), 'a') as f:
        f.write(json.dumps(results, sort_keys=True) + "\n")
