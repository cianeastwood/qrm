from types import SimpleNamespace
import torch
import math
from wilds.common.utils import get_counts
from algorithms.ERM import ERM
from algorithms.AFN import AFN
from algorithms.DANN import DANN
from algorithms.groupDRO import GroupDRO
from algorithms.deepCORAL import DeepCORAL
from algorithms.IRM import IRM
from algorithms.fixmatch import FixMatch
from algorithms.pseudolabel import PseudoLabel
from algorithms.noisy_student import NoisyStudent
from algorithms.QRM import QRM_Normal, QRM_KDE
from configs.supported import algo_log_metrics, losses
from losses import initialize_loss

def initialize_algorithm(config, datasets, train_grouper, unlabeled_dataset=None):
    train_dataset = datasets['train']['dataset']
    train_loader = datasets['train']['loader']
    d_out = infer_d_out(train_dataset, config)

    # Other config
    n_train_steps = math.ceil(len(train_loader)/config.gradient_accumulation_steps) * config.n_epochs
    loss = initialize_loss(config.loss_function, config)
    metric = algo_log_metrics[config.algo_log_metric]

    if config.algorithm == 'ERM':
        algorithm = ERM(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'QRM_Normal':
        algorithm = QRM_Normal(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'QRM_KDE':
        algorithm = QRM_KDE(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'groupDRO':
        train_g = train_grouper.metadata_to_group(train_dataset.metadata_array)
        is_group_in_train = get_counts(train_g, train_grouper.n_groups) > 0
        algorithm = GroupDRO(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
            is_group_in_train=is_group_in_train)
    elif config.algorithm == 'deepCORAL':
        algorithm = DeepCORAL(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'IRM':
        algorithm = IRM(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'DANN':
        if unlabeled_dataset is not None:
            unlabeled_dataset = unlabeled_dataset['dataset']
            metadata_array = torch.cat(
                [train_dataset.metadata_array, unlabeled_dataset.metadata_array]
            )
        else:
            metadata_array = train_dataset.metadata_array

        groups = train_grouper.metadata_to_group(metadata_array)
        group_counts = get_counts(groups, train_grouper.n_groups)
        group_ids_to_domains = group_counts.tolist()
        domain_idx = 0
        for i, count in enumerate(group_ids_to_domains):
            if count > 0:
                group_ids_to_domains[i] = domain_idx
                domain_idx += 1
        group_ids_to_domains = torch.tensor(group_ids_to_domains, dtype=torch.long)
        algorithm = DANN(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
            n_domains = domain_idx,
            group_ids_to_domains=group_ids_to_domains,
        )
    elif config.algorithm == 'AFN':
        algorithm = AFN(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps
        )
    elif config.algorithm == 'FixMatch':
        algorithm = FixMatch(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'PseudoLabel':
        algorithm = PseudoLabel(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
    elif config.algorithm == 'NoisyStudent':
        if config.soft_pseudolabels:
            unlabeled_loss = initialize_loss("cross_entropy_logits", config)
        else:
            unlabeled_loss = loss
        algorithm = NoisyStudent(
            config=config,
            d_out=d_out,
            grouper=train_grouper,
            loss=loss,
            unlabeled_loss=unlabeled_loss,
            metric=metric,
            n_train_steps=n_train_steps)
    else:
        raise ValueError(f"Algorithm {config.algorithm} not recognized")

    return algorithm

def infer_d_out(train_dataset, config):
    # Configure the final layer of the networks used
    # The code below are defaults. Edit this if you need special config for your model.
    if train_dataset.is_classification:
        if train_dataset.y_size == 1:
            # For single-task classification, we have one output per class
            d_out = train_dataset.n_classes
        elif train_dataset.y_size is None:
            d_out = train_dataset.n_classes
        elif (train_dataset.y_size > 1) and (train_dataset.n_classes == 2):
            # For multi-task binary classification (each output is the logit for each binary class)
            d_out = train_dataset.y_size
        else:
            raise RuntimeError('d_out not defined.')
    elif train_dataset.is_detection:
        # For detection, d_out is the number of classes
        d_out = train_dataset.n_classes
        if config.algorithm in ['deepCORAL', 'IRM']:
            raise ValueError(f'{config.algorithm} is not currently supported for detection datasets.')
    else:
        # For regression, we have one output per target dimension
        d_out = train_dataset.y_size
    return d_out
