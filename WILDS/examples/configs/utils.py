import copy
from configs.algorithm import algorithm_defaults
from configs.model import model_defaults
from configs.scheduler import scheduler_defaults
from configs.data_loader import loader_defaults
from configs.datasets import dataset_defaults, split_defaults

def populate_defaults(config):
    """Populates hyperparameters with defaults implied by choices
    of other hyperparameters."""

    orig_config = copy.deepcopy(config)
    assert config.dataset is not None, 'dataset must be specified'
    assert config.algorithm is not None, 'algorithm must be specified'

    # Run oracle using ERM with unlabeled split
    if config.use_unlabeled_y:
        assert config.algorithm == 'ERM', 'Only ERM is currently supported for training on the true labels of unlabeled data.'
        assert config.unlabeled_split is not None, 'Specify an unlabeled split'
        assert config.dataset in ['amazon', 'civilcomments', 'fmow', 'iwildcam'], 'The unlabeled data in this dataset are truly unlabeled, and we do not have true labels for them.'

    # Validations
    if config.groupby_fields == ['from_source_domain']:
        if config.n_groups_per_batch is None:
            config.n_groups_per_batch = 1
        elif config.n_groups_per_batch != 1:
            raise ValueError(
                f"from_source_domain was specified for groupby_fields, but n_groups_per_batch "
                f"was {config.n_groups_per_batch}, when it should be 1."
            )

        if config.unlabeled_n_groups_per_batch is None:
            config.unlabeled_n_groups_per_batch = 1
        elif config.unlabeled_n_groups_per_batch != 1:
            raise ValueError(
                f"from_source_domain was specified for groupby_fields, but unlabeled_n_groups_per_batch "
                f"was {config.unlabeled_n_groups_per_batch}, when it should be 1."
            )

    if config.algorithm == 'DANN' and config.lr is not None:
        raise ValueError(
            "Cannot pass in a value for lr. For DANN, only dann_classifier_lr, dann_featurizer_lr "
            "and dann_discriminator_lr are valid learning rate parameters."
        )

    if config.additional_train_transform is not None:
        if config.algorithm == "NoisyStudent":
            raise ValueError(
                "Cannot pass in a value for additional_train_transform, NoisyStudent "
                "already has a default transformation for the training data."
            )

    if config.load_featurizer_only:
        if config.pretrained_model_path is None:
            raise ValueError(
                "load_featurizer_only cannot be set when there is no pretrained_model_path "
                "specified."
            )

    if config.dataset == 'globalwheat':
        if config.additional_train_transform is not None:
            raise ValueError(
                f"Augmentations not supported for detection dataset: {config.dataset}."
            )
        config.additional_train_transform = ''

        if config.algorithm == "NoisyStudent":
            if config.process_pseudolabels_function is None:
                config.process_pseudolabels_function = 'pseudolabel_detection'
            elif config.process_pseudolabels_function == 'pseudolabel_detection_discard_empty':
                raise ValueError(
                    f"Filtering out empty images when generating pseudo-labels for {config.algorithm} "
                    f"is not supported for detection."
                )

    # implied defaults from choice of dataset
    config = populate_config(
        config,
        dataset_defaults[config.dataset]
    )

    # implied defaults from choice of split
    if config.dataset in split_defaults and config.split_scheme in split_defaults[config.dataset]:
        config = populate_config(
            config,
            split_defaults[config.dataset][config.split_scheme]
        )

    # implied defaults from choice of algorithm
    config = populate_config(
        config,
        algorithm_defaults[config.algorithm]
    )

    # implied defaults from choice of loader
    config = populate_config(
        config,
        loader_defaults
    )
    # implied defaults from choice of model
    if config.model: config = populate_config(
        config,
        model_defaults[config.model],
    )

    # implied defaults from choice of scheduler
    if config.scheduler: config = populate_config(
        config,
        scheduler_defaults[config.scheduler]
    )

    # misc implied defaults
    if config.groupby_fields is None:
        config.no_group_logging = True
    config.no_group_logging = bool(config.no_group_logging)

    # basic checks
    required_fields = [
        'split_scheme', 'train_loader', 'uniform_over_groups', 'batch_size', 'eval_loader', 'model', 'loss_function',
        'val_metric', 'val_metric_decreasing', 'n_epochs', 'optimizer', 'lr', 'weight_decay',
        ]
    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    # data loader validations
    # we only raise this error if the train_loader is standard, and
    # n_groups_per_batch or distinct_groups are
    # specified by the user (instead of populated as a default)
    if config.train_loader == 'standard':
        if orig_config.n_groups_per_batch is not None:
            raise ValueError("n_groups_per_batch cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")
        if orig_config.distinct_groups is not None:
            raise ValueError("distinct_groups cannot be specified if the data loader is 'standard'. Consider using a 'group' data loader instead.")

    return config


def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = vars(config)
    for key, val in template.items():
        if not isinstance(val, dict): # config[key] expected to be a non-index-able
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")

        else: # config[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if kwargs_key not in d_config[key] or d_config[key][kwargs_key] is None:
                    d_config[key][kwargs_key] = kwargs_val
                elif d_config[key][kwargs_key] != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")
    return config
