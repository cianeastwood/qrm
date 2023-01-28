algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,     # When running ERM + data augmentation
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 1.,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 100.,
        'irm_penalty_anneal_iters': 500,
    },
    'DANN': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'AFN': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'use_hafn': False,
        'afn_penalty_weight': 0.01,
        'safn_delta_r': 1.0,
        'hafn_r': 1.0,
        'additional_train_transform': 'randaugment',    # Apply strong augmentation to labeled & unlabeled examples
        'randaugment_n': 2,
    },
    'FixMatch': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled examples
    },
    'PseudoLabel': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_lambda': 1,
        'self_training_threshold': 0.7,
        'pseudolabel_T2': 0.4,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'NoisyStudent': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'noisystudent_add_dropout': True,
        'noisystudent_dropout_rate': 0.5,
        'scheduler': 'FixMatchLR',
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'CVaR': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'cvar_beta': 0.1,
        'cvar_n_steps': 10,
        'cvar_step_size': 0.001,
    },
    'QRM_Normal': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'var_quantile': 0.75,
    },
    'QRM_KDE': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'var_quantile': 0.75,
    }
}
