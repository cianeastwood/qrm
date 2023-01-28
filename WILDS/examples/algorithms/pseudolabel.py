import torch
import torch.nn.functional as F
from models.initializer import initialize_model
from algorithms.ERM import ERM
from algorithms.single_model_algorithm import SingleModelAlgorithm
from scheduler import LinearScheduleWithWarmupAndThreshold
from wilds.common.utils import split_into_groups, numel
from configs.supported import process_pseudolabels_functions
import copy
from utils import load, move_to, detach_and_clone, collate_list, concat_input


class PseudoLabel(SingleModelAlgorithm):
    """
    PseudoLabel.
    This is a vanilla pseudolabeling algorithm which updates the model per batch and incorporates a confidence threshold.

    Original paper:
        @inproceedings{lee2013pseudo,
            title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
            author={Lee, Dong-Hyun and others},
            booktitle={Workshop on challenges in representation learning, ICML},
            volume={3},
            number={2},
            pages={896},
            year={2013}
            }
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out=d_out)
        model = model.to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.lambda_scheduler = LinearScheduleWithWarmupAndThreshold(
            max_value=config.self_training_lambda,
            step_every_batch=True, # step per batch
            last_warmup_step=0,
            threshold_step=config.pseudolabel_T2 * n_train_steps
        )
        self.schedulers.append(self.lambda_scheduler)
        self.scheduler_metric_names.append(None)
        self.confidence_threshold = config.self_training_threshold
        if config.process_pseudolabels_function is not None:
            self.process_pseudolabels_function = process_pseudolabels_functions[config.process_pseudolabels_function]
        # Additional logging
        self.logged_fields.append("pseudolabels_kept_frac")
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("consistency_loss")

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pseudo (Tensor): pseudolabels on the unlabeled batch, already thresholded 
                - unlabeled_y_pred (Tensor): model output on the unlabeled batch, already thresholded 
        """
        # Labeled examples
        x, y_true, metadata = batch
        n_lab = len(metadata)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'metadata': metadata
        }

        if unlabeled_batch is not None:
            x_unlab, metadata_unlab = unlabeled_batch
            x_unlab = move_to(x_unlab, self.device)
            g_unlab = move_to(self.grouper.metadata_to_group(metadata_unlab), self.device)
            results['unlabeled_metadata'] = metadata_unlab
            results['unlabeled_g'] = g_unlab

            # Special case for models where we need to pass in y:
            # we handle these in two separate forward passes
            # and turn off training to avoid errors when y is None
            # Note: we have to specifically turn training in the model off
            # instead of using self.train, which would reset the log
            if self.model.needs_y:
                self.model.train(mode=False)
                unlabeled_output = self.get_model_output(x_unlab, None)

                _, unlabeled_y_pseudo, pseudolabels_kept_frac, mask = self.process_pseudolabels_function(
                    unlabeled_output,
                    self.confidence_threshold
                )
                x_unlab = x_unlab[mask]

                self.model.train(mode=True)
                outputs = self.get_model_output(
                    torch.cat((x, x_unlab), dim=0),
                    collate_list([y_true, unlabeled_y_pseudo]),
                )
                unlabeled_y_pred = outputs[n_lab:]
            else:
                x_cat = concat_input(x, x_unlab)
                outputs = self.get_model_output(x_cat, None)
                unlabeled_output = outputs[n_lab:]
                unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, _ = self.process_pseudolabels_function(
                    unlabeled_output,
                    self.confidence_threshold
                )

            results['y_pred'] = outputs[:n_lab]
            results['unlabeled_y_pred'] = unlabeled_y_pred
            results['unlabeled_y_pseudo'] = detach_and_clone(unlabeled_y_pseudo)
        else:
            results['y_pred'] = self.get_model_output(x, y_true)
            pseudolabels_kept_frac = 0

        self.save_metric_for_logging(
            results, "pseudolabels_kept_frac", pseudolabels_kept_frac
        )
        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss.compute(
            results['y_pred'],
            results['y_true'],
            return_dict=False)
        # Pseudolabeled loss
        if 'unlabeled_y_pseudo' in results:
            loss_output = self.loss.compute(
                results['unlabeled_y_pred'],
                results['unlabeled_y_pseudo'],
                return_dict=False,
            )
            consistency_loss = loss_output * results['pseudolabels_kept_frac']
        else:
            consistency_loss = 0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "consistency_loss", consistency_loss
        )

        return classification_loss + self.lambda_scheduler.value * consistency_loss
