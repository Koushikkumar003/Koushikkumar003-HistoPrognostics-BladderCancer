import os
import torch
import numpy as np
import json
from timeit import default_timer as timer
import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    roc_auc_score, precision_score, recall_score, cohen_kappa_score,
    accuracy_score, f1_score
)
from losses_class import FocalTverskyLoss

class NMILTrainer():
    """Trainer for NMIL models, handling training, validation, and evaluation."""

    def __init__(self, dir_out, network, lr, id, early_stopping, scheduler,
                 virtual_batch_size, criterion, class_weights, loss_function,
                 tfl_alpha, tfl_gamma, opt_name):
        # Setup output directory
        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        self.network = network
        self.lr = lr
        self.id = id
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.class_weights = class_weights
        self.loss_function = loss_function
        self.tfl_alpha = tfl_alpha
        self.tfl_gamma = tfl_gamma
        self.opt_name = opt_name

        # Optimizer selection
        params = list(self.network.parameters())
        self.opt = torch.optim.SGD(params, lr=lr) if opt_name == 'sgd' else torch.optim.Adam(params, lr=lr)

        # Loss function selection
        if loss_function == 'tversky':
            self.L = FocalTverskyLoss(alpha=tfl_alpha, beta=1-tfl_alpha, gamma=tfl_gamma).cuda()
        else:
            self.L = torch.nn.BCEWithLogitsLoss(weight=class_weights).cuda()

        # Best criterion initialization
        self.best_criterion = 1e6 if criterion == 'loss' else 0

        # Tracking metrics
        self.L_lc, self.Lce_lc_val = [], []
        self.macro_auc_lc_val, self.macro_auc_lc_train = [], []
        self.f1_lc_val, self.f1_lc_train = [], []
        self.k2_lc_val, self.k2_lc_train = [], []

    def train(self, train_generator, val_generator, test_generator, epochs):
        """Main training loop."""
        self.epochs = epochs
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.network.cuda()
        self.init_time = timer()

        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            self.L_epoch = 0
            self.preds_train, self.refs_train = [], []

            # Learning rate scheduler
            if self.scheduler and (i_epoch + 1 == self.best_epoch + 5):
                for g in self.opt.param_groups:
                    g['lr'] = self.lr / 2

            # Training loop
            for i_iter, (X, Y, clinical_data, region_info) in enumerate(train_generator):
                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()
                clinical_data = torch.tensor(clinical_data).cuda().float()
                region_info = torch.tensor(region_info).cuda().float()

                self.network.train()
                Yhat, _ = self.network(X, clinical_data, region_info)
                Lce = self.L(Yhat, torch.squeeze(Y))
                (Lce / self.virtual_batch_size).backward()

                # Weight update
                if ((i_epoch + 1) % self.virtual_batch_size) == 0:
                    self.opt.step()
                    self.opt.zero_grad()

                # Save predictions for metrics
                self.preds_train.append(Yhat.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())
                self.L_epoch += Lce.cpu().detach().numpy() / len(train_generator)

            # Epoch end: run validation, save best, plot, etc.
            self.on_epoch_end()

            # Early stopping condition
            if self.early_stopping and (i_epoch + 1 >= self.best_epoch + 30) and (i_epoch + 1 >= 100):
                break

    def on_epoch_end(self):
        """Handle end-of-epoch metrics, validation, and model saving."""
        # Compute train AUC
        macro_auc_train = roc_auc_score(
            np.squeeze(np.array(self.refs_train)), np.array(self.preds_train), multi_class='ovr'
        ) if not np.isnan(np.sum(self.preds_train)) else 0.5
        self.macro_auc_lc_train.append(macro_auc_train)
        self.L_lc.append(self.L_epoch)

        # Compute train metrics
        _, _, acc_train, f1_train, k2_train = self.test_bag_level_classification(self.train_generator)
        self.f1_lc_train.append(f1_train)
        self.k2_lc_train.append(k2_train)

        # Validation metrics
        Lce_val, macro_auc_val, acc_val, f1_val, k2_val = self.test_bag_level_classification(self.val_generator)
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.f1_lc_val.append(f1_val)
        self.k2_lc_val.append(k2_val)

        # Save metrics
        metrics = {
            'epoch': self.i_epoch + 1,
            'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
            'AUCval': np.round(self.macro_auc_lc_val[-1], 4),
            'F1val': np.round(self.f1_lc_val[-1], 4),
            'K2val': np.round(self.k2_lc_val[-1], 4)
        }
        with open(self.dir_results + self.id + 'metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        # Save best model based on chosen criterion
        if (self.i_epoch + 1) > 5:
            current_metric = {
                'auc': self.macro_auc_lc_val[-1],
                'k2': self.k2_lc_val[-1],
                'f1': self.f1_lc_val[-1],
                'loss': self.L_lc[-1]
            }[self.criterion]
            is_better = current_metric > self.best_criterion if self.criterion != 'loss' else current_metric < self.best_criterion
            if is_better:
                self.best_criterion = current_metric
                self.best_epoch = self.i_epoch + 1
                torch.save(self.network, self.dir_results + self.id + 'network_weights_best.pth')

        # Save and plot every 5 epochs
        if (self.i_epoch + 1) % 5 == 0:
            torch.save(self.network, self.dir_results + self.id + 'network_weights.pth')
            self.plot_learning_curves()

        # Final evaluation at training end or early stopping
        if (self.epochs == self.i_epoch + 1) or (
            self.early_stopping and (self.i_epoch + 1 >= self.best_epoch + 30) and self.i_epoch + 1 >= 100
        ):
            self.network = torch.load(self.dir_results + self.id + 'network_weights_best.pth')
            self.plot_learning_curves()
            # Validation and test metrics
            _, macro_auc_val, acc_val, f1_val, k2_val = self.test_bag_level_classification(self.val_generator)
            _, macro_auc_test, acc_test, f1_test, k2_test = self.test_bag_level_classification(self.test_generator)
            # Instance-level metrics (if applicable)
            acc, f1, k2 = 0.0, 0.0, 0.0
            if not self.test_generator.dataset.only_clinical and not (self.network.aggregation in ['mean', 'max']):
                X = self.test_generator.dataset.X[self.test_generator.dataset.y_instances[:] != -1, :]
                Y = self.test_generator.dataset.y_instances[self.test_generator.dataset.y_instances[:] != -1]
                clinical_data = self.test_generator.dataset.clinical_data
                acc, f1, k2 = self.test_instance_level_classification(X, Y, clinical_data, self.test_generator.dataset.classes)
            # Save final metrics
            metrics = {
                'epoch': self.best_epoch, 'AUCtest': np.round(macro_auc_test, 4),
                'acc_test': np.round(acc_test, 4), 'f1_test': np.round(f1_test, 4), 'k2_test': np.round(k2_test, 4),
                'AUCval': np.round(macro_auc_val, 4), 'acc_val': np.round(acc_val, 4),
                'f1_val': np.round(f1_val, 4), 'k2_val': np.round(k2_val, 4),
                'acc_ins': np.round(acc, 4), 'f1_ins': np.round(f1, 4), 'k2_ins': np.round(k2, 4),
            }
            with open(self.dir_results + self.id + 'best_metrics.json', 'w') as fp:
                json.dump(metrics, fp)
            print(metrics)
        self.metrics = metrics

