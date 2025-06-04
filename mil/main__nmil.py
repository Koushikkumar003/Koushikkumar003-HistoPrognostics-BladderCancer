import os
import pandas as pd
import numpy as np
import json
import random
import torch
import argparse

from datasets_nmil import *
from model_nmil import *
from train_nmil import *

# Clear GPU memory and set random seeds for reproducibility
torch.cuda.empty_cache()
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def main(args):
    """
    Main NMIL training loop: loads data, trains for several iterations, saves metrics.
    """
    # Load patient splits from Excel
    wsi_df = pd.read_excel(args.patient_set_split)
    train_wsi = wsi_df[wsi_df['Set'] == 'Train']['WSI_OLD'].to_list()
    val_wsi = wsi_df[wsi_df['Set'] == 'Val']['WSI_OLD'].to_list()
    test_wsi = wsi_df[wsi_df['Set'] == 'Test']['WSI_OLD'].to_list()
    metrics = []

    for i in range(args.iterations):
        print(f"[Iteration {i+1}/{args.iterations}]")
        # Create data generators for train/val/test
        train_gen = create_data_generator(args, train_wsi, shuffle=True, max_instances=args.max_instances)
        val_gen = create_data_generator(args, val_wsi, shuffle=False, max_instances=1000000)
        test_gen = create_data_generator(args, test_wsi, shuffle=False, max_instances=1000000)

        # Initialize NMIL model
        network = NMILArchitecture(
            args.classes, aggregation=args.aggregation,
            only_images=args.only_images, only_clinical=args.only_clinical,
            clinical_classes=args.clinical_data,
            neurons_1=args.neurons_1, neurons_2=args.neurons_2, neurons_3=args.neurons_3,
            neurons_att_1=args.neurons_att_1, neurons_att_2=args.neurons_att_2,
            dropout_rate=args.dropout_rate
        )

        # Compute class weights if enabled
        if args.class_weights_enable:
            pos_ratio = sum(train_gen.dataset.y_instances) / len(train_gen.dataset.y_instances)
            class_weights = torch.softmax(torch.tensor([1, 1/pos_ratio]), dim=0) * 2
        else:
            class_weights = torch.ones([2])

        # Initialize trainer and train
        trainer = NMILTrainer(
            dir_out=f"{args.dir_results}{args.experiment_name}/",
            network=network, lr=args.lr, id=f"{i}_",
            early_stopping=args.early_stopping, scheduler=args.scheduler,
            virtual_batch_size=args.virtual_batch_size,
            criterion=args.criterion, class_weights=class_weights,
            loss_function=args.loss_function,
            tfl_alpha=args.tfl_alpha, tfl_gamma=args.tfl_gamma,
            opt_name=args.opt_name
        )
        trainer.train(train_gen, val_gen, test_gen, epochs=args.epochs)
        metrics.append(list(trainer.metrics.values())[1:])  # Skip first metric (loss)

    save_metrics(metrics, args)

def create_data_generator(args, wsi_list, shuffle, max_instances):
    """Create NMIL data generator for a WSI list."""
    dataset = NMILDataset(
        args.dir_images, args.dir_embeddings, wsi_list, args.classes,
        clinical_dataframe=args.clinical_dataframe,
        data_augmentation=args.data_augmentation,
        channel_first=True,
        only_clinical=args.only_clinical,
        clinical_parameters=args.clinical_data
    )
    return NMILDataGenerator(dataset, batch_size=1, shuffle=shuffle, max_instances=max_instances)

def save_metrics(metrics, args):
    """Save mean/std metrics if multiple iterations, else single run metrics."""
    metrics = np.squeeze(np.array(metrics))
    if args.iterations > 1:
        mu = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0)
        info = (f"AUC_test={mu[0]:.4f}({std[0]:.4f}); "
                f"AUC_val={mu[4]:.4f}({std[4]:.4f}); "
                f"acc={mu[5]:.4f}({std[5]:.4f}); "
                f"f1-score={mu[6]:.4f}({std[6]:.4f}); "
                f"kappa={mu[7]:.4f}({std[7]:.4f})")
    else:
        info = (f"AUC_test={metrics[0]:.4f}; AUC_val={metrics[4]:.4f}; "
                f"acc={metrics[5]:.4f}; f1-score={metrics[6]:.4f}; "
                f"kappa={metrics[7]:.4f}")
    # Write metrics to file
    with open(f"{args.dir_results}{args.experiment_name}/method_metrics.txt", 'w') as f:
        f.write(info)
    print(f"Final Results: {info}")

def setup_args():
    """Parse command line arguments for NMIL training."""
    parser = argparse.ArgumentParser(description="NMIL Training Pipeline")
    # Add all needed arguments (see original for full list)
    parser.add_argument("--dir_images", default='extracted_tiles_of_.../')
    parser.add_argument("--dir_embeddings", default='../contrastive/data/embeddings_.../')
    parser.add_argument("--patient_set_split", default='patient_set_split.xlsx')
    parser.add_argument("--clinical_dataframe", default='clinical_dataframe.xlsx')
    parser.add_argument("--dir_results", default='results/')
    parser.add_argument("--experiment_name", default="nmil")
    parser.add_argument("--classes", default=['Responsive', 'Failure'])
    parser.add_argument("--clinical_data", default=['Yrs_age', 'Gender', 'Smoking'])
    parser.add_argument("--only_images", default=True)
    parser.add_argument("--only_clinical", default=False)
    parser.add_argument("--aggregation", default="attentionMIL")
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--max_instances", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--opt_name", default="sgd")
    parser.add_argument("--loss_function", default="tversky")
    parser.add_argument("--neurons_1", default=1024, type=int)
    parser.add_argument("--neurons_2", default=4096, type=int)
    parser.add_argument("--neurons_3", default=2048, type=int)
    parser.add_argument("--neurons_att_1", default=1024, type=int)
    parser.add_argument("--neurons_att_2", default=4096, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--criterion", default='auc')
    parser.add_argument("--early_stopping", default=True)
    parser.add_argument("--scheduler", default=True)
    parser.add_argument("--class_weights_enable", default=True)
    parser.add_argument("--virtual_batch_size", default=1, type=int)
    parser.add_argument("--tfl_alpha", default=0.9, type=float)
    parser.add_argument("--tfl_gamma", default=2, type=float)
    parser.add_argument("--data_augmentation", default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    # Create results directory and save arguments
    os.makedirs(f"{args.dir_results}{args.experiment_name}", exist_ok=True)
    with open(f"{args.dir_results}{args.experiment_name}/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"Starting NMIL training with {args.iterations} iterations...")
    print(f"Results will be saved to: {args.dir_results}{args.experiment_name}/")
    main(args)
