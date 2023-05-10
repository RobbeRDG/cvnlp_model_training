import util.local_config as local_config
from helper_functions.model_evaluation_helper_functions.model_evaluation_metrics_helper_functions import calculate_all_evaluation_metrics
from helper_functions.model_helper_functions.model_storage_helper_functions import save_model_weights
from helper_functions.model_helper_functions.model_setup_helper_functions import build_transform_module_lists_dict, get_loss_function, get_model, get_optimizer, get_train_and_validation_dataset_and_dataloader
from trainer.trainer import train_one_epoch
import global_utils.global_config as global_config
import wandb
import datetime
from os.path import join
import pandas as pd
import numpy as np
import torch
from validator.validator import validate_one_epoch
import os

def train():
    # Set the wandb config object
    config = {
        'epochs': 10,
        'training_batch_size': 16,
        'learning_rate': 1e-4,
        'model_id': 'wav2vec2',
        'model_config_dict': {
            'dropout_probability': 0.0,
            'pooling_strategy': 'mean',
            'freeze_feature_extractor': False,
            'unfreeze_encoder_layers_count': 12, # Does not work
            'freeze_feature_projection': True
        },
        'class_weights_dict': None,
        'optimizer': 'adamw',
        'loss_fn': 'cross_entropy',
        'labels_for_training_file_ids': ['train'],
        'labels_for_validation_file_ids': ['test'],
        'augmentation_config_dict': {
            'train_input_augmentation_ids': [],
            'train_target_augmentation_ids': [],
            'train_both_augmentation_ids': [],
            'validation_input_augmentation_ids': [],
            'validation_target_augmentation_ids': [],
            'validation_both_augmentation_ids': []
        }
    }

    # Set the run name
    now = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    run_name = f'baseline__{now}'

    # Set the run checpoint path to store the models
    run_checkpont_path = join(local_config.CHECKPOINT_BASE_PATH, f'{run_name}.pth')

    # Set the bases validation results path for this run and create the folder
    run_validation_results_base_path = join(local_config.VALIDATION_RESULTS_BASE_PATH, run_name)
    os.makedirs(run_validation_results_base_path, exist_ok=True)
    
    # Init the wandb run session
    wandb.init(
        project="cvnlp_speech_sentiment_recognition", 
        name=run_name,
        config=config,
    )

    # Clear gpu cache
    if local_config.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # Construct the transform module lists as a dict
    transform_dict = build_transform_module_lists_dict(
        augmentation_config_dict = config['augmentation_config_dict']
    )

    # Get the datasets and dataloaders
    training_dataset, training_dataloader, validation_dataset, validation_dataloader = get_train_and_validation_dataset_and_dataloader(
        training_data_ids=config['labels_for_training_file_ids'],
        validation_data_ids=config['labels_for_validation_file_ids'],
        train_batch_size=config['training_batch_size'],
        class_weights_dict=config['class_weights_dict'],
        project_base_path=local_config.BASE_PATH
    )

    # Get the model
    model = get_model(
        model_id=config['model_id'],
        model_config_dict=config['model_config_dict']
    )
    model = model.to(local_config.DEVICE)

    # Get the optimizer
    optimizer = get_optimizer(
        optimizer_id=config['optimizer'],
        model=model,
        learning_rate=config['learning_rate']
    )

    # Get the loss function
    loss_fn = get_loss_function(
        loss_function_id=config['loss_fn']
    )

    # Set the gradient scaler
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # Training loop
    for epoch in range(config['epochs']):       
        # Set the model in training mode
        model.train()
        
        # Train the model
        mean_train_loss_this_epoch = train_one_epoch(
            model,
            optimizer,
            loss_fn,
            grad_scaler,
            training_dataloader
        )
        
        # Set the model in evaluation mode
        model.eval()
        
        # Validate the model
        mean_val_loss_this_epoch, val_results_df = validate_one_epoch(
            model,
            loss_fn,
            validation_dataloader
        )

        # Calculate the evaluation metrics
        val_weighted_f1_score, val_accuracy, val_weighted_recall, confusion_matrix = calculate_all_evaluation_metrics(
            ground_truths=val_results_df['ground_truth'],
            model_predictions=val_results_df['model_prediction']
        )

        # Save the validation results
        val_results_df.to_csv(join(run_validation_results_base_path, f'{epoch}_val_results.csv'), index=False)

        # Save the model
        save_model_weights(
            model=model,
            model_checkpoint_path=run_checkpont_path,
            save_as_artifact=True,
            artifact_name=run_name,
            model_validation_score=mean_val_loss_this_epoch
        )

        # Log this epoch
        wandb.log({
            'mean_train_loss': mean_train_loss_this_epoch,
            'mean_val_loss': mean_val_loss_this_epoch,
            'val_weighted_f1_score': val_weighted_f1_score,
            'accuracy': val_accuracy,
            'weighted_recall': val_weighted_recall,
            'val_results_df': wandb.Table(dataframe=val_results_df)
        })

        print(f'epoch: {epoch}, mean_train_loss: {mean_train_loss_this_epoch}, mean_val_loss: {mean_val_loss_this_epoch}, val_weighted_f1_score: {val_weighted_f1_score}, val_accuracy: {val_accuracy}, val_weighted_recall: {val_weighted_recall}')
        
    # Finish wandb logging
    wandb.finish()

def run():
    # Run the training
    train()



