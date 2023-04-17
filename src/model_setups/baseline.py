from helper_functions.model_helper_functions.model_setup_helper_functions import build_transform_module_lists_dict, get_loss_function, get_model, get_optimizer, get_train_and_validation_dataset_and_dataloader
from trainer.trainer import train_one_epoch
import util.local_config as local_config
import config.global_config as global_config
import wandb
import datetime
from os.path import join
import pandas as pd
import numpy as np
import torch

from validator.validator import validate_one_epoch

def run():
    # Set the wandb config object
    config = {
        'epochs': 20,
        'training_batch_size': 32,
        'learning_rate': 1e-3,
        'sample_dimensions': (250, 500),
        'model_id': 'resnet_18_untrained',
        'model_config_dict': {
            'stored_weights': '',
            'num_input_channels': 3,
            'num_outputs': 0,
            'dropout': 0.5
        },
        'optimizer': 'adamw',
        'loss_fn': 'cross_entropy',
        'labels_for_training_file_ids': ['train_'],
        'labels_for_validation_file_ids': ['dev'],
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
    
    '''
    # Init the wandb run session
    wandb.init(
        project="cvnlp_speech_sentiment_recognition", 
        name=run_name,
        config=config,
    )
    '''
    

    # Clear gpu cache
    if local_config.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # Construct the transform module lists as a dict
    transform_dict = build_transform_module_lists_dict(
        augmentation_config_dict = config['augmentation_config_dict'],
        sample_dimensions = config['sample_dimensions']
    )

    # Get the datasets and dataloaders
    training_dataset, training_dataloader, validation_dataset, validation_dataloader = get_train_and_validation_dataset_and_dataloader(
        training_data_ids=config['labels_for_training_file_ids'],
        validation_data_ids=config['labels_for_validation_file_ids'],
        labels_and_data_paths_dict=global_config.LABELS_AND_DATA_PATHS_DICT,
        transform_dict=transform_dict,
        train_batch_size=config['training_batch_size'],
        train_num_workers=global_config.TRAINING_NUM_WORKERS,
        validation_batch_size=global_config.VALIDATION_BATCH_SIZE,
        validation_num_workers=global_config.VALIDATION_NUM_WORKERS,
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
        mean_val_loss_this_epoch = validate_one_epoch(
            model,
            loss_fn,
            validation_dataloader
        )

        # Save the model
        save_model_weights(
            model=model,
            model_checkpoint_path=run_checkpont_path,
            save_as_artifact=True,
            artifact_name=run_name,
            model_validation_score=mean_val_loss_this_epoch
        )

        '''
        # Log this epoch
        wandb.log({
            'mean_train_loss': mean_train_loss_this_epoch,
            'mean_val_loss': mean_val_loss_this_epoch,
            "specific_sample_performance_plot": wandb.Image(specific_sample_plot)
        })
        '''

        print(f'epoch: {epoch}, mean_train_loss: {mean_train_loss_this_epoch}, mean_val_loss: {mean_val_loss_this_epoch}')
    
    # Finish wandb logging
    #wandb.finish()



