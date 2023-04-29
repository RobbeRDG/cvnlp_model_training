import util.local_config as local_config
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

def run():
    # Set the wandb config object
    config = {
        'epochs': 10,
        'training_batch_size': 32,
        'learning_rate': 1e-5,
        'model_id': 'wav2vec2',
        'model_config_dict': {
            'dropout_probability': 0.7,
            'pooling_strategy': 'mean',
            'freeze_feature_extractor': True
        },
        'class_weights_dict': global_config.EMOTION_WEIGHTS_DICT,
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
    run_name = f'baseline_weighted__{now}'

    # Set the run checpoint path to store the models
    run_checkpont_path = join(local_config.BASELINE_CHECKPOINT_BASE_PATH, f'{run_name}.pth')
    
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
        mean_val_loss_this_epoch, val_weighted_f1_score, val_results_df = validate_one_epoch(
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

        # Log this epoch
        wandb.log({
            'mean_train_loss': mean_train_loss_this_epoch,
            'mean_val_loss': mean_val_loss_this_epoch,
            'val_weighted_f1_score': val_weighted_f1_score,
            'val_results_df': wandb.Table(dataframe=val_results_df)
        })

        print(f'epoch: {epoch}, mean_train_loss: {mean_train_loss_this_epoch}, mean_val_loss: {mean_val_loss_this_epoch}, val_weighted_f1_score: {val_weighted_f1_score}')
    
    # Finish wandb logging
    wandb.finish()



