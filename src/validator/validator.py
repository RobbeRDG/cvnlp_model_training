import torch
from tqdm import tqdm
import util.local_config as local_config
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import json

def validate_one_epoch(
        model,
        loss_fn,
        dataloader
    ):
    # Set the evaluation metrics
    dialogue_ids_this_epoch = np.array([])
    utterance_ids_this_epoch = np.array([])
    ground_truths_this_epoch = np.array([])
    model_predictions_this_epoch = np.array([])
    model_outputs_this_epoch = np.array([])
    losses_this_epoch = np.array([])

    # No gradients needed during validation
    with torch.no_grad():
        for idx, (inputs, labels, sample_metadata) in enumerate(tqdm(dataloader, leave=True)):
            # Send the input and label to device
            inputs = inputs.to(local_config.DEVICE, dtype=torch.float)
            labels = labels.to(local_config.DEVICE)

            # Run the forward pass.
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels).mean()

            # Add the loss to the total
            losses_this_epoch = np.append(losses_this_epoch, loss.item())

            # Get the outputs as an array of json strings
            outputs_json_strings = []
            for idx, output in enumerate(outputs.detach().cpu().numpy()):
                outputs_json_strings.append(json.dumps(output.tolist()))

            # Get the model predictions and ground truths
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            ground_truths = torch.argmax(labels, dim=1).cpu().numpy()

            # Add the model predictions, ground truths, dialogue id and utterance id to the total
            dialogue_ids_this_epoch = np.append(dialogue_ids_this_epoch, sample_metadata[:, 0].cpu().numpy())
            utterance_ids_this_epoch = np.append(utterance_ids_this_epoch, sample_metadata[:, 1].cpu().numpy())
            ground_truths_this_epoch = np.append(ground_truths_this_epoch, ground_truths)
            model_predictions_this_epoch = np.append(model_predictions_this_epoch, preds)
            model_outputs_this_epoch = np.append(model_outputs_this_epoch, outputs_json_strings)
            

    # Calculate the mean loss in this epoch
    mean_epoch_loss = losses_this_epoch.mean()

    # Create a dataframe with the relevant results
    results_df = pd.DataFrame({
        'dialogue_id': dialogue_ids_this_epoch,
        'utterance_id': utterance_ids_this_epoch,
        'ground_truth': ground_truths_this_epoch,
        'model_prediction': model_predictions_this_epoch,
        'model_outputs': model_outputs_this_epoch
    })

    return mean_epoch_loss, results_df