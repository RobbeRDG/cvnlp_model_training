from tqdm import tqdm
import util.local_config as local_config
import torch
from torch import autocast
import numpy as np

def train_one_epoch(
        model,
        optimizer,
        loss_fn,
        scaler,
        dataloader
    ):
    losses_this_epoch = np.array([])
    for idx, (inputs, labels, sample_metadata) in enumerate(tqdm(dataloader)):
        # Send the inputs and labels to device
        inputs, labels = inputs.to(local_config.DEVICE, dtype=torch.float), labels.to(local_config.DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type=local_config.DEVICE.type):
            # Run the forward pass  
            outputs = model(inputs)  

            # Calculate the loss
            loss = loss_fn(outputs, labels)

        # Update the total loss
        losses_this_epoch = np.append(losses_this_epoch, loss.item())

        # Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Calculate the mean loss in this epoch
    mean_epoch_loss = losses_this_epoch.mean()

    # Return the mean epoch loss
    return mean_epoch_loss