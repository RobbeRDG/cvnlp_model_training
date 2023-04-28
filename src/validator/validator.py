import torch
from tqdm import tqdm
import util.local_config as local_config
import numpy as np


def validate_one_epoch(
        model,
        loss_fn,
        dataloader
    ):
    # Set the evaluation metrics
    losses_this_epoch = np.array([])

    # No gradients needed during validation
    with torch.no_grad():
        for idx, (inputs, waveforms, labels, sample_metadata) in enumerate(tqdm(dataloader, leave=True)):
            # Send the input and label to device
            inputs, labels = inputs.to(local_config.DEVICE, dtype=torch.float), labels.to(local_config.DEVICE)

            # Run the forward pass.
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels).mean()

            # Add the loss to the total
            losses_this_epoch = np.append(losses_this_epoch, loss.item())

    # Calculate the mean loss in this epoch
    mean_epoch_loss = losses_this_epoch.mean()

    return mean_epoch_loss