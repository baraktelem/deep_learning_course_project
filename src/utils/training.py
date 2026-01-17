from typing import Any, Callable, Dict, Optional, Tuple

# Standard Libraries
import time
import numpy as np
from pathlib import Path

# PyTorch
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Kymatio
from kymatio.torch import Scattering2D

# Kornia
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
from torch.utils.data import random_split

# Project Libraries
from src.utils.config import *


def train_model(
    model : nn.Module, 
    trainloader : DataLoader, 
    valloader : DataLoader, 
    optimizer : torch.optim.Optimizer, 
    scheduler : torch.optim, 
    criterion, 
    n_epochs : int, 
    device, 
    # checkpoint_dir, 
    experiment_name : str,
    model_name : str ="Model", 
    do_augmentations : bool = True,
    aug_list : Optional[Callable] = None,
    val_accuracy_storing_threshold : float = 50,
    print_progress_every : int = 1,
    DEBUG : bool = False
) -> Dict[str, Any]:
    """
    Trains a model and returns the stats dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the training statistics.
            - total_training_time: The total time taken to train the model.
            - loss: A list of the loss values for each epoch.
            - time_per_epoch: A list of the time taken to train the model for each epoch.
            - total_time_per_epoch: A list of the total time taken to train the model for each epoch.
            - val_accuracy: A list of the validation accuracy for each epoch.
            - max_val_accuracy: The maximum validation accuracy achieved.
            - allocated_memory: A list of the allocated memory for each epoch.
            - reserved_memory: A list of the reserved memory for each epoch.
    """
    
    # Initialize Stats
    stats = {
        'total_training_time': 0,
        'loss': [],
        'time_per_epoch': [],
        'total_time_per_epoch': [],
        'val_accuracy': [],
        'max_val_accuracy': 0,
        'allocated_memory': [],
        'reserved_memory': []
    }

    # checkpoint_path = Path(checkpoint_dir) # Ensure it's a Path object
    # checkpoint_path.mkdir(parents=True, exist_ok=True) # Create dir if missing
    if DEBUG:
        checkpoint_file = CHECKPOINTS_PATH / f"{experiment_name}_{model_name}_DEBUG.pth"
        n_epochs = 1 # DEBUG mode only runs for 1 epoch
    else:
        checkpoint_file = CHECKPOINTS_PATH / f"{experiment_name}_{model_name}.pth"

    if aug_list is None:
        aug_list = get_augmentations(device)

    print(f"\n=== Starting Training: {model_name} with {n_epochs} epochs ===")

    start_time = time.time()
    for epoch in range(n_epochs):
        model.train()
        iteration_losses = []
        epoch_start_time = time.time()
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if do_augmentations:
                inputs = aug_list(inputs)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            iteration_losses.append(loss.item())

        scheduler.step()
        epoch_end_time = time.time()

        # Validation
        # model.eval()
        val_accuracy = calculate_accuracy(model, valloader, device)

        # Track stats
        stats['loss'].append(np.mean(iteration_losses))
        stats['val_accuracy'].append(val_accuracy)
        stats['allocated_memory'].append(torch.cuda.memory_allocated())
        stats['reserved_memory'].append(torch.cuda.memory_reserved())
        stats['time_per_epoch'].append(epoch_end_time - epoch_start_time)
        stats['total_time_per_epoch'].append(time.time() - start_time)

        # Store best model
        if val_accuracy > stats['max_val_accuracy']:
            if val_accuracy > val_accuracy_storing_threshold:
                stats['max_val_accuracy'] = val_accuracy
                # save_file = checkpoint_path / f"{model_name}_finetune_exp.pth"
                
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                    'acc': val_accuracy
                }
                # torch.save(state, save_file)
                torch.save(state, checkpoint_file)
                if (epoch % print_progress_every) == 0:
                    print(f"    --> New Best Saved: {val_accuracy:.2f}%")

        if DEBUG:
            print('==> Saving model ... DEBUG')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
                'acc':val_accuracy
            }
            # save_path = checkpoint_dir / f"{model_name}_finetune_exp.pth"
            # torch.save(state, save_path)
            torch.save(state, checkpoint_file)
            
        # Print progress
        if (epoch % print_progress_every) == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {stats['loss'][-1]:.3f} | Test Acc: {stats['val_accuracy'][-1]:.2f}%")

    stats['total_training_time'] = time.time() - start_time
    print(f"=== Finished {model_name}. Total Time: {stats['total_training_time']:.1f}s ===")
    
    return stats

def get_normalizer(device: torch.device) -> Callable:
    """
    Returns a normalizer function that can be used to normalize the images.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
    return K.Normalize(mean=mean, std=std)

def get_augmentations(device: torch.device) -> Callable:
    """
    Returns a list of augmentations that can be used to augment the images.
    """
    return AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.2),
        K.RandomResizedCrop(size=(32,32), scale=(0.7, 1.0), p=0.5),
        get_normalizer(device),
        same_on_batch=False
    ).to(device)

def calculate_accuracy(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    normalize: Callable=None
) -> float:
    model.eval()
    total_correct = 0
    total_images = 0
    if normalize is None:
        normalize = get_normalizer(device)

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            images = normalize(images)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=-1)
            total_images += labels.size(0)
            total_correct += (predictions == labels).sum().item()

    model_accuracy = total_correct / total_images * 100
    return model_accuracy

def get_all_predictions(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    normalize: Callable=None
) -> Tuple[Tensor, Tensor]:
    """
    Runs the model on the full dataloader and returns:
    - all_preds: Tensor of shape (N,) containing model predictions
    - all_labels: Tensor of shape (N,) containing ground truth labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    if normalize is None:
        normalize = get_normalizer(device)

    # Ensure we are in evaluation mode
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. Normalize
            images = normalize(images)
            
            # 2. Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=-1)
            
            # 3. Store results on CPU to save GPU memory
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    # Concatenate into single long tensors
    return torch.cat(all_preds), torch.cat(all_labels)
    