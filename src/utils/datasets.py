from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from collections import Counter 

from src.utils.config import *

# Default transform: just convert to tensor
DEFAULT_TRANSFORM = transforms.ToTensor()


class CIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset with pre-configured defaults.
    
    Args:
        root: Data directory. Defaults to DATA_PATH from config.
        download: Whether to download. Defaults to True.
        transform: Transform to apply. Defaults to ToTensor().
        **kwargs: Other args passed to CIFAR10 (train, etc.)
    """
    def __init__(
        self,
        root: Optional[Union[str, Path]]=None,  # Defaults to DATA_PATH
        download: bool=True,
        transform: Optional[Callable]=DEFAULT_TRANSFORM,
        **kwargs
    ):
        if root is None:
            root = DATA_PATH
        super(CIFAR10, self).__init__(root=root, download=download, transform=transform, **kwargs)


class FilteredCIFAR10(CIFAR10):
    """CIFAR10 dataset that loads only specified classes.
    
    Args:
        keep_classes: List of class indices to keep (e.g., [0, 1, 2] for first 3 classes)
        **kwargs: Other args passed to CIFAR10 (root, download, train, transform, etc.)
    """
    def __init__(self, keep_classes: Optional[List[int]]=None, **kwargs):
        super(FilteredCIFAR10, self).__init__(**kwargs)
        
        # Filter the data and targets
        if keep_classes is not None:
            # Convert keep_classes to a set for faster lookup
            keep_set = set(keep_classes)
            
            # Find indices of images that belong to our selected classes
            indices = [i for i, label in enumerate(self.targets) if label in keep_set]
            
            # Overwrite the data and targets with only the selected indices
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices].tolist()
            
            # Create a mapping to make labels 0, 1, 2, 3, 4
            self.class_map = {c: i for i, c in enumerate(keep_classes)}
            
            # Apply the mapping to all targets
            self.targets = [self.class_map[t] for t in self.targets]

    # Override __getitem__ to ensure we return the transformed image and mapped label
    def __getitem__(self, index):
        img, target = super(FilteredCIFAR10, self).__getitem__(index)
        return img, target


def get_cifar10_splits(
    keep_classes: Optional[List[int]] = None,
    # train_val_split: float = 0.8,
    n_samples_per_class_train: int = None,
    n_samples_per_class_val: int = None,
    seed: int = SEED,
    **kwargs
) -> Tuple[Subset, Subset, Union[CIFAR10, FilteredCIFAR10]]:
    """Get train, validation, and test splits for CIFAR10.
    
    Args:
        keep_classes: List of class indices to keep. None for all classes.
        train_val_split: Fraction of training data to use for training (rest is validation).
        seed: Random seed for reproducible splits.
        **kwargs: Other args passed to dataset (transform, etc.)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    
    Example:
        train, val, test = get_cifar10_splits(keep_classes=[0, 1, 2])
    """
    # Choose dataset class based on whether we're filtering
    DatasetClass = FilteredCIFAR10 if keep_classes is not None else CIFAR10
    
    # Load train+val data
    train_val = DatasetClass(keep_classes=keep_classes, train=True, **kwargs) if keep_classes else DatasetClass(train=True, **kwargs)
    
    # Split into train and val
    # Set default split of 80/20% if not specified
    if n_samples_per_class_train is None:
        train_size = int(0.8 * len(train_val))
        val_size = len(train_val) - train_size
        train, val = random_split(
            train_val,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        if n_samples_per_class_val is not None:
            print(
                f"WARNING: Samples per class wasn't set for train, but was set for val with {n_samples_per_class_val} samples per class."
            )
        print(f"Using default 80/20% split.")

    else: # Samples per class were set for train and val
        if n_samples_per_class_val is None:
            n_samples_per_class_val = 500
            print(f"Using default {n_samples_per_class_val} samples per class for val.")
        targets = np.array(train_val.targets)
        indices = np.arange(len(targets))
        # Use keep_classes count if filtering, otherwise use dataset classes
        n_classes = len(keep_classes) if keep_classes else len(train_val.classes)
        train_size = n_samples_per_class_train * n_classes
        val_size = n_samples_per_class_val * n_classes

        train_indices, remaining_indices = train_test_split(
            indices,
            train_size=train_size,
            stratify=targets,
            random_state=SEED
        )

        validation_indices, _ = train_test_split(
            remaining_indices,
            train_size=val_size,
            stratify=targets[remaining_indices],
            random_state=SEED
        )
        train = Subset(train_val, train_indices)
        val = Subset(train_val, validation_indices)
    
    # Load test data
    test = DatasetClass(keep_classes=keep_classes, train=False, **kwargs) if keep_classes else DatasetClass(train=False, **kwargs)
    # Print sizes and samples per class
    print(f"Original train-val size: {len(train_val)}")
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")
    if n_samples_per_class_train is not None:
        train_labels = [targets[i] for i in train_indices]
        val_labels = [targets[i] for i in validation_indices]
        print("Samples per class (train):", Counter(train_labels))
        print("Samples per class (val):", Counter(val_labels))

    return train, val, test


def get_cifar10_loaders_and_splits(
    keep_classes: Optional[List[int]] = None,
    batch_size: int = BATCH_SIZE,
    # train_val_split: float = 0.8,
    n_samples_per_class_train: int = None,
    n_samples_per_class_val: int = None,
    seed: int = SEED,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test DataLoaders for CIFAR10.
    
    Args:
        keep_classes: List of class indices to keep. None for all classes.
        batch_size: Batch size for all loaders.
        train_val_split: Fraction of training data to use for training.
        seed: Random seed for reproducible splits.
        **kwargs: Other args passed to dataset (transform, etc.)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        train_loader, val_loader, test_loader = get_cifar10_loaders(keep_classes=[0, 1, 2], batch_size=32)
    """
    train, val, test = get_cifar10_splits(
        keep_classes=keep_classes,
        # train_val_split=train_val_split,
        n_samples_per_class_train=n_samples_per_class_train,
        n_samples_per_class_val=n_samples_per_class_val,
        seed=seed,
        **kwargs
    )
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train, val, test