from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

from src.utils.config import DATA_PATH

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
    train_val_split: float = 0.8,
    seed: int = 42,
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
    train_size = int(train_val_split * len(train_val))
    val_size = len(train_val) - train_size
    train, val = random_split(
        train_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Load test data
    test = DatasetClass(keep_classes=keep_classes, train=False, **kwargs) if keep_classes else DatasetClass(train=False, **kwargs)
    
    return train, val, test


def get_cifar10_loaders(
    keep_classes: Optional[List[int]] = None,
    batch_size: int = 64,
    train_val_split: float = 0.8,
    seed: int = 42,
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
        train_val_split=train_val_split,
        seed=seed,
        **kwargs
    )
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader