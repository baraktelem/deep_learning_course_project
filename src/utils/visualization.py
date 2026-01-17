from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import Dataset


def visualize_model_diff(
    mask: Tensor, 
    dataset: Dataset, 
    preds_1: Tensor, 
    preds_2: Tensor, 
    true_labels: Tensor, 
    class_names: Optional[List[str]] = None, 
    title: str = "Comparison", 
    num_show: int = 5
):
    """
    Plots images where 'mask' is True.
    Displays: True Label, Model 1 Prediction, Model 2 Prediction (with Names).
    """
    # 1. Find the indices (IDs) where the mask is True
    indices = torch.nonzero(mask).flatten()
    
    count = len(indices)
    print(f"\n--- {title} ---")
    print(f"Total images in this category: {count}")
    
    if count == 0:
        return

    # 2. Setup Plot
    num_to_plot = min(num_show, count)
    plt.figure(figsize=(3 * num_to_plot, 4)) # Adjusted width dynamically
    
    for i in range(num_to_plot):
        idx = indices[i].item() # The Dataset Index
        
        # 3. Grab the image directly from the dataset
        img, _ = dataset[idx] 
        
        # 4. Get the specific predictions as integers
        p1 = preds_1[idx].item()
        p2 = preds_2[idx].item()
        truth = true_labels[idx].item()
        
        # 5. Get the Class Names (if provided)
        if class_names:
            truth_str = f"{truth} ({class_names[truth]})"
            p1_str = f"{p1} ({class_names[p1]})"
            p2_str = f"{p2} ({class_names[p2]})"
        else:
            truth_str, p1_str, p2_str = str(truth), str(p1), str(p2)
        
        # 6. Plot Image
        plt.subplot(1, num_to_plot, i+1)
        
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        title_text = (f"ID: {idx}\n"
                      f"Truth: {truth_str}\n"
                      f"Base: {p1_str}\n"
                      f"Scat: {p2_str}")
        
        plt.title(title_text, fontsize=9, loc='left')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()