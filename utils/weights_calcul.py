import numpy as np
import torch
from torch.utils.data import DataLoader

def calculate_class_weights(dataset, n_classes):
    """Calcule les poids des classes en fonction de leur fréquence dans les masques."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    class_counts = np.zeros(n_classes, dtype=np.int64)

    for batch in loader:
        masks = batch['mask'].numpy().flatten()
        class_counts += np.bincount(masks, minlength=n_classes)

    total_pixels = class_counts.sum()
    class_weights = total_pixels / (n_classes * class_counts + 1e-6)  # Évite la division par 0
    return torch.tensor(class_weights, dtype=torch.float32)
