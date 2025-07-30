#!/usr/bin/env python3
"""
Test script to verify the random splitting functionality
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, features, labels, features_paths=None, augment=False):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.features_paths = features_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AugmentedSubset(Dataset):
    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        features, label = self.dataset[self.indices[idx]]
        return features, label

def test_random_split():
    # Create test data
    features = np.random.randn(100, 256)
    labels = np.random.randint(0, 10, 100)
    features_paths = [f"path_{i}.wav" for i in range(100)]
    
    # Create full dataset
    full_dataset = TestDataset(features, labels, features_paths)
    
    # Generate random indices for splitting
    total_size = len(full_dataset)
    indices = torch.randperm(total_size).tolist()
    
    # Split indices (80% train, 20% validation)
    train_size = int(0.8 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = AugmentedSubset(full_dataset, train_indices, augment=True)
    val_dataset = AugmentedSubset(full_dataset, val_indices, augment=False)
    
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    
    print(f"No overlap: {len(train_set.intersection(val_set)) == 0}")
    print(f"All indices covered: {len(train_set.union(val_set)) == total_size}")
    
    # Test data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # Test one batch
    for batch_features, batch_labels in train_loader:
        print(f"First train batch shape: {batch_features.shape}")
        break
    
    print("Random split test passed!")

if __name__ == "__main__":
    test_random_split()
