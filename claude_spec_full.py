import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path


class SpecAugment(nn.Module):
    """
    SpecAugment implementation for PyTorch.
    
    Args:
        freq_mask_param (int): Maximum frequency mask size
        time_mask_param (int): Maximum time mask size  
        num_freq_masks (int): Number of frequency masks to apply
        num_time_masks (int): Number of time masks to apply
        p (float): Probability of applying SpecAugment to selected samples
        time_warp_param (int): Maximum time warp parameter (0 to disable)
        mask_value (float): Value to use for masking
        batch_ratio (float): Percentage of batch samples to randomly select for augmentation (0.0-1.0)
    """
    
    def __init__(self, freq_mask_param=27, time_mask_param=100, 
                 num_freq_masks=1, num_time_masks=1, p=0.8, 
                 time_warp_param=0, mask_value=0.0, batch_ratio=1.0):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        self.time_warp_param = time_warp_param
        self.mask_value = mask_value
        self.batch_ratio = batch_ratio  # Percentage of batch to augment
        
    def forward(self, x):
        """
        Apply SpecAugment to input spectrograms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, freq_bins, time_steps)
                             or (batch_size, freq_bins, time_steps)
        
        Returns:
            torch.Tensor: Augmented spectrograms with same shape as input
        """
        if not self.training:
            return x
            
        if random.random() > self.p:
            return x
            
        # (batch_size, channels, freq_bins, time_steps)
        batch_size, channels, freq_bins, time_steps = x.shape         

        # Apply augmentation to each sample in the batch
        augmented = x.clone()
        
        # Randomly select which samples in the batch to augment
        num_samples_to_augment = int(batch_size * self.batch_ratio)
        if num_samples_to_augment > 0:
            # Randomly select indices
            selected_indices = random.sample(range(batch_size), num_samples_to_augment)
        else:
            selected_indices = []

        print(f"Selected {len(selected_indices)} samples for augmentation out of {batch_size} total samples.")

        for i in selected_indices:
            # # Check individual sample probability
            # if random.random() > self.p:
            #     continue
                
            # Frequency masking
            for _ in range(self.num_freq_masks):
                augmented[i, 0] = self._freq_mask(augmented[i, 0])
            
            # Time masking
            for _ in range(self.num_time_masks):
                augmented[i, 0] = self._time_mask(augmented[i, 0])
        

        return augmented
    
    def _freq_mask(self, spec):
        """Apply frequency masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if freq_bins == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.freq_mask_param, freq_bins))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, freq_bins - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[mask_start:mask_end, :] = self.mask_value

        print(f"Applying frequency mask: start={mask_start}, end={mask_end}, size={mask_size}")
        
        return masked_spec
    
    def _time_mask(self, spec):
        """Apply time masking to a single spectrogram."""
        freq_bins, time_steps = spec.shape
        
        if time_steps == 0:
            return spec
        
        # Random mask size
        mask_size = random.randint(0, min(self.time_mask_param, time_steps))
        
        if mask_size == 0:
            return spec
        
        # Random mask position
        mask_start = random.randint(0, time_steps - mask_size)
        mask_end = mask_start + mask_size
        
        # Apply mask
        masked_spec = spec.clone()
        masked_spec[:, mask_start:mask_end] = self.mask_value

        print(f"Applying time mask: start={mask_start}, end={mask_end}, size={mask_size}")
        
        return masked_spec


# Standalone SpecAugment function for easy integration
def apply_specaugment(spectrograms, freq_mask_param=5, time_mask_param=40,
                     num_freq_masks=2, num_time_masks=2, p=0.8, 
                     time_warp_param=0, mask_value=0.0, batch_ratio=1.0):
    """
    Standalone function to apply SpecAugment to spectrograms.
    
    Args:
        spectrograms (torch.Tensor): Input spectrograms
        batch_ratio (float): Percentage of batch to augment (0.0-1.0)
        Other args: SpecAugment parameters
    
    Returns:
        torch.Tensor: Augmented spectrograms
    """
    augmenter = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        p=p,
        time_warp_param=time_warp_param,
        mask_value=mask_value,
        batch_ratio=batch_ratio
    )
    
    # Set to training mode to enable augmentation
    augmenter.train()
    
    with torch.no_grad():
        return augmenter(spectrograms)


def plot_specaugment_comparison(original, augmented, sample_idx=0, channel_idx=0, 
                               figsize=(15, 6), title_prefix="Spectrogram"):
    """
    Plot original and augmented spectrograms side by side for visualization.
    
    Args:
        original (torch.Tensor): Original spectrograms tensor
        augmented (torch.Tensor): Augmented spectrograms tensor  
        sample_idx (int): Which sample from batch to plot (default: 0)
        channel_idx (int): Which channel to plot for 4D tensors (default: 0)
        figsize (tuple): Figure size (width, height)
        title_prefix (str): Prefix for plot titles
    """
    # Convert to numpy and handle different tensor dimensions
    if original.dim() == 4:
        # (batch_size, channels, freq_bins, time_steps)
        orig_spec = original[sample_idx, channel_idx].detach().cpu().numpy()
        aug_spec = augmented[sample_idx, channel_idx].detach().cpu().numpy()
    elif original.dim() == 3:
        # (batch_size, freq_bins, time_steps)
        orig_spec = original[sample_idx].detach().cpu().numpy()
        aug_spec = augmented[sample_idx].detach().cpu().numpy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {original.dim()}D")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original spectrogram
    im1 = ax1.imshow(orig_spec, aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='nearest')
    ax1.set_title(f'{title_prefix} - Original (Sample {sample_idx})')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot augmented spectrogram
    im2 = ax2.imshow(aug_spec, aspect='auto', origin='lower', 
                     cmap='viridis', interpolation='nearest')
    ax2.set_title(f'{title_prefix} - SpecAugmented (Sample {sample_idx})')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Print augmentation statistics
    print(f"Original tensor stats - Min: {orig_spec.min():.3f}, Max: {orig_spec.max():.3f}, Mean: {orig_spec.mean():.3f}")
    print(f"Augmented tensor stats - Min: {aug_spec.min():.3f}, Max: {aug_spec.max():.3f}, Mean: {aug_spec.mean():.3f}")
    
    # Check if augmentation was applied (look for masked regions)
    diff = np.abs(orig_spec - aug_spec)
    augmented_pixels = np.sum(diff > 1e-6)
    total_pixels = orig_spec.size
    aug_percentage = (augmented_pixels / total_pixels) * 100
    print(f"Percentage of pixels modified: {aug_percentage:.2f}%")


def visualize_batch_augmentation(spectrograms, specaugment_params=None, num_samples=4):
    """
    Visualize SpecAugment effects on multiple samples from a batch.
    
    Args:
        spectrograms (torch.Tensor): Batch of spectrograms
        specaugment_params (dict): SpecAugment parameters (optional)
        num_samples (int): Number of samples to visualize
    """
    if specaugment_params is None:
        specaugment_params = {
            'freq_mask_param': 27,
            'time_mask_param': 100,
            'batch_ratio': 0.8,  # Augment 80% of batch
            'p': 0.9  # High probability to see effects
        }
    
    # Apply SpecAugment
    augmented = apply_specaugment(spectrograms, **specaugment_params)
    
    # Plot comparisons for multiple samples
    batch_size = min(spectrograms.size(0), num_samples)
    
    for i in range(batch_size):
        print(f"\n--- Sample {i} ---")
        plot_specaugment_comparison(spectrograms, augmented, sample_idx=i, 
                                  title_prefix=f"Mel Spectrogram {i}")


# Example usage with visualization
def demo_specaugment_with_plots():
    """
    Demonstration function showing SpecAugment with visualizations.
    """
    print("=== SpecAugment Demo with Visualizations ===\n")
    
    # Create sample mel spectrogram batch
    batch_size = 4
    freq_bins = 128
    time_steps = 200
    
    # Generate realistic-looking mel spectrograms
    torch.manual_seed(42)  # For reproducible demo
    spectrograms = torch.abs(torch.randn(batch_size, freq_bins, time_steps)) * 2
    
    # Add some structure to make it look more like real spectrograms
    for i in range(batch_size):
        # Add frequency bands
        spectrograms[i, 20:40, :] += 1.5
        spectrograms[i, 60:80, :] += 1.0
        # Add time-varying components
        spectrograms[i, :, 50:150] += 0.8
    
    print(f"Created batch of spectrograms: {spectrograms.shape}")
    
    # Test 1: Standard SpecAugment on full batch
    print("\n1. Standard SpecAugment (100% of batch):")
    specaugment_full = SpecAugment(batch_ratio=1.0, p=1.0)  # Augment all samples
    specaugment_full.train()
    augmented_full = specaugment_full(spectrograms)
    plot_specaugment_comparison(spectrograms, augmented_full, sample_idx=0)
    
    # Test 2: Partial batch augmentation
    print("\n2. Partial Batch SpecAugment (50% of batch):")
    specaugment_partial = SpecAugment(batch_ratio=0.5, p=1.0)  # Augment 50% of batch
    specaugment_partial.train()
    augmented_partial = specaugment_partial(spectrograms)
    
    # Visualize multiple samples to see which were augmented
    visualize_batch_augmentation(spectrograms, 
                               {'batch_ratio': 0.5, 'p': 1.0, 'freq_mask_param': 30}, 
                               num_samples=4)
    
    return spectrograms, augmented_full, augmented_partial


# Example usage
if __name__ == "__main__":
    # Create sample mel spectrogram batch
    root_dir = Path.home().joinpath('Dropbox','DATASETS_AUDIO')
    feat_path_dir = root_dir / 'Dvectors/noisy_all_18K/input_feats' 

    # List all files with *.pkl in directory, pathlib style
    feats_list = list(feat_path_dir.glob('*.pkl'))

    spectrograms_list = []
    for idx in range(0, 6):
        with open(feats_list[idx], 'rb') as f:
            current_feat_and_label = pickle.load(f)

            current_feat = current_feat_and_label['feat']

            # Swap axes to match (freq_bins, time_steps)
            current_feat = np.swapaxes(current_feat, 0, 1)


        spectrograms_list.append(current_feat)
    
    # Convert to tensor with shape (batch_size, freq_bins, time_steps)
    # Pad features to the maximum length in the batch
    max_length = max([f.shape[1] for f in spectrograms_list])
    spectrograms_tensor = torch.zeros(len(spectrograms_list), 1, spectrograms_list[0].shape[0], max_length)
    for i, feature in enumerate(spectrograms_list):
        spectrograms_tensor[i, :, :, :feature.shape[1]] = torch.from_numpy(feature)

    print(f"Created batch of spectrograms: {spectrograms_tensor.shape}")

    # Method 2: Using standalone function with partial batch augmentation
    augmented_fn = apply_specaugment(spectrograms_tensor, batch_ratio=1.0)  # Augment 70% of batch
    print(f"Function augmented shape: {augmented_fn.shape}")
    
    # Method 3: Visualization
    plot_specaugment_comparison(spectrograms_tensor, augmented_fn, sample_idx=0, 
                              title_prefix="Mel Spectrogram")
    