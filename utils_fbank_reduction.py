import numpy as np
import hdbscan
from pathlib import Path
import sys
import warnings
import argparse
import re
import pickle

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import librosa

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import dct


class SpectrogramCompressor:
    """
    Various methods to convert 2D spectrograms to compact 1D vectors
    Input shape: [n_filters, n_frames] where n_filters=40 (no batch dimension)
    """
    
    def __init__(self, n_filters=40):
        self.n_filters = n_filters
        self.pca = None
        self.scaler = None
        
    def statistical_features(self, spectrogram):
        """
        Extract statistical features across time dimension
        Returns: 1D vector of size n_filters * 6 = 240
        """
        # Handle input without batch dimension: [n_filters, n_frames]
        spec = spectrogram if len(spectrogram.shape) == 2 else spectrogram.squeeze(0)
        
        features = []
        for i in range(spec.shape[0]):  # For each filter
            freq_band = spec[i, :]  # Time series for this frequency band
            
            # Statistical moments
            features.extend([
                np.mean(freq_band),      # Mean energy
                np.std(freq_band),       # Energy variation
                np.max(freq_band),       # Peak energy
                np.min(freq_band),       # Minimum energy
                skew(freq_band),         # Skewness
                kurtosis(freq_band)      # Kurtosis
            ])
        
        return np.array(features)
    
    def temporal_pooling(self, spectrogram, pool_type='adaptive'):
        """
        Reduce temporal dimension using pooling
        Returns: 1D vector of size n_filters * pool_size
        """
        spec = spectrogram if len(spectrogram.shape) == 2 else spectrogram.squeeze(0)  # [n_filters, n_frames]
        
        if pool_type == 'adaptive':
            # Adaptive pooling to fixed size (e.g., 8 time steps)
            pool_size = 8
            pooled = np.zeros((spec.shape[0], pool_size))
            
            for i in range(spec.shape[0]):
                # Divide time dimension into pool_size segments and take max
                segment_size = spec.shape[1] // pool_size
                for j in range(pool_size):
                    start_idx = j * segment_size
                    end_idx = min((j + 1) * segment_size, spec.shape[1])
                    pooled[i, j] = np.max(spec[i, start_idx:end_idx])
            
            return pooled.flatten()  # Size: n_filters * pool_size = 320
        
        elif pool_type == 'global':
            # Global pooling: max, mean, std for each frequency band
            features = []
            for i in range(spec.shape[0]):
                features.extend([
                    np.max(spec[i, :]),
                    np.mean(spec[i, :]),
                    np.std(spec[i, :])
                ])
            return np.array(features)  # Size: n_filters * 3 = 120
    
    def delta_features(self, spectrogram):
        """
        Extract delta and delta-delta features, then apply statistical pooling
        Returns: 1D vector of size n_filters * 9 (static + delta + delta-delta, each with 3 stats)
        """
        spec = spectrogram if len(spectrogram.shape) == 2 else spectrogram.squeeze(0)  # [n_filters, n_frames]
        
        # Compute delta features
        delta = np.diff(spec, axis=1)
        delta_delta = np.diff(delta, axis=1)
        
        features = []
        for freq_band, d1, d2 in zip(spec, delta, delta_delta):
            # Statistical features for static, delta, and delta-delta
            features.extend([
                np.mean(freq_band), np.std(freq_band), np.max(freq_band),  # Static
                np.mean(d1), np.std(d1), np.max(d1),                      # Delta
                np.mean(d2), np.std(d2), np.max(d2)                       # Delta-delta
            ])
        
        return np.array(features)  # Size: n_filters * 9 = 360
    
    def mfcc_inspired(self, spectrogram):
        """
        Apply DCT to get MFCC-like features, then temporal pooling
        Returns: 1D vector of size n_coeffs * 3 (mean, std, max)
        """
        spec = spectrogram if len(spectrogram.shape) == 2 else spectrogram.squeeze(0)  # [n_filters, n_frames]
        
        # Apply DCT to get coefficients (keep first 13 like MFCC)
        n_coeffs = 13
        dct_features = np.zeros((n_coeffs, spec.shape[1]))
        
        for t in range(spec.shape[1]):
            # Apply DCT to each time frame
            dct_features[:, t] = dct(spec[:, t], n=n_coeffs)
        
        # Temporal pooling
        features = []
        for i in range(n_coeffs):
            features.extend([
                np.mean(dct_features[i, :]),
                np.std(dct_features[i, :]),
                np.max(dct_features[i, :])
            ])
        
        return np.array(features)  # Size: n_coeffs * 3 = 39
    
    
    
    def autoencoder_features(self, spectrograms, encoding_dim=64):
        """
        Train a simple autoencoder for dimensionality reduction
        Returns: trained encoder model
        """
        # Prepare data
        data = np.array([spec.flatten() for spec in spectrograms])
        input_dim = data.shape[1]
        
        # Simple autoencoder architecture
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, encoding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        # Train autoencoder
        model = Autoencoder(input_dim, encoding_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        data_tensor = torch.FloatTensor(data)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(data_tensor)
            loss = criterion(outputs, data_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        return model.encoder
    
    def frequency_bands_pooling(self, spectrogram, n_bands=8):
        """
        Group frequency bands and apply pooling
        Returns: 1D vector of size n_bands * 3 (max, mean, std per band)
        """
        spec = spectrogram if len(spectrogram.shape) == 2 else spectrogram.squeeze(0)  # [n_filters, n_frames]
        
        # Group frequency bands
        band_size = self.n_filters // n_bands
        features = []
        
        for i in range(n_bands):
            start_freq = i * band_size
            end_freq = min((i + 1) * band_size, self.n_filters)
            
            # Extract band data
            band_data = spec[start_freq:end_freq, :]
            
            # Compute statistics across both frequency and time
            features.extend([
                np.max(band_data),
                np.mean(band_data),
                np.std(band_data)
            ])
        
        return np.array(features)  # Size: n_bands * 3 = 24

def apply_spec_compression(spectro, method_sel = 0):

    compressor = SpectrogramCompressor()
    
    # Test different methods
    sample_spec = spectro 
    
    print("Original shape:", sample_spec.shape)
    print("Flattened size:", sample_spec.size)
    print("\nCompression methods:")

    comp_feats = None

    if method_sel == 0:
    
        # Method 0: Statistical features
        stat_features = compressor.statistical_features(sample_spec)
        print(f"Statistical features: {stat_features.shape} (compression: {sample_spec.size/len(stat_features):.1f}x)")

        comp_feats = stat_features
    
    elif method_sel == 1:
    
        # Method 1: Temporal pooling
        pool_features = compressor.temporal_pooling(sample_spec, 'adaptive')
        print(f"Adaptive pooling: {pool_features.shape} (compression: {sample_spec.size/len(pool_features):.1f}x)")
    
        global_pool = compressor.temporal_pooling(sample_spec, 'global')
        print(f"Global pooling: {global_pool.shape} (compression: {sample_spec.size/len(global_pool):.1f}x)")

        comp_feats = pool_features
    
    elif method_sel == 2:

        # Method 2: Delta features
        delta_features = compressor.delta_features(sample_spec)
        print(f"Delta features: {delta_features.shape} (compression: {sample_spec.size/len(delta_features):.1f}x)")

        comp_feats = delta_features

    elif method_sel == 3:
    
        # Method 3: MFCC-inspired
        mfcc_features = compressor.mfcc_inspired(sample_spec)
        print(f"MFCC-inspired: {mfcc_features.shape} (compression: {sample_spec.size/len(mfcc_features):.1f}x)")

        comp_feats = mfcc_features
    
    elif method_sel == 4:
    
        # Method 4: Frequency bands pooling
        band_features = compressor.frequency_bands_pooling(sample_spec, n_bands=8)
        print(f"Frequency bands: {band_features.shape} (compression: {sample_spec.size/len(band_features):.1f}x)")

        comp_feats = band_features


    return comp_feats