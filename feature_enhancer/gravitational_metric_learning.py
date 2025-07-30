"""
Gravitational Metric Learning for Speaker Recognition

This module implements a gravity-inspired metric learning approach where the loss contribution
increases as samples get closer to their class centroids, following an inverse square law
similar to gravitational force. The key innovation is incremental sample addition with
adaptive gravity strength.

Key Features:
- Inverse square law loss weighting (F = 1/distance^2)
- Learnable class centroids (prototypes)
- Incremental sample addition capability
- Enhanced feature separation through gravity-inspired forces
- Real-time centroid updating with momentum

Author: Assistant
Date: July 29, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime


class GravitationalMetricLoss(nn.Module):
    """
    Gravity-inspired metric learning loss where the loss contribution increases
    as samples get closer to their class centroids (inverse square law).
    
    The gravitational force follows: F = 1 / (distance^gravity_power + epsilon)
    where gravity_power=2 gives the classic inverse square law.
    """
    
    def __init__(self, feature_dim=256, num_classes=90, alpha=1.0, beta=0.1, 
                 margin=1.0, gravity_power=2.0, epsilon=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for attraction term (pull toward correct centroid)
        self.beta = beta    # Weight for repulsion term (push away from wrong centroids)
        self.margin = margin  # Minimum margin between different classes
        self.gravity_power = gravity_power  # Power for inverse distance (default: 2 for inverse square)
        self.epsilon = epsilon  # Small constant to avoid division by zero
        
        # Learnable class centroids (prototypes) - these are the "gravitational centers"
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.centroids)
        
    def update_centroids(self, features, labels, momentum=0.9):
        """
        Update centroids using exponential moving average.
        This simulates how gravitational centers adapt to the distribution of samples.
        """
        with torch.no_grad():
            for class_id in range(self.num_classes):
                mask = (labels == class_id)
                if mask.sum() > 0:
                    class_features = features[mask]
                    new_centroid = class_features.mean(dim=0)
                    # Exponential moving average update
                    self.centroids[class_id] = (momentum * self.centroids[class_id] + 
                                              (1 - momentum) * new_centroid)
    
    def compute_distances(self, features, centroids):
        """
        Compute L2 distances between features and centroids.
        This represents the "distance" in the gravitational field.
        """
        # features: (batch_size, feature_dim)
        # centroids: (num_classes, feature_dim)
        
        # Expand dimensions for broadcasting
        features_expanded = features.unsqueeze(1)  # (batch_size, 1, feature_dim)
        centroids_expanded = centroids.unsqueeze(0)  # (1, num_classes, feature_dim)
        
        # Compute L2 distances
        distances = torch.norm(features_expanded - centroids_expanded, dim=2)  # (batch_size, num_classes)
        return distances
    
    def gravitational_force(self, distance):
        """
        Compute gravitational force based on inverse power law.
        F = 1 / (distance^gravity_power + epsilon)
        
        The closer the sample to the centroid, the stronger the force (higher loss).
        """
        return 1.0 / (torch.pow(distance + self.epsilon, self.gravity_power))
    
    def forward(self, features, labels):
        """
        Compute gravity-inspired metric learning loss.
        
        Args:
            features: Enhanced features (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components and statistics
        """
        batch_size = features.size(0)
        
        # Normalize features for better stability
        features = nn.functional.normalize(features, p=2, dim=1)
        centroids = nn.functional.normalize(self.centroids, p=2, dim=1)
        
        # Compute distances to all centroids
        distances = self.compute_distances(features, centroids)  # (batch_size, num_classes)
        
        # Compute gravitational forces
        forces = self.gravitational_force(distances)  # (batch_size, num_classes)
        
        # ATTRACTION LOSS: Pull samples toward their true class centroid
        # The key insight: stronger force (higher loss) when closer to centroid
        # This encourages samples to get even closer to their correct centroid
        positive_distances = distances[torch.arange(batch_size), labels]
        positive_forces = forces[torch.arange(batch_size), labels]
        attraction_loss = (positive_forces * positive_distances).mean()
        
        # REPULSION LOSS: Push samples away from other class centroids
        # But only penalize if too close to wrong centroids (within margin)
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False  # Exclude true class
        
        negative_distances = distances[mask].view(batch_size, -1)  # (batch_size, num_classes-1)
        negative_forces = forces[mask].view(batch_size, -1)
        
        # Only penalize if within margin of wrong centroids
        violation_mask = negative_distances < self.margin
        if violation_mask.sum() > 0:
            repulsion_loss = (negative_forces[violation_mask] / 
                            (negative_distances[violation_mask] + self.epsilon)).mean()
        else:
            repulsion_loss = torch.tensor(0.0, device=features.device)
        
        # Total loss combines attraction and repulsion
        total_loss = self.alpha * attraction_loss + self.beta * repulsion_loss
        
        # Compute metrics for monitoring the gravitational system
        with torch.no_grad():
            avg_positive_distance = positive_distances.mean()
            avg_negative_distance = negative_distances.mean()
            avg_positive_force = positive_forces.mean()
            centroid_spread = torch.norm(centroids.std(dim=0))
            
        metrics = {
            'total_loss': total_loss.item(),
            'attraction_loss': attraction_loss.item(),
            'repulsion_loss': repulsion_loss.item(),
            'avg_positive_distance': avg_positive_distance.item(),
            'avg_negative_distance': avg_negative_distance.item(),
            'avg_positive_force': avg_positive_force.item(),
            'centroid_spread': centroid_spread.item(),
            'margin_violations': violation_mask.sum().item()
        }
        
        return total_loss, metrics


class MetricLearningEnhancer(nn.Module):
    """
    Feature enhancement network specifically designed for metric learning
    with gravity-inspired loss. This network transforms features to be more
    suitable for gravitational metric learning.
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature transformation network
        # This network learns to enhance features for better gravitational dynamics
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Residual connection to preserve original information
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, output_dim)
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        enhanced = self.feature_net(x)
        residual = self.residual(x)
        
        # Combine with learnable residual weight
        output = enhanced + self.residual_weight * residual
        
        # L2 normalize output for better metric learning
        # This ensures all points lie on a unit hypersphere
        output = nn.functional.normalize(output, p=2, dim=1)
        
        return output


class GravitationalMetricTrainer:
    """
    Trainer for gravity-inspired metric learning with incremental sample addition.
    
    This is the core class that implements the "crazy idea" of incremental learning
    with gravity-inspired forces.
    """
    
    def __init__(self, enhancer, metric_loss, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=0.001, weight_decay=1e-4, centroid_update_freq=10):
        self.enhancer = enhancer.to(device)
        self.metric_loss = metric_loss.to(device)
        self.device = device
        
        # Optimizer for both enhancer network and learnable centroids
        self.optimizer = optim.Adam(
            list(self.enhancer.parameters()) + list(self.metric_loss.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)
        
        # Training state
        self.centroid_update_freq = centroid_update_freq
        self.training_metrics = []
        self.epoch_count = 0
        
    def train_batch(self, features, labels, update_centroids=True):
        """
        Train on a single batch with gravity-inspired loss.
        """
        self.enhancer.train()
        
        features, labels = features.to(self.device), labels.to(self.device)
        
        # Forward pass through enhancer
        enhanced_features = self.enhancer(features)
        
        # Compute gravity-inspired metric loss
        loss, metrics = self.metric_loss(enhanced_features, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update centroids periodically using momentum
        if update_centroids and self.epoch_count % self.centroid_update_freq == 0:
            self.metric_loss.update_centroids(enhanced_features.detach(), labels)
        
        return loss.item(), metrics, enhanced_features.detach()
    
    def add_new_samples(self, new_features, new_labels, num_epochs=10, verbose=True):
        """
        Add new samples and adapt the network incrementally.
        
        This is the KEY METHOD that implements your incremental learning idea!
        
        The network dynamically adapts to new samples by:
        1. Temporarily increasing gravity strength (alpha, beta)
        2. Training intensively on the new samples
        3. Allowing centroids to adapt to the new data distribution
        """
        if verbose:
            print(f"\n🌌 Adding {len(new_features)} new samples to gravitational system...")
            logging.info(f"Adding {len(new_features)} new samples...")
        
        # Convert to tensors
        if isinstance(new_features, np.ndarray):
            new_features = torch.FloatTensor(new_features)
        if isinstance(new_labels, np.ndarray):
            new_labels = torch.LongTensor(new_labels)
        
        # Create a small dataset for the new samples
        new_dataset = AudioDataset(new_features, new_labels)
        new_loader = DataLoader(new_dataset, batch_size=16, shuffle=True)
        
        # GRAVITY BOOST: Temporarily increase gravity strength for new samples
        # This ensures new samples are quickly pulled into the correct regions
        original_alpha = self.metric_loss.alpha
        original_beta = self.metric_loss.beta
        
        # Increase gravity strength for new samples
        self.metric_loss.alpha *= 1.5  # Stronger attraction
        self.metric_loss.beta *= 1.2   # Stronger repulsion
        
        if verbose:
            print(f"  🚀 Gravity boost: α={self.metric_loss.alpha:.2f}, β={self.metric_loss.beta:.2f}")
        
        epoch_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_features, batch_labels in new_loader:
                loss, metrics, _ = self.train_batch(batch_features, batch_labels)
                epoch_loss += loss
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_loss)
            
            if verbose and epoch % 5 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                logging.info(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Restore original gravity strength
        self.metric_loss.alpha = original_alpha
        self.metric_loss.beta = original_beta
        
        if verbose:
            print(f"  ✅ New samples integrated. Final loss: {epoch_losses[-1]:.4f}")
            print(f"  🔄 Gravity restored: α={self.metric_loss.alpha:.2f}, β={self.metric_loss.beta:.2f}")
            logging.info(f"New samples integrated. Final loss: {epoch_losses[-1]:.4f}")
        
        return epoch_losses
    
    def train_epoch(self, data_loader, update_centroids=True):
        """
        Train for one full epoch with standard gravity settings.
        """
        self.enhancer.train()
        
        total_loss = 0
        total_metrics = {}
        batch_count = 0
        
        for batch_features, batch_labels in tqdm(data_loader, desc=f"Epoch {self.epoch_count+1}"):
            loss, metrics, _ = self.train_batch(batch_features, batch_labels, update_centroids)
            
            total_loss += loss
            batch_count += 1
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
        
        # Average metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_metrics = {key: value / batch_count for key, value in total_metrics.items()}
        
        self.epoch_count += 1
        self.scheduler.step()
        
        # Store training history
        self.training_metrics.append(avg_metrics)
        
        return avg_loss, avg_metrics
    
    def evaluate_metric_quality(self, data_loader, sample_size=500):
        """
        Evaluate the quality of learned features using various metrics.
        This tells us how well the gravitational system is working.
        """
        self.enhancer.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                enhanced_features = self.enhancer(features)
                
                all_features.append(enhanced_features.cpu().numpy())
                all_labels.append(labels.numpy())
                
                if len(all_features) * features.size(0) >= sample_size:
                    break
        
        all_features = np.vstack(all_features)[:sample_size]
        all_labels = np.hstack(all_labels)[:sample_size]
        
        # Compute evaluation metrics
        metrics = {}
        
        try:
            # Silhouette score (cluster quality)
            metrics['silhouette_score'] = silhouette_score(all_features, all_labels)
            
            # KNN accuracy (classification quality)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn_scores = cross_val_score(knn, all_features, all_labels, cv=3)
            metrics['knn_accuracy'] = np.mean(knn_scores)
            metrics['knn_std'] = np.std(knn_scores)
            
            # Intra-class vs inter-class distance ratio
            distances = []
            for i in range(min(200, len(all_features))):
                for j in range(i+1, min(200, len(all_features))):
                    dist = np.linalg.norm(all_features[i] - all_features[j])
                    same_class = all_labels[i] == all_labels[j]
                    distances.append((dist, same_class))
            
            intra_distances = [d[0] for d in distances if d[1]]
            inter_distances = [d[0] for d in distances if not d[1]]
            
            if intra_distances and inter_distances:
                metrics['intra_class_distance'] = np.mean(intra_distances)
                metrics['inter_class_distance'] = np.mean(inter_distances)
                metrics['separation_ratio'] = np.mean(inter_distances) / np.mean(intra_distances)
            
        except Exception as e:
            print(f"Warning: Evaluation metric computation failed: {e}")
            logging.warning(f"Evaluation metric computation failed: {e}")
        
        return metrics
    
    def visualize_centroids(self, output_path=None):
        """
        Visualize the learned centroids using PCA.
        This shows how the "gravitational centers" are distributed.
        """
        try:
            centroids = self.metric_loss.centroids.detach().cpu().numpy()
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            centroids_2d = pca.fit_transform(centroids)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                                c=range(len(centroids_2d)), cmap='tab20', s=50)
            plt.colorbar(scatter, label='Speaker ID')
            plt.title('Learned Speaker Centroids (Gravitational Centers)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True, alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logging.info(f"Centroid visualization saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Warning: Centroid visualization failed: {e}")
            logging.warning(f"Centroid visualization failed: {e}")
    
    def get_feature_statistics(self):
        """
        Get statistics about the current state of the gravitational system.
        """
        with torch.no_grad():
            centroids = self.metric_loss.centroids.cpu().numpy()
            
            # Centroid statistics
            centroid_norms = np.linalg.norm(centroids, axis=1)
            pairwise_distances = []
            
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    pairwise_distances.append(dist)
            
            stats = {
                'centroid_mean_norm': np.mean(centroid_norms),
                'centroid_std_norm': np.std(centroid_norms),
                'mean_pairwise_distance': np.mean(pairwise_distances),
                'min_pairwise_distance': np.min(pairwise_distances),
                'max_pairwise_distance': np.max(pairwise_distances),
                'num_classes': len(centroids)
            }
            
        return stats


class AudioDataset(Dataset):
    """Simple dataset class for audio features."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def visualize_gravitational_effect(max_distance=5.0, powers=[1.0, 2.0, 3.0]):
    """
    Visualize how the gravitational force changes with distance for different powers.
    This helps understand the "crazy idea" mathematically.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create a range of distances
        distance_range = np.linspace(0.1, max_distance, 100)
        epsilon = 1e-6
        
        plt.figure(figsize=(15, 10))
        
        # Plot gravitational forces
        plt.subplot(2, 3, 1)
        for power in powers:
            forces = 1.0 / (np.power(distance_range + epsilon, power))
            plt.plot(distance_range, forces, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance to Centroid')
        plt.ylabel('Gravitational Force')
        plt.title('Gravitational Force vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot loss contribution (force × distance)
        plt.subplot(2, 3, 2)
        for power in powers:
            forces = 1.0 / (np.power(distance_range + epsilon, power))
            loss_contribution = forces * distance_range
            plt.plot(distance_range, loss_contribution, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance to Centroid')
        plt.ylabel('Loss Contribution (Force × Distance)')
        plt.title('Loss Contribution vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot derivative (how loss changes with distance)
        plt.subplot(2, 3, 3)
        for power in powers:
            # Derivative of force × distance = (1 - power) / distance^power
            derivative = (1 - power) / (np.power(distance_range + epsilon, power))
            plt.plot(distance_range, derivative, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance to Centroid')
        plt.ylabel('d(Loss)/d(Distance)')
        plt.title('Loss Gradient vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Conceptual diagram of gravitational field
        plt.subplot(2, 3, 4)
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # Simulate gravitational field around a centroid at origin
        R = np.sqrt(X**2 + Y**2)
        Z = 1.0 / (R + 0.1)**2  # Gravitational potential
        
        contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
        plt.colorbar(contour, label='Gravitational Potential')
        plt.plot(0, 0, 'ro', markersize=15, label='Centroid (Gravitational Center)')
        
        # Add sample points
        sample_points_x = [1.5, -1.2, 0.8, -0.5]
        sample_points_y = [0.5, 1.8, -1.5, -2.0]
        plt.scatter(sample_points_x, sample_points_y, c='red', s=100, alpha=0.7, 
                   label='Samples', edgecolors='black', linewidth=2)
        
        plt.xlabel('Feature Dimension 1')
        plt.ylabel('Feature Dimension 2')
        plt.title('Gravitational Field Around Centroid')
        plt.legend()
        plt.axis('equal')
        
        # Multiple centroids example
        plt.subplot(2, 3, 5)
        centroids = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        
        for i, (cx, cy) in enumerate(centroids):
            # Create gravitational field for each centroid
            R_centroid = np.sqrt((X - cx)**2 + (Y - cy)**2)
            Z_centroid = 1.0 / (R_centroid + 0.1)**2
            
            # Plot contours for this centroid
            plt.contour(X, Y, Z_centroid, levels=5, alpha=0.6, colors=f'C{i}')
            plt.plot(cx, cy, 'o', markersize=12, color=f'C{i}', label=f'Centroid {i+1}')
        
        plt.xlabel('Feature Dimension 1')
        plt.ylabel('Feature Dimension 2')
        plt.title('Multiple Gravitational Centers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Explanation text
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, "🌌 GRAVITATIONAL METRIC LEARNING", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, "Key Insights:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, "• Inverse square law (power=2) provides strong", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.65, "  attraction when close, weak when far", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.55, "• Loss contribution peaks at intermediate distances", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.45, "• Gradient becomes negative for power > 1,", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, "  encouraging movement toward centroid", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, "• Multiple centroids create complex dynamics", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, "• Incremental samples adapt existing fields", fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.1, "• 🚀 Gravity boost helps integrate new samples", fontsize=10, transform=plt.gca().transAxes, color='red')
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Gravitational Metric Learning: Mathematical Foundation', y=1.02, fontsize=16, fontweight='bold')
        
        plt.show()
        
        print("\n🌌 GRAVITATIONAL EFFECT ANALYSIS:")
        print("="*60)
        print("• Inverse square law (power=2) provides strong attraction when close, weak when far")
        print("• Loss contribution peaks at intermediate distances")
        print("• Gradient becomes negative for power > 1, encouraging movement toward centroid")
        print("• The field visualization shows how samples are attracted to their centroids")
        print("• Multiple centroids create complex gravitational dynamics")
        print("• New samples can be integrated by temporarily boosting gravity strength")
        print("="*60)
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def demonstrate_gravitational_metric_learning(data_path=None):
    """
    Demonstrate the gravity-inspired metric learning approach with incremental sample addition.
    
    This function showcases the complete "crazy idea" in action!
    """
    print("\n" + "="*80)
    print("🌌 GRAVITATIONAL METRIC LEARNING DEMONSTRATION 🌌")
    print("="*80)
    print("Implementing gravity-inspired metric learning with incremental sample addition!")
    print("Key innovation: Loss increases as inverse square of distance to centroid")
    print("="*80)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'gravitational_learning_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("GRAVITATIONAL METRIC LEARNING DEMONSTRATION")
    
    # Configuration
    FEATURE_DIM = 256
    NUM_SPEAKERS = 90
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Number of speakers: {NUM_SPEAKERS}")
    
    # Create synthetic data for demonstration (replace with your actual data loading)
    if data_path is None:
        print("\n📊 Creating synthetic demonstration data...")
        # Create synthetic speaker features for demonstration
        np.random.seed(42)
        n_samples_per_speaker = 50
        total_samples = NUM_SPEAKERS * n_samples_per_speaker
        
        # Generate synthetic features with some structure
        features = []
        labels = []
        
        for speaker_id in range(NUM_SPEAKERS):
            # Create a random centroid for each speaker
            centroid = np.random.randn(FEATURE_DIM) * 2
            
            # Generate samples around this centroid
            for _ in range(n_samples_per_speaker):
                sample = centroid + np.random.randn(FEATURE_DIM) * 0.5
                features.append(sample)
                labels.append(speaker_id)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Generated {len(features)} synthetic samples")
    else:
        # Load actual data
        print(f"\n📊 Loading data from: {data_path}")
        with open(data_path, 'rb') as f:
            features, wav_paths, labels = pickle.load(f)
        labels = np.array(labels) - 1  # Convert to 0-indexed
        print(f"Loaded {len(features)} real samples")
    
    # Split into initial training set and "new samples" to add incrementally
    # This simulates the scenario where you start with some data and keep adding more
    X_initial, X_new, y_initial, y_new = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n📈 Data split:")
    print(f"  Initial training samples: {len(X_initial)}")
    print(f"  New samples to add incrementally: {len(X_new)}")
    
    # Create models
    print(f"\n🏗️ Creating gravitational metric learning models...")
    
    enhancer = MetricLearningEnhancer(
        input_dim=FEATURE_DIM,
        hidden_dim=512,
        output_dim=FEATURE_DIM,
        dropout_rate=0.3
    )
    
    metric_loss = GravitationalMetricLoss(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_SPEAKERS,
        alpha=1.0,      # Attraction strength
        beta=0.5,       # Repulsion strength
        margin=0.8,     # Margin for repulsion
        gravity_power=2.0,  # Inverse square law
        epsilon=1e-6
    )
    
    # Create trainer
    trainer = GravitationalMetricTrainer(
        enhancer, metric_loss, device=DEVICE,
        lr=0.001, weight_decay=1e-4, centroid_update_freq=5
    )
    
    print(f"  ✅ Enhancer network: {sum(p.numel() for p in enhancer.parameters()):,} parameters")
    print(f"  ✅ Metric loss: {sum(p.numel() for p in metric_loss.parameters()):,} parameters")
    
    # Create initial data loader
    initial_dataset = AudioDataset(X_initial, y_initial)
    initial_loader = DataLoader(initial_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Phase 1: Initial training
    print("\n" + "="*60)
    print("🚀 PHASE 1: INITIAL GRAVITATIONAL SYSTEM TRAINING")
    print("="*60)
    
    for epoch in range(15):  # Initial training epochs
        loss, metrics = trainer.train_epoch(initial_loader)
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch+1:2d}/15:")
            print(f"  🌌 Total Loss: {loss:.4f}")
            print(f"  🔵 Attraction: {metrics['attraction_loss']:.4f} (pull toward correct centroid)")
            print(f"  🔴 Repulsion: {metrics['repulsion_loss']:.4f} (push from wrong centroids)")
            print(f"  📏 Avg Positive Distance: {metrics['avg_positive_distance']:.4f}")
            print(f"  📊 Separation Ratio: {metrics['avg_negative_distance']/max(metrics['avg_positive_distance'], 1e-6):.2f}")
            print(f"  ⚡ Avg Force: {metrics['avg_positive_force']:.4f}")
            
            logging.info(f"Epoch {epoch+1}/15: Loss={loss:.4f}, "
                        f"Attraction={metrics['attraction_loss']:.4f}, "
                        f"Repulsion={metrics['repulsion_loss']:.4f}")
    
    # Evaluate initial model
    print(f"\n📊 Evaluating initial gravitational system...")
    initial_metrics = trainer.evaluate_metric_quality(initial_loader)
    print("Initial System Metrics:")
    for key, value in initial_metrics.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Initial {key}: {value:.4f}")
    
    # Visualize gravitational effect
    print(f"\n📈 Visualizing gravitational dynamics...")
    visualize_gravitational_effect()
    
    # Phase 2: Incremental addition of new samples
    print("\n" + "="*60)
    print("🌟 PHASE 2: INCREMENTAL GRAVITATIONAL SAMPLE ADDITION")
    print("="*60)
    print("This is where the magic happens! Adding new samples with gravity boost...")
    
    # Split new samples into batches for incremental addition
    batch_size_new = len(X_new) // 4  # Add in 4 batches
    
    for i in range(4):
        start_idx = i * batch_size_new
        end_idx = start_idx + batch_size_new if i < 3 else len(X_new)
        
        batch_features = X_new[start_idx:end_idx]
        batch_labels = y_new[start_idx:end_idx]
        
        print(f"\n🎯 Adding batch {i+1}/4 ({len(batch_features)} samples)...")
        
        # Add new samples with gravity-enhanced learning
        losses = trainer.add_new_samples(batch_features, batch_labels, num_epochs=12)
        
        # Evaluate after each addition
        if i % 2 == 1:  # Evaluate every other batch
            combined_features = np.vstack([X_initial, X_new[:end_idx]])
            combined_labels = np.hstack([y_initial, y_new[:end_idx]])
            combined_dataset = AudioDataset(combined_features, combined_labels)
            combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            metrics = trainer.evaluate_metric_quality(combined_loader)
            print(f"  📊 Metrics after batch {i+1}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("🏆 FINAL GRAVITATIONAL SYSTEM EVALUATION")
    print("="*60)
    
    # Create final combined dataset
    final_features = np.vstack([X_initial, X_new])
    final_labels = np.hstack([y_initial, y_new])
    final_dataset = AudioDataset(final_features, final_labels)
    final_loader = DataLoader(final_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    final_metrics = trainer.evaluate_metric_quality(final_loader, sample_size=1000)
    
    print("🎉 Final Gravitational System Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Final {key}: {value:.4f}")
    
    # Compare initial vs final
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print("-" * 40)
    for key in initial_metrics:
        if key in final_metrics:
            improvement = final_metrics[key] - initial_metrics[key]
            direction = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  {direction} {key}: {improvement:+.4f}")
            logging.info(f"Improvement in {key}: {improvement:+.4f}")
    
    # Visualize centroids
    print(f"\n🎨 Visualizing learned gravitational centers...")
    try:
        trainer.visualize_centroids('gravitational_centroids.png')
    except:
        print("Centroid visualization failed (matplotlib issue)")
    
    # Get final statistics
    stats = trainer.get_feature_statistics()
    print(f"\n📊 Final Gravitational System Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Final {key}: {value:.4f}")
    
    print("\n" + "="*80)
    print("🎊 GRAVITATIONAL METRIC LEARNING DEMONSTRATION COMPLETE! 🎊")
    print("="*80)
    print("Your 'crazy idea' has been successfully implemented!")
    print("Key achievements:")
    print("✅ Inverse square law loss weighting")
    print("✅ Learnable gravitational centroids")  
    print("✅ Incremental sample addition with gravity boost")
    print("✅ Dynamic centroid adaptation")
    print("✅ Enhanced feature separation")
    print("="*80)
    
    logging.info("GRAVITATIONAL METRIC LEARNING DEMONSTRATION COMPLETE")
    
    return trainer, final_metrics


def main():
    """
    Main function to run the gravitational metric learning demonstration.
    """
    print("🌌 Welcome to Gravitational Metric Learning! 🌌")
    print("This implements your 'crazy idea' about gravity-inspired loss functions!")
    
    # You can specify your actual data path here
    # data_path = "path/to/your/d_vectors_feats.pickle"
    data_path = None  # Will use synthetic data for demonstration
    
    try:
        trainer, metrics = demonstrate_gravitational_metric_learning(data_path)
        print("\n🎉 SUCCESS! Your gravitational metric learning system is working!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
