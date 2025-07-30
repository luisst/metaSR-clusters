import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
import math

class FeatureRefinementNetwork(nn.Module):
    """
    Network to refine pretrained speech features for better clustering.
    Uses residual connections to preserve original feature information.
    """
    def __init__(self, input_dim=256, hidden_dim=512, dropout=0.3):
        super(FeatureRefinementNetwork, self).__init__()
        
        # Encoder path with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Bottleneck layer for refined features
        self.bottleneck = nn.Linear(hidden_dim, input_dim)
        
        # Decoder path to reconstruct original features
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Refined features (bottleneck)
        refined_features = self.bottleneck(encoded)
        
        # Add residual connection to preserve original information
        refined_features = refined_features + 0.1 * x  # Small residual weight
        
        # Decoder for reconstruction
        reconstructed = self.decoder(refined_features)
        
        return refined_features, reconstructed

class PrototypeContrastiveLoss(nn.Module):
    """
    Custom loss combining prototype-based learning with contrastive loss
    and reconstruction loss.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, temperature=0.1):
        super(PrototypeContrastiveLoss, self).__init__()
        self.alpha = alpha  # Prototype loss weight
        self.beta = beta    # Contrastive loss weight
        self.gamma = gamma  # Reconstruction loss weight
        self.temperature = temperature
        
    def forward(self, refined_features, reconstructed, original_features, labels, prototypes):
        batch_size = refined_features.size(0)
        
        # 1. Prototype Loss - Pull samples toward their class prototype
        prototype_loss = 0
        for i in range(batch_size):
            label = labels[i].item()
            if label in prototypes:
                prototype = prototypes[label]
                prototype_loss += F.mse_loss(refined_features[i], prototype)
        prototype_loss = prototype_loss / batch_size if batch_size > 0 else 0
        
        # 2. Contrastive Loss - Push different classes apart
        contrastive_loss = 0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                dist = F.pairwise_distance(refined_features[i:i+1], refined_features[j:j+1])
                
                if labels[i] == labels[j]:  # Same class - minimize distance
                    contrastive_loss += dist
                else:  # Different class - maximize distance
                    margin = 2.0
                    contrastive_loss += F.relu(margin - dist)
                num_pairs += 1
        
        contrastive_loss = contrastive_loss / num_pairs if num_pairs > 0 else 0
        
        # 3. Reconstruction Loss - Maintain feature information
        reconstruction_loss = F.mse_loss(reconstructed, original_features)
        
        # Combined loss
        total_loss = (self.alpha * prototype_loss + 
                     self.beta * contrastive_loss + 
                     self.gamma * reconstruction_loss)
        
        return total_loss, prototype_loss, contrastive_loss, reconstruction_loss

class SpeechFeatureDataset(Dataset):
    """Dataset class for speech features and labels"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FeatureRefinementTrainer:
    """
    Main trainer class for the feature refinement network
    """
    def __init__(self, input_dim=256, hidden_dim=512, learning_rate=1e-3, device='cuda'):
        self.device = device
        self.model = FeatureRefinementNetwork(input_dim, hidden_dim).to(device)
        self.criterion = PrototypeContrastiveLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Store prototypes for each class
        self.prototypes = {}
        
    def update_prototypes(self, features, labels):
        """Update class prototypes based on current refined features"""
        self.model.eval()
        class_features = defaultdict(list)
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            refined_features, _ = self.model(features_tensor)
            refined_features = refined_features.cpu().numpy()
            
            # Group features by class
            for feature, label in zip(refined_features, labels):
                class_features[label].append(feature)
            
            # Compute prototypes as class centroids
            for label, feats in class_features.items():
                if len(feats) > 0:
                    self.prototypes[label] = torch.FloatTensor(np.mean(feats, axis=0)).to(self.device)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_proto_loss = 0
        total_contrastive_loss = 0
        total_recon_loss = 0
        
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            refined_features, reconstructed = self.model(batch_features)
            
            # Compute loss
            loss, proto_loss, cont_loss, recon_loss = self.criterion(
                refined_features, reconstructed, batch_features, batch_labels, self.prototypes
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_proto_loss += proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss
            total_contrastive_loss += cont_loss.item() if isinstance(cont_loss, torch.Tensor) else cont_loss
            total_recon_loss += recon_loss.item()
        
        return (total_loss / len(dataloader), 
                total_proto_loss / len(dataloader),
                total_contrastive_loss / len(dataloader), 
                total_recon_loss / len(dataloader))
    
    def fit(self, features, labels, epochs=50, batch_size=32, validation_split=0.2):
        """
        Main training loop with prototype updates
        
        Args:
            features: numpy array of shape (n_samples, 256)
            labels: numpy array of shape (n_samples,)
            epochs: number of training epochs
            batch_size: batch size for training
            validation_split: fraction of data to use for validation
        """
        # Split data
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_features = features[indices[n_val:]]
        train_labels = labels[indices[n_val:]]
        val_features = features[indices[:n_val]]
        val_labels = labels[indices[:n_val]]
        
        # Create datasets
        train_dataset = SpeechFeatureDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initial prototype computation
        self.update_prototypes(train_features, train_labels)
        
        print(f"Training with {len(train_features)} samples, {len(self.prototypes)} classes")
        print(f"Validation with {len(val_features)} samples")
        
        for epoch in range(epochs):
            # Training
            train_loss, proto_loss, cont_loss, recon_loss = self.train_epoch(train_loader)
            
            # Update prototypes every few epochs
            if epoch % 5 == 0:
                self.update_prototypes(train_features, train_labels)
            
            # Validation
            val_loss = self.evaluate(val_features, val_labels)
            
            self.scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f} "
                      f"(Proto: {proto_loss:.4f}, Cont: {cont_loss:.4f}, Recon: {recon_loss:.4f}) "
                      f"Val Loss: {val_loss:.4f}")
    
    def evaluate(self, features, labels):
        """Evaluate model on validation data"""
        self.model.eval()
        val_dataset = SpeechFeatureDataset(features, labels)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        total_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                refined_features, reconstructed = self.model(batch_features)
                loss, _, _, _ = self.criterion(
                    refined_features, reconstructed, batch_features, batch_labels, self.prototypes
                )
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def refine_features(self, features):
        """
        Refine input features using the trained model
        
        Args:
            features: numpy array of shape (n_samples, 256)
            
        Returns:
            refined_features: numpy array of shape (n_samples, 256)
        """
        self.model.eval()
        refined_features = []
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Process in batches to handle memory constraints
            batch_size = 64
            for i in range(0, len(features), batch_size):
                batch = features_tensor[i:i+batch_size]
                refined_batch, _ = self.model(batch)
                refined_features.append(refined_batch.cpu().numpy())
        
        return np.vstack(refined_features)
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'prototypes': self.prototypes,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.prototypes = checkpoint['prototypes']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Example usage and integration with active learning loop
def active_learning_pipeline_example():
    """
    Example of how to integrate with your active learning pipeline
    """
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = FeatureRefinementTrainer(device=device)
    
    # Dummy data for example (replace with your actual cached features)
    n_samples = 1000
    n_features = 256
    n_classes = 10
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)
    
    print("Starting initial training...")
    trainer.fit(features, labels, epochs=100, batch_size=32)
    
    # Active learning loop simulation
    for iteration in range(5):
        print(f"\n--- Active Learning Iteration {iteration + 1} ---")
        
        # 1. Refine features with current model
        refined_features = trainer.refine_features(features)
        
        # 2. Apply HDBSCAN clustering (you would do this)
        # from sklearn.cluster import HDBSCAN
        # clusterer = HDBSCAN(min_cluster_size=5, metric='euclidean')
        # cluster_labels = clusterer.fit_predict(refined_features)
        
        # 3. Present clustering results to user for correction (your existing interface)
        # corrected_labels = your_active_learning_interface(features, cluster_labels)
        
        # 4. Simulate getting some corrections from user
        # In practice, you'd get these from your interface
        n_corrections = 50
        correction_indices = np.random.choice(len(features), n_corrections, replace=False)
        
        # Simulate label corrections
        for idx in correction_indices:
            # User provides correct label for misclassified sample
            # labels[idx] = corrected_label  # This would come from your interface
            pass
        
        # 5. Retrain model with corrected labels
        print(f"Retraining with {n_corrections} corrections...")
        trainer.fit(features, labels, epochs=20, batch_size=32)
        
        # 6. Save checkpoint
        trainer.save_model(f'checkpoint_iteration_{iteration + 1}.pth')
    
    return trainer

if __name__ == "__main__":
    # Run example
    trainer = active_learning_pipeline_example()
    print("\nTraining completed!")