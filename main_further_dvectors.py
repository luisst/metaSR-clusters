import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

# Import torchinfo for model summary
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("Warning: torchinfo not available. Install with: pip install torchinfo")


def print_model_summary(model, input_size, model_name="Model"):
    """
    Print comprehensive model summary with trainable parameters per layer/block.

    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
        model_name: Name of the model for logging
    """

    def count_parameters(model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    # Create formatted headers
    header = f"\n{'='*100}\n{model_name.upper()} - COMPREHENSIVE ARCHITECTURE SUMMARY\n{'='*100}"
    print(header)
    logging.info(header)

    # Use torchinfo if available for detailed summary
    if TORCHINFO_AVAILABLE:
        print(f"\n{'-'*60}\nDETAILED ARCHITECTURE (torchinfo)\n{'-'*60}")
        logging.info(f"\nDETAILED ARCHITECTURE (torchinfo)")

        try:
            # Capture torchinfo summary
            summary_str = str(summary(
                model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                row_settings=["var_names"],
                verbose=2,
                depth=4
            ))
            print(summary_str)
            logging.info(summary_str)
        except Exception as e:
            print(f"Error generating torchinfo summary: {e}")
            logging.error(f"Error generating torchinfo summary: {e}")

    # Manual parameter analysis
    print(f"\n{'-'*60}\nPARAMETER ANALYSIS BY LAYER\n{'-'*60}")
    logging.info(f"\nPARAMETER ANALYSIS BY LAYER")

    # Header for parameter table
    param_header = f"{'Layer Name':<50} | {'Parameters':>12} | {'Trainable':>12} | {'Shape':>20} | {'Status'}"
    param_separator = "-" * 120
    print(param_header)
    print(param_separator)
    logging.info(param_header)
    logging.info(param_separator)

    total_params = 0
    trainable_params = 0

    # Analyze each parameter
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
            status = "✓ TRAINABLE"
            trainable_str = "YES"
        else:
            status = "✗ FROZEN"
            trainable_str = "NO"

        shape_str = str(list(param.shape))
        param_row = f"{name:<50} | {num_params:>12,} | {trainable_str:>12} | {shape_str:>20} | {status}"
        print(param_row)
        logging.info(param_row)

    # Parameter summary
    print(param_separator)
    logging.info(param_separator)
    summary_row = f"{'TOTAL':<50} | {total_params:>12,} | {trainable_params:>12,} | {'':<20} |"
    print(summary_row)
    logging.info(summary_row)

    # Module-wise parameter analysis
    print(f"\n{'-'*60}\nMODULE-WISE PARAMETER BREAKDOWN\n{'-'*60}")
    logging.info(f"\nMODULE-WISE PARAMETER BREAKDOWN")

    module_header = f"{'Module Name':<40} | {'Total':>10} | {'Trainable':>10} | {'Frozen':>10} | {'% Train':>8} | {'Description'}"
    module_separator = "-" * 120
    print(module_header)
    print(module_separator)
    logging.info(module_header)
    logging.info(module_separator)

    # Analyze modules
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # Only modules with parameters
            module_params = sum(p.numel() for p in module.parameters())
            trainable_module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen_module_params = module_params - trainable_module_params

            if module_params > 0:
                pct_trainable = 100 * trainable_module_params / module_params
                module_type = type(module).__name__

                module_row = f"{name or 'root':<40} | {module_params:>10,} | {trainable_module_params:>10,} | {frozen_module_params:>10,} | {pct_trainable:>7.1f}% | {module_type}"
                print(module_row)
                logging.info(module_row)

    # Final statistics
    print(f"\n{'-'*60}\nFINAL STATISTICS\n{'-'*60}")
    logging.info(f"\nFINAL STATISTICS")

    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    trainable_size_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 * 1024)

    stats = [
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Frozen parameters: {total_params - trainable_params:,}",
        f"Percentage trainable: {100 * trainable_params / total_params:.2f}%",
        f"Model size: {model_size_mb:.2f} MB",
        f"Trainable size: {trainable_size_mb:.2f} MB",
        f"Model type: {type(model).__name__}"
    ]

    for stat in stats:
        print(stat)
        logging.info(stat)

    print(f"{'='*100}")
    logging.info(f"{'='*100}")


def analyze_model_efficiency(model, model_name="Model"):
    """
    Analyze model efficiency and provide optimization suggestions.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    print(f"\n{'='*80}\n{model_name.upper()} - EFFICIENCY ANALYSIS\n{'='*80}")
    logging.info(f"\n{model_name.upper()} - EFFICIENCY ANALYSIS")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layer types and their parameter distribution
    layer_analysis = {}
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            layer_type = type(module).__name__
            layer_params = sum(p.numel() for p in module.parameters())
            
            if layer_type not in layer_analysis:
                layer_analysis[layer_type] = {'count': 0, 'total_params': 0, 'layers': []}
            
            layer_analysis[layer_type]['count'] += 1
            layer_analysis[layer_type]['total_params'] += layer_params
            layer_analysis[layer_type]['layers'].append((name, layer_params))
    
    # Print layer type analysis
    print(f"{'Layer Type':<20} | {'Count':>6} | {'Total Params':>12} | {'Avg Params':>12} | {'% of Model':>10}")
    print("-" * 80)
    logging.info(f"{'Layer Type':<20} | {'Count':>6} | {'Total Params':>12} | {'Avg Params':>12} | {'% of Model':>10}")
    
    for layer_type, info in sorted(layer_analysis.items(), key=lambda x: x[1]['total_params'], reverse=True):
        avg_params = info['total_params'] / info['count']
        percentage = 100 * info['total_params'] / total_params
        
        row = f"{layer_type:<20} | {info['count']:>6} | {info['total_params']:>12,} | {avg_params:>12,.0f} | {percentage:>9.1f}%"
        print(row)
        logging.info(row)
    
    # Efficiency metrics
    print(f"\n{'='*60}\nEFFICIENCY METRICS\n{'='*60}")
    logging.info(f"\nEFFICIENCY METRICS")
    
    efficiency_metrics = [
        f"Parameters per MB: {total_params / (sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)):,.0f}",
        f"Trainable ratio: {100 * trainable_params / total_params:.1f}%",
        f"Parameter density: {total_params / 1000:.1f}K parameters"
    ]
    
    for metric in efficiency_metrics:
        print(metric)
        logging.info(metric)
    
    # Suggestions for optimization
    print(f"\n{'='*60}\nOPTIMIZATION SUGGESTIONS\n{'='*60}")
    logging.info(f"\nOPTIMIZATION SUGGESTIONS")
    
    suggestions = []
    
    # Check for large Linear layers
    for layer_type, info in layer_analysis.items():
        if layer_type == 'Linear' and info['total_params'] > total_params * 0.3:
            suggestions.append(f"• Consider reducing Linear layer dimensions (currently {info['total_params']:,} params)")
    
    # Check parameter distribution
    if trainable_params < total_params * 0.8:
        suggestions.append(f"• Consider unfreezing more layers ({100 * (total_params - trainable_params) / total_params:.1f}% frozen)")
    
    # Check for potential bottlenecks
    if total_params > 1000000:  # 1M parameters
        suggestions.append("• Model is quite large - consider pruning or knowledge distillation")
    elif total_params < 100000:  # 100K parameters
        suggestions.append("• Model is compact - good for deployment")
    
    if not suggestions:
        suggestions.append("• Model appears well-optimized for the task")
    
    for suggestion in suggestions:
        print(suggestion)
        logging.info(suggestion)
    
    print(f"{'='*80}")
    logging.info(f"{'='*80}")


# Optimized training setup
trainer_config = {
    'lr': 0.0005,  # Slightly lower learning rate
    'weight_decay': 5e-4,  # Moderate regularization
    'batch_size': 32,  # Smaller batches for better generalization
    'patience': 8,  # Early stopping patience
    'scheduler_step': 15,  # LR decay every 15 epochs
    'scheduler_gamma': 0.7  # LR decay factor
}

#TODO: add trainer_config to FeatureEnhancementTrainer

class FeatureEnhancementNetwork(nn.Module):
    """
    A network designed to enhance speaker features from a pre-trained ResNet model.
    Uses residual connections and attention mechanisms to improve feature quality
    while maintaining the same dimensionality.
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, dropout_rate=0.3):
        super(FeatureEnhancementNetwork, self).__init__()
        
        # Feature enhancement blocks with residual connections
        self.enhancement_block1 = self._make_enhancement_block(input_dim, hidden_dim, dropout_rate)
        self.enhancement_block2 = self._make_enhancement_block(hidden_dim, hidden_dim, dropout_rate)
        self.enhancement_block3 = self._make_enhancement_block(hidden_dim, hidden_dim, dropout_rate)
        
        # Attention mechanism to focus on important features
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection back to original dimensionality
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(input_dim, input_dim)
        )
        
        # Residual connection from input to output
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def _make_enhancement_block(self, input_dim, hidden_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        # Store original input for residual connection
        residual_input = x
        
        # Pass through enhancement blocks
        x = self.enhancement_block1(x)
        x = self.enhancement_block2(x)
        x = self.enhancement_block3(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Project back to original dimensionality
        enhanced_features = self.output_projection(x)
        
        # Add residual connection with learnable weight
        output = enhanced_features + self.residual_weight * residual_input
        
        return output


class LightweightFeatureEnhancer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        
        # Simpler architecture
        self.enhancer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Lightweight attention
        self.attention = nn.Linear(input_dim, input_dim)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        enhanced = self.enhancer(x)
        attention_weights = torch.sigmoid(self.attention(enhanced))
        enhanced = enhanced * attention_weights
        return enhanced + self.residual_weight * x


class MinimalEnhancer(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        # Only ~131K parameters instead of 500K+
        self.enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, input_dim)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        return self.enhancer(x) * 0.1 + x * 0.9  # Heavy residual bias


class OptimalFeatureEnhancer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=384, dropout_rate=0.25):
        super().__init__()
        
        # Two-layer enhancement with moderate complexity
        self.enhancer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        
        # Lightweight attention (much simpler than before)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        enhanced = self.enhancer(x)
        attention_weights = self.attention(enhanced)
        enhanced = enhanced * attention_weights
        
        # Strong residual connection
        return enhanced + self.residual_weight * x


class LightweightSpeakerClassifier(nn.Module):
    """
    Lightweight classifier optimized for the enhanced features and limited data scenario.
    Designed to work well with ~30K effective samples and 90 speakers.
    """
    
    def __init__(self, input_dim=256, num_speakers=90, hidden_dim=320, dropout_rate=0.3):
        super().__init__()
        
        # Two-layer classifier with moderate complexity
        self.classifier = nn.Sequential(
            # First layer: dimension expansion
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer: dimension reduction
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),  # Slightly less dropout in final layer
            
            # Output layer
            nn.Linear(hidden_dim // 2, num_speakers)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)
    
    def forward_with_features(self, x):
        """Forward pass that also returns intermediate features for analysis."""
        # First layer
        x1 = self.classifier[0](x)  # Linear
        x1 = self.classifier[1](x1)  # BatchNorm
        x1 = self.classifier[2](x1)  # ReLU
        features_1 = x1.clone()
        x1 = self.classifier[3](x1)  # Dropout
        
        # Second layer
        x2 = self.classifier[4](x1)  # Linear
        x2 = self.classifier[5](x2)  # BatchNorm
        x2 = self.classifier[6](x2)  # ReLU
        features_2 = x2.clone()
        x2 = self.classifier[7](x2)  # Dropout
        
        # Output layer
        output = self.classifier[8](x2)  # Linear
        
        return output, {'layer1_features': features_1, 'layer2_features': features_2}


class UltraLightClassifier(nn.Module):
    def __init__(self, input_dim=256, num_speakers=90, hidden_dim=128, dropout_rate=0.25):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_speakers)
        )
    
    def forward(self, x):
        return self.classifier(x)


class AdaptiveClassifier(nn.Module):
    """
    Classifier that adapts to the quality of input features.
    Since your cached features are already good, this focuses on optimal classification.
    """
    
    def __init__(self, input_dim=256, num_speakers=90, use_attention=True):
        super().__init__()
        
        # Feature attention (to focus on most discriminative dimensions)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim),
                nn.Sigmoid()
            )
        
        # Main classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_speakers)
        )
        
        # Skip connection for feature preservation
        self.skip_projection = nn.Linear(input_dim, num_speakers)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x_attended = x * attention_weights
        else:
            x_attended = x
        
        # Main path
        main_output = self.classifier(x_attended)
        
        # Skip connection
        skip_output = self.skip_projection(x)
        
        # Combine outputs
        return main_output + self.skip_weight * skip_output


class MinimalOptimalClassifier(nn.Module):
    """
    Minimal classifier optimized for your specific case.
    """
    
    def __init__(self, input_dim=256, num_speakers=90, hidden_dim=384):
        super().__init__()
        
        # Single hidden layer with optimal size
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_speakers)
        )
        
        # Feature normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        x = self.input_norm(x)
        return self.classifier(x)


class GravitationalMetricLoss(nn.Module):
    """
    Gravity-inspired metric learning loss where the loss contribution increases
    as samples get closer to their class centroids (inverse square law).
    """
    
    def __init__(self, feature_dim=256, num_classes=90, alpha=1.0, beta=0.1, 
                 margin=1.0, gravity_power=2.0, epsilon=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for attraction term
        self.beta = beta    # Weight for repulsion term
        self.margin = margin  # Minimum margin between different classes
        self.gravity_power = gravity_power  # Power for inverse distance (default: 2 for inverse square)
        self.epsilon = epsilon  # Small constant to avoid division by zero
        
        # Learnable class centroids (prototypes)
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.centroids)
        
    def update_centroids(self, features, labels, momentum=0.9):
        """
        Update centroids using exponential moving average.
        """
        with torch.no_grad():
            for class_id in range(self.num_classes):
                mask = (labels == class_id)
                if mask.sum() > 0:
                    class_features = features[mask]
                    new_centroid = class_features.mean(dim=0)
                    self.centroids[class_id] = (momentum * self.centroids[class_id] + 
                                              (1 - momentum) * new_centroid)
    
    def compute_distances(self, features, centroids):
        """
        Compute L2 distances between features and centroids.
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
        
        # Attraction loss: pull samples toward their true class centroid
        # Higher force (loss) when closer to centroid
        positive_distances = distances[torch.arange(batch_size), labels]
        positive_forces = forces[torch.arange(batch_size), labels]
        attraction_loss = (positive_forces * positive_distances).mean()
        
        # Repulsion loss: push samples away from other class centroids
        # But with margin - only penalize if too close to wrong centroids
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
        
        # Total loss
        total_loss = self.alpha * attraction_loss + self.beta * repulsion_loss
        
        # Compute metrics for monitoring
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
    with gravity-inspired loss.
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature transformation network
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
        
        # Residual connection
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, output_dim)
        
        self.residual_weight = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, x):
        enhanced = self.feature_net(x)
        residual = self.residual(x)
        
        # Combine with learnable residual weight
        output = enhanced + self.residual_weight * residual
        
        # L2 normalize output for better metric learning
        output = nn.functional.normalize(output, p=2, dim=1)
        
        return output


class GravitationalMetricTrainer:
    """
    Trainer for gravity-inspired metric learning with incremental sample addition.
    """
    
    def __init__(self, enhancer, metric_loss, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=0.001, weight_decay=1e-4, centroid_update_freq=10):
        self.enhancer = enhancer.to(device)
        self.metric_loss = metric_loss.to(device)
        self.device = device
        
        # Optimizer for enhancer network
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
        
        # Update centroids periodically
        if update_centroids and self.epoch_count % self.centroid_update_freq == 0:
            self.metric_loss.update_centroids(enhanced_features.detach(), labels)
        
        return loss.item(), metrics, enhanced_features.detach()
    
    def add_new_samples(self, new_features, new_labels, num_epochs=10, verbose=True):
        """
        Add new samples and adapt the network incrementally.
        This is the key method for your incremental learning idea.
        """
        if verbose:
            print(f"\nAdding {len(new_features)} new samples...")
            logging.info(f"Adding {len(new_features)} new samples...")
        
        # Convert to tensors
        if isinstance(new_features, np.ndarray):
            new_features = torch.FloatTensor(new_features)
        if isinstance(new_labels, np.ndarray):
            new_labels = torch.LongTensor(new_labels)
        
        # Create a small dataset for the new samples
        new_dataset = AudioDataset(new_features, new_labels)
        new_loader = DataLoader(new_dataset, batch_size=16, shuffle=True)
        
        # Train on new samples with increased attention to gravity forces
        original_alpha = self.metric_loss.alpha
        original_beta = self.metric_loss.beta
        
        # Increase gravity strength for new samples
        self.metric_loss.alpha *= 1.5  # Stronger attraction
        self.metric_loss.beta *= 1.2   # Stronger repulsion
        
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
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                logging.info(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Restore original loss weights
        self.metric_loss.alpha = original_alpha
        self.metric_loss.beta = original_beta
        
        if verbose:
            print(f"New samples integrated. Final loss: {epoch_losses[-1]:.4f}")
            logging.info(f"New samples integrated. Final loss: {epoch_losses[-1]:.4f}")
        
        return epoch_losses
    
    def train_epoch(self, data_loader, update_centroids=True):
        """
        Train for one full epoch.
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
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        metrics = {}
        
        try:
            # Silhouette score
            metrics['silhouette_score'] = silhouette_score(all_features, all_labels)
            
            # KNN accuracy
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
        """
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            centroids = self.metric_loss.centroids.detach().cpu().numpy()
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            centroids_2d = pca.fit_transform(centroids)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                                c=range(len(centroids_2d)), cmap='tab20', s=50)
            plt.colorbar(scatter, label='Speaker ID')
            plt.title('Learned Speaker Centroids (PCA Projection)')
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
        Get statistics about the current state of centroids and features.
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


class FeatureEnhancementTrainer:
    """
    Training class for the feature enhancement network.
    """
    
    def __init__(self, enhancement_net, classifier, output_folder=None, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 enable_augmentation=True, noise_std=0.01, mask_prob=0.1):

        # self.enhancement_net = enhancement_net.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.output_folder = output_folder
        
        # Data augmentation
        self.enable_augmentation = enable_augmentation
        if self.enable_augmentation:
            self.augmentor = DataAugmentor(noise_std=noise_std, mask_prob=mask_prob)
        
        ### Optimizers
        # self.enhancement_optimizer = optim.Adam(self.enhancement_net.parameters(), lr=0.001, weight_decay=1e-4)
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=1e-4)
        
        ### Schedulers
        # self.enhancement_scheduler = optim.lr_scheduler.StepLR(self.enhancement_optimizer, step_size=10, gamma=0.8)
        self.classifier_scheduler = optim.lr_scheduler.StepLR(self.classifier_optimizer, step_size=10, gamma=0.8)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, val_loader):
        """Train for one epoch."""
        # self.enhancement_net.train()
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # # Apply additional augmentation during training (on top of dataset augmentation)
            # if self.enable_augmentation and self.enhancement_net.training:

            # ### Apply augmentation with 50% probability
            # if torch.rand(1).item() < 0.5:
            #     features = self.augmentor.augment(features)
            
            ### Forward pass
            # enhanced_features = self.enhancement_net(features)
            # outputs = self.classifier(enhanced_features)

            outputs = self.classifier(features)

            loss = self.criterion(outputs, labels)
            
            ### Backward pass
            # self.enhancement_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            loss.backward()
            # self.enhancement_optimizer.step()
            self.classifier_optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc = self.evaluate(val_loader)
        
        ### Update learning rates
        # self.enhancement_scheduler.step()
        self.classifier_scheduler.step()
        
        # Store history
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        
        return train_loss, train_acc, val_loss, val_acc
    
    def evaluate(self, data_loader):
        """Evaluate the model."""

        # self.enhancement_net.eval()
        self.classifier.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # enhanced_features = self.enhancement_net(features)
                # outputs = self.classifier(enhanced_features)

                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop."""
        logging.info("Starting training...")
        print("Starting training...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc, val_loss, val_acc = self.train_epoch(train_loader, val_loader)
            
            epoch_info = f'Epoch {epoch+1}/{epochs}:'
            train_info = f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
            val_info = f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            # lr_info = f'  Learning Rate: {self.enhancement_optimizer.param_groups[0]["lr"]:.6f}'
            
            print(epoch_info)
            print(train_info)
            print(val_info)
            # print(lr_info)
            
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            # logging.info(lr_info)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # self.save_model('best_enhancement_model.pth')
                best_info = f'  New best validation accuracy: {best_val_acc:.2f}%'
                print(best_info)
                logging.info(best_info)
            
            print('-' * 50)
            logging.info('-' * 50)
        
        final_info = f'Training completed. Best validation accuracy: {best_val_acc:.2f}%'
        print(final_info)
        logging.info(final_info)
        self.plot_training_history()
    
    def save_model(self, path):
        """Save the trained models."""
        # If output_folder is set, save in that folder
        if self.output_folder:
            path = Path(self.output_folder) / path
        
        torch.save({
            'enhancement_net': self.enhancement_net.state_dict(),
            'classifier': self.classifier.state_dict(),
            'enhancement_optimizer': self.enhancement_optimizer.state_dict(),
            'classifier_optimizer': self.classifier_optimizer.state_dict(),
        }, path)
        
        logging.info(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load trained models."""
        checkpoint = torch.load(path, map_location=self.device)
        # self.enhancement_net.load_state_dict(checkpoint['enhancement_net'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        # self.enhancement_optimizer.load_state_dict(checkpoint['enhancement_optimizer'])
        self.classifier_optimizer.load_state_dict(checkpoint['classifier_optimizer'])
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot in output folder if specified
        if self.output_folder:
            plot_path = Path(self.output_folder) / 'training_history.png'
            plt.savefig(plot_path)
            logging.info(f"Training history plot saved to: {plot_path}")
        else:
            plt.savefig('training_history.png')
            
        plt.show()

    def print_training_setup(self):
        """Print comprehensive training setup information."""
        setup_header = f"\n{'='*80}\nTRAINING SETUP SUMMARY\n{'='*80}"
        print(setup_header)
        logging.info(setup_header)
        
        ### Print model summaries
        # print_model_summary(self.enhancement_net, input_size=(256,), model_name="Enhancement Network")
        print_model_summary(self.classifier, input_size=(256,), model_name="Speaker Classifier")
        
        ### Analyze model efficiency
        # analyze_model_efficiency(self.enhancement_net, "Enhancement Network")
        analyze_model_efficiency(self.classifier, "Speaker Classifier")
        
        # Print optimizer information
        #optimizer_info = f"""
# {'-'*60}
# OPTIMIZER CONFIGURATION
# {'-'*60}
# Enhancement Network Optimizer: {type(self.enhancement_optimizer).__name__}
#   - Learning Rate: {self.enhancement_optimizer.param_groups[0]['lr']:.6f}
#   - Weight Decay: {self.enhancement_optimizer.param_groups[0]['weight_decay']:.6f}

# Speaker Classifier Optimizer: {type(self.classifier_optimizer).__name__}
#   - Learning Rate: {self.classifier_optimizer.param_groups[0]['lr']:.6f}
#   - Weight Decay: {self.classifier_optimizer.param_groups[0]['weight_decay']:.6f}

# Learning Rate Schedulers:
#   - Enhancement: {type(self.enhancement_scheduler).__name__}
#   - Classifier: {type(self.classifier_scheduler).__name__}

# Loss Function: {type(self.criterion).__name__}
# Device: {self.device}
# Augmentation: {'Enabled' if self.enable_augmentation else 'Disabled'}
# """
#         print(optimizer_info)
#         logging.info(optimizer_info)
        
#         if self.enable_augmentation:
#             aug_info = f"""
# Augmentation Settings:
#   - Noise Std: {self.augmentor.noise_std}
#   - Mask Probability: {self.augmentor.mask_prob}
# """
#             print(aug_info)
#             logging.info(aug_info)
        
#         print(f"{'='*80}")
#         logging.info(f"{'='*80}")


class SpeakerInference:
    """
    Inference class for the enhanced speaker features.
    """
    
    def __init__(self, enhancement_net, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.enhancement_net = enhancement_net.to(device)
        self.device = device
        self.enhancement_net.eval()
    
    def enhance_features(self, features):
        """
        Enhance input features using the trained network.
        
        Args:
            features: numpy array or torch tensor of shape (batch_size, 256) or (256,)
            
        Returns:
            Enhanced features as numpy array
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension if needed
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            enhanced_features = self.enhancement_net(features)
        
        return enhanced_features.cpu().numpy()
    
    def batch_enhance_features(self, features_list):
        """
        Enhance a list of features in batches.
        
        Args:
            features_list: List of numpy arrays or single numpy array
            
        Returns:
            List of enhanced features
        """
        if isinstance(features_list, np.ndarray):
            return self.enhance_features(features_list)
        
        enhanced_list = []
        for features in features_list:
            enhanced = self.enhance_features(features)
            enhanced_list.append(enhanced.squeeze())
        
        return enhanced_list


class DataAugmentor:
    """
    Data augmentation class for speaker features.
    """
    
    def __init__(self, noise_std=0.01, mask_prob=0.1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
    
    def add_noise(self, features):
        """
        Add Gaussian noise to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            
        Returns:
            Noisy features
        """
        noise = torch.randn_like(features) * self.noise_std
        return features + noise
    
    def apply_mask(self, features):
        """
        Apply random masking to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            
        Returns:
            Masked features
        """
        mask = torch.rand_like(features) > self.mask_prob
        return features * mask.float()
    
    def augment(self, features, apply_noise=True, apply_mask=True):
        """
        Apply data augmentation to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            apply_noise: whether to add noise
            apply_mask: whether to apply masking
            
        Returns:
            Augmented features
        """
        augmented = features.clone()
        
        if apply_noise:
            augmented = self.add_noise(augmented)
        
        if apply_mask:
            augmented = self.apply_mask(augmented)
            
        return augmented


class AudioDataset(Dataset):
    def __init__(self, features, labels, augmentor=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augmentor = augmentor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        
        # Apply augmentation if augmentor is provided
        if self.augmentor is not None:
            features = self.augmentor.augment(features)
            
        return features, labels


def create_dataloaders(pickle_file_path, batch_size=32, test_size=0.2, random_state=42, num_workers=4, 
                      augment_training=True, noise_std=0.01, mask_prob=0.1):
    """
    Create training and validation DataLoaders from a pickle file.
    
    Args:
        pickle_file_path (str): Path to the pickle file containing 'features', 'wav_paths', 'labels'
        batch_size (int): Batch size for DataLoaders
        test_size (float): Proportion of data to use for validation (default: 0.2 for 80-20 split)
        random_state (int): Random seed for reproducible splits
        num_workers (int): Number of worker processes for data loading
        augment_training (bool): Whether to apply data augmentation to training data
        noise_std (float): Standard deviation for Gaussian noise in augmentation
        mask_prob (float): Probability of masking features in augmentation
    
    Returns:
        tuple: (train_loader, val_loader)
    """

    # Load the pickle file
    with open(pickle_file_path, 'rb') as f:
        features, wavs_paths, labels = pickle.load(f)

    # Convert labels from 1 - 90 to 0 - 89
    labels = np.array(labels) - 1  # Assuming labels are 1-indexed

    feature_dim = features.shape[1]
    num_speakers = len(set(labels))  # Assuming labels are speaker IDs

    # Convert to numpy arrays if they aren't already
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Split the data with shuffling
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True,
        stratify=None  # Ensures balanced split across classes
    )
    
    # Create augmentor for training data if requested
    augmentor = DataAugmentor(noise_std=noise_std, mask_prob=mask_prob) if augment_training else None
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train, augmentor=augmentor)
    val_dataset = AudioDataset(X_val, y_val, augmentor=None)  # No augmentation for validation
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Additional shuffling for training
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    train_samples_msg = f"Training samples: {len(train_dataset)}"
    val_samples_msg = f"Validation samples: {len(val_dataset)}"
    features_shape_msg = f"Features shape: {features.shape}"
    labels_shape_msg = f"Labels shape: {labels.shape}"
    augmentation_msg = f"Data augmentation: {'Enabled' if augment_training else 'Disabled'}"
    
    print(train_samples_msg)
    print(val_samples_msg)
    print(features_shape_msg)
    print(labels_shape_msg)
    print(augmentation_msg)
    
    logging.info(train_samples_msg)
    logging.info(val_samples_msg)
    logging.info(features_shape_msg)
    logging.info(labels_shape_msg)
    logging.info(augmentation_msg)
    
    if augment_training:
        noise_msg = f"  - Noise std: {noise_std}"
        mask_msg = f"  - Mask probability: {mask_prob}"
        print(noise_msg)
        print(mask_msg)
        logging.info(noise_msg)
        logging.info(mask_msg)

    return train_loader, val_loader, feature_dim, num_speakers

def demonstrate_gravitational_metric_learning():
    """
    Demonstrate the gravity-inspired metric learning approach with incremental sample addition.
    """
    print("\n" + "="*80)
    print("GRAVITATIONAL METRIC LEARNING DEMONSTRATION")
    print("="*80)
    logging.info("GRAVITATIONAL METRIC LEARNING DEMONSTRATION")
    
    # Configuration
    FEATURE_DIM = 256
    NUM_SPEAKERS = 90
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    root_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO')
    mfcc_folder_ex = root_ex / Path('Dvectors/noisy_all_18K/input_feats')
    feats_pickle_path = mfcc_folder_ex.parent / Path('d_vectors_feats.pickle')
    
    print(f"Loading data from: {feats_pickle_path}")
    with open(feats_pickle_path, 'rb') as f:
        features, wav_paths, labels = pickle.load(f)
    
    labels = np.array(labels) - 1  # Convert to 0-indexed
    
    # Split into initial training set and "new samples" to add incrementally
    from sklearn.model_selection import train_test_split
    
    # Use 70% for initial training, 30% as "new samples"
    X_initial, X_new, y_initial, y_new = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Initial training samples: {len(X_initial)}")
    print(f"New samples to add incrementally: {len(X_new)}")
    logging.info(f"Initial training samples: {len(X_initial)}")
    logging.info(f"New samples to add incrementally: {len(X_new)}")
    
    # Create models
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
    
    print_model_summary(enhancer, input_size=(FEATURE_DIM,), model_name="Metric Learning Enhancer")
    
    # Create initial data loader
    initial_dataset = AudioDataset(X_initial, y_initial)
    initial_loader = DataLoader(initial_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Phase 1: Initial training
    print("\n" + "-"*60)
    print("PHASE 1: INITIAL TRAINING")
    print("-"*60)
    logging.info("PHASE 1: INITIAL TRAINING")
    
    for epoch in range(20):  # Initial training epochs
        loss, metrics = trainer.train_epoch(initial_loader)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/20:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Attraction Loss: {metrics['attraction_loss']:.4f}")
            print(f"  Repulsion Loss: {metrics['repulsion_loss']:.4f}")
            print(f"  Avg Positive Distance: {metrics['avg_positive_distance']:.4f}")
            print(f"  Separation Ratio: {metrics['avg_negative_distance']/metrics['avg_positive_distance']:.2f}")
            
            logging.info(f"Epoch {epoch+1}/20: Loss={loss:.4f}, "
                        f"Attraction={metrics['attraction_loss']:.4f}, "
                        f"Repulsion={metrics['repulsion_loss']:.4f}")
    
    # Evaluate initial model
    print("\nEvaluating initial model...")
    initial_metrics = trainer.evaluate_metric_quality(initial_loader)
    print("Initial Model Metrics:")
    for key, value in initial_metrics.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Initial {key}: {value:.4f}")
    
    # Phase 2: Incremental addition of new samples
    print("\n" + "-"*60)
    print("PHASE 2: INCREMENTAL SAMPLE ADDITION")
    print("-"*60)
    logging.info("PHASE 2: INCREMENTAL SAMPLE ADDITION")
    
    # Split new samples into batches for incremental addition
    batch_size_new = len(X_new) // 5  # Add in 5 batches
    
    for i in range(5):
        start_idx = i * batch_size_new
        end_idx = start_idx + batch_size_new if i < 4 else len(X_new)
        
        batch_features = X_new[start_idx:end_idx]
        batch_labels = y_new[start_idx:end_idx]
        
        print(f"\nAdding batch {i+1}/5 ({len(batch_features)} samples)...")
        
        # Add new samples with gravity-enhanced learning
        losses = trainer.add_new_samples(batch_features, batch_labels, num_epochs=15)
        
        print(f"  Integration complete. Final loss: {losses[-1]:.4f}")
        
        # Evaluate after each addition
        if i % 2 == 1:  # Evaluate every other batch
            combined_features = np.vstack([X_initial, X_new[:end_idx]])
            combined_labels = np.hstack([y_initial, y_new[:end_idx]])
            combined_dataset = AudioDataset(combined_features, combined_labels)
            combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            metrics = trainer.evaluate_metric_quality(combined_loader)
            print(f"  Metrics after batch {i+1}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")
    
    # Final evaluation
    print("\n" + "-"*60)
    print("FINAL EVALUATION")
    print("-"*60)
    logging.info("FINAL EVALUATION")
    
    # Create final combined dataset
    final_features = np.vstack([X_initial, X_new])
    final_labels = np.hstack([y_initial, y_new])
    final_dataset = AudioDataset(final_features, final_labels)
    final_loader = DataLoader(final_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    final_metrics = trainer.evaluate_metric_quality(final_loader, sample_size=1000)
    
    print("Final Model Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Final {key}: {value:.4f}")
    
    # Compare initial vs final
    print("\nImprovement Summary:")
    for key in initial_metrics:
        if key in final_metrics:
            improvement = final_metrics[key] - initial_metrics[key]
            print(f"  {key}: {improvement:+.4f}")
            logging.info(f"Improvement in {key}: {improvement:+.4f}")
    
    # Visualize centroids
    output_folder = feats_pickle_path.parent / 'gravitational_metric_output'
    output_folder.mkdir(exist_ok=True)
    
    centroid_plot_path = output_folder / 'learned_centroids.png'
    trainer.visualize_centroids(centroid_plot_path)
    
    # Get final statistics
    stats = trainer.get_feature_statistics()
    print("\nCentroid Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
        logging.info(f"Centroid {key}: {value:.4f}")
    
    print("\n" + "="*80)
    print("GRAVITATIONAL METRIC LEARNING DEMONSTRATION COMPLETE")
    print("="*80)
    logging.info("GRAVITATIONAL METRIC LEARNING DEMONSTRATION COMPLETE")
    
    return trainer, final_metrics


def main():
    """
    Main function to demonstrate different approaches.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speaker Recognition with Enhanced Features')
    parser.add_argument('--mode', choices=['standard', 'gravitational'], default='gravitational',
                       help='Training mode: standard classification or gravitational metric learning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=21, help='Random seed')
    
    # For now, let's just demonstrate the gravitational approach
    mode = 'gravitational'  # Change this to 'standard' to run the original approach
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED = 21  # For reproducibility

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == 'gravitational':
        print("\n" + "="*80)
        print("🌌 GRAVITATIONAL METRIC LEARNING MODE 🌌")
        print("="*80)
        print("Implementing gravity-inspired metric learning with incremental sample addition!")
        print("Key features:")
        print("- Inverse square law for loss weighting")
        print("- Learnable class centroids")
        print("- Incremental sample addition")
        print("- Enhanced feature separation")
        print("="*80)
        
        # Run the gravitational demonstration
        try:
            demonstrate_gravitational_metric_learning()
        except Exception as e:
            print(f"Error in gravitational metric learning: {e}")
            logging.error(f"Error in gravitational metric learning: {e}")
            # Fall back to standard mode
            print("Falling back to standard training mode...")
            mode = 'standard'
    
    if mode == 'standard':
        print("\n" + "="*80)
        print("STANDARD CLASSIFICATION MODE")
        print("="*80)
        
        root_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO')
        mfcc_folder_ex = root_ex / Path('Dvectors/noisy_all_18K/input_feats')
        feats_pickle_path = mfcc_folder_ex.parent / Path('d_vectors_feats.pickle')
        
        # Create output folder in the same directory as feats_pickle_path
        output_folder = feats_pickle_path.parent / 'output_standard_training'
        output_folder.mkdir(exist_ok=True)
        
        log_file = output_folder / f'training_log_{timestamp}.txt'
        
        # Configure logging to write to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Output folder created at: {output_folder}")
        logging.info(f"Log file: {log_file}")
        logging.info(f"Random seed set to: {SEED}")
        
        train_loader, val_loader, feature_dim, num_speakers = create_dataloaders(
            feats_pickle_path, 
            batch_size=64, 
            augment_training=True, 
            noise_std=0.01, 
            mask_prob=0.1
        )

        info_msg = f"Feature dimension: {feature_dim}, Number of speakers: {num_speakers}"
        print(info_msg)
        logging.info(info_msg)

        # Analyze the quality of cached features
        with open(feats_pickle_path, 'rb') as f:
            features, wavs_paths, labels = pickle.load(f)
        
        labels = np.array(labels) - 1  # Convert to 0-indexed
        analyze_feature_quality(features, labels, "Cached ResNet Features")

        # Create models
        enhancement_net = FeatureEnhancementNetwork(input_dim=feature_dim)
        classifier = LightweightSpeakerClassifier(input_dim=feature_dim, num_speakers=num_speakers)

        # Create trainer with augmentation enabled
        trainer = FeatureEnhancementTrainer(enhancement_net, classifier, output_folder=output_folder,
                                          enable_augmentation=True, noise_std=0.01, mask_prob=0.1)
        
        # Print comprehensive training setup before starting
        trainer.print_training_setup()
        
        # Train the model
        trainer.train(train_loader, val_loader, epochs=EPOCHS)
        
        final_model_msg = f"Final model saved in output folder: {output_folder}"
        print(final_model_msg)
        logging.info(final_model_msg)
        
        logging.info("Training pipeline completed successfully!")





def visualize_gravitational_effect(distances, gravity_power=2.0, epsilon=1e-6):
    """
    Visualize how the gravitational force changes with distance.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create a range of distances
        distance_range = np.linspace(0.1, 5.0, 100)
        
        # Compute gravitational forces for different powers
        powers = [1.0, 2.0, 3.0]  # Linear, inverse square, inverse cube
        
        plt.figure(figsize=(12, 8))
        
        # Plot gravitational forces
        plt.subplot(2, 2, 1)
        for power in powers:
            forces = 1.0 / (np.power(distance_range + epsilon, power))
            plt.plot(distance_range, forces, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance')
        plt.ylabel('Gravitational Force')
        plt.title('Gravitational Force vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot loss contribution (force × distance)
        plt.subplot(2, 2, 2)
        for power in powers:
            forces = 1.0 / (np.power(distance_range + epsilon, power))
            loss_contribution = forces * distance_range
            plt.plot(distance_range, loss_contribution, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance')
        plt.ylabel('Loss Contribution (Force × Distance)')
        plt.title('Loss Contribution vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot derivative (how loss changes with distance)
        plt.subplot(2, 2, 3)
        for power in powers:
            # Derivative of force × distance
            derivative = (1 - power) / (np.power(distance_range + epsilon, power))
            plt.plot(distance_range, derivative, label=f'Power = {power}', linewidth=2)
        
        plt.xlabel('Distance')
        plt.ylabel('d(Loss)/d(Distance)')
        plt.title('Loss Gradient vs Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Conceptual diagram
        plt.subplot(2, 2, 4)
        
        # Create a simple 2D visualization of the concept
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # Simulate gravitational field around a centroid at origin
        R = np.sqrt(X**2 + Y**2)
        Z = 1.0 / (R + 0.1)**2  # Gravitational potential
        
        contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')
        plt.colorbar(contour, label='Gravitational Potential')
        plt.plot(0, 0, 'ro', markersize=10, label='Centroid')
        
        # Add sample points
        sample_points_x = [1.5, -1.2, 0.8, -0.5]
        sample_points_y = [0.5, 1.8, -1.5, -2.0]
        plt.scatter(sample_points_x, sample_points_y, c='red', s=50, alpha=0.7, label='Samples')
        
        plt.xlabel('Feature Dimension 1')
        plt.ylabel('Feature Dimension 2')
        plt.title('Gravitational Field Around Centroid')
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.suptitle('Gravitational Metric Learning Visualization', y=1.02, fontsize=14, fontweight='bold')
        
        plt.show()
        
        print("\nGravitational Effect Analysis:")
        print("- Inverse square law (power=2) provides strong attraction when close, weak when far")
        print("- Loss contribution peaks at intermediate distances")
        print("- Gradient becomes negative for power > 1, encouraging movement toward centroid")
        print("- The field visualization shows how samples are attracted to their centroids")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def analyze_feature_quality(features, labels, model_name="Features"):
    """
    Analyze the quality and discriminative power of features.
    
    Args:
        features: numpy array of features (N, feature_dim)
        labels: numpy array of labels (N,)
        model_name: name for logging
    """
    print(f"\n{'='*80}\n{model_name.upper()} - FEATURE QUALITY ANALYSIS\n{'='*80}")
    logging.info(f"\n{model_name.upper()} - FEATURE QUALITY ANALYSIS")
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    # Basic statistics
    print(f"Feature Statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std: {np.std(features):.4f}")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    
    # Check for potential issues
    zero_variance_dims = np.var(features, axis=0) < 1e-8
    print(f"  Zero variance dimensions: {np.sum(zero_variance_dims)}")
    
    # Dimensionality analysis
    pca = PCA()
    pca.fit(features)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    print(f"\nDimensionality Analysis:")
    print(f"  Variance explained by first 50 components: {np.sum(explained_variance_ratio[:50]):.3f}")
    print(f"  Variance explained by first 100 components: {np.sum(explained_variance_ratio[:100]):.3f}")
    print(f"  Effective dimensionality (95% variance): {np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1}")
    
    # Separability analysis
    try:
        # Sample subset for computational efficiency
        n_samples = min(1000, len(features))
        indices = np.random.choice(len(features), n_samples, replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]
        
        silhouette_avg = silhouette_score(sample_features, sample_labels)
        print(f"\nSeparability Analysis:")
        print(f"  Silhouette Score: {silhouette_avg:.3f} (higher is better, max=1.0)")
        
        # Intra-class vs inter-class distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(sample_features)
        
        intra_class_distances = []
        inter_class_distances = []
        
        for i in range(len(sample_features)):
            for j in range(i+1, len(sample_features)):
                if sample_labels[i] == sample_labels[j]:
                    intra_class_distances.append(distances[i, j])
                else:
                    inter_class_distances.append(distances[i, j])
        
        if intra_class_distances and inter_class_distances:
            print(f"  Avg intra-class distance: {np.mean(intra_class_distances):.3f}")
            print(f"  Avg inter-class distance: {np.mean(inter_class_distances):.3f}")
            print(f"  Separation ratio: {np.mean(inter_class_distances) / np.mean(intra_class_distances):.3f}")
    
    except Exception as e:
        print(f"  Separability analysis failed: {e}")
    
    print(f"{'='*80}")
    logging.info(f"{'='*80}")


def compare_feature_representations(original_features, enhanced_features, labels, sample_size=1000):
    """
    Compare original vs enhanced features to see if enhancement is beneficial.
    """
    print(f"\n{'='*80}\nFEATURE ENHANCEMENT COMPARISON\n{'='*80}")
    logging.info(f"\nFEATURE ENHANCEMENT COMPARISON")
    
    # Sample for computational efficiency
    n_samples = min(sample_size, len(original_features))
    indices = np.random.choice(len(original_features), n_samples, replace=False)
    
    orig_sample = original_features[indices]
    enh_sample = enhanced_features[indices]
    labels_sample = labels[indices]
    
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    # Silhouette scores
    try:
        orig_silhouette = silhouette_score(orig_sample, labels_sample)
        enh_silhouette = silhouette_score(enh_sample, labels_sample)
        
        print(f"Silhouette Scores:")
        print(f"  Original features: {orig_silhouette:.3f}")
        print(f"  Enhanced features: {enh_silhouette:.3f}")
        print(f"  Improvement: {enh_silhouette - orig_silhouette:.3f}")
    except Exception as e:
        print(f"Silhouette comparison failed: {e}")
    
    # KNN classification performance
    try:
        knn = KNeighborsClassifier(n_neighbors=5)
        
        orig_scores = cross_val_score(knn, orig_sample, labels_sample, cv=3)
        enh_scores = cross_val_score(knn, enh_sample, labels_sample, cv=3)
        
        print(f"\nKNN Classification (k=5):")
        print(f"  Original features: {np.mean(orig_scores):.3f} ± {np.std(orig_scores):.3f}")
        print(f"  Enhanced features: {np.mean(enh_scores):.3f} ± {np.std(enh_scores):.3f}")
        print(f"  Improvement: {np.mean(enh_scores) - np.mean(orig_scores):.3f}")
    except Exception as e:
        print(f"KNN comparison failed: {e}")
    
    # Feature correlation analysis
    correlation = np.corrcoef(orig_sample.flatten(), enh_sample.flatten())[0, 1]
    print(f"\nFeature Correlation: {correlation:.3f}")
    
    if correlation > 0.95:
        print("  → Enhancement provides minimal change (high correlation)")
    elif correlation > 0.8:
        print("  → Enhancement provides moderate change")
    else:
        print("  → Enhancement provides significant change")
    
    print(f"{'='*80}")
    logging.info(f"{'='*80}")

if __name__ == "__main__":
    main()