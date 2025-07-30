import torch
import torch.nn.functional as F

def gravitational_loss_normalized(x, labels, lambda_repel=1.0, epsilon=1e-6):
    """
    x: Tensor of shape (batch_size, embedding_dim)
    labels: LongTensor of shape (batch_size,)
    lambda_repel: Weight for the repulsion term
    """
    # Normalize embeddings to unit norm (L2)
    x = F.normalize(x, p=2, dim=1)
    unique_labels = labels.unique()
    
    # Precompute centroids for each class in the batch
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        class_embeddings = x[mask]
        centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        centroids[lbl.item()] = centroid

    pull_loss = 0.0
    repel_loss = 0.0

    for i in range(x.size(0)):
        xi = x[i].unsqueeze(0)  # shape: (1, embedding_dim)
        yi = labels[i].item()

        # Pull term (inverse square distance to own centroid)
        centroid_pos = centroids[yi]
        dist_sq = torch.sum((xi - centroid_pos) ** 2)
        pull_loss += 1.0 / (dist_sq + epsilon)

        # Repel term: inverse square distance to all *other* centroids
        for yj, centroid_neg in centroids.items():
            if yj == yi:
                continue
            dist = torch.norm(xi - centroid_neg, p=2)
            repel_loss += 1.0 / ((dist + epsilon) ** 2)

    pull_loss /= x.size(0)
    repel_loss /= x.size(0)

    total_loss = pull_loss + lambda_repel * repel_loss
    return total_loss



def gravitational_loss1(x, labels, epsilon=1e-6):
    # x: (batch_size, embedding_dim)
    # labels: (batch_size,)
    unique_labels = labels.unique()
    loss = 0.0
    
    for lbl in unique_labels:
        mask = labels == lbl
        x_pos = x[mask]
        x_neg = x[~mask]

        centroid_pos = x_pos.mean(dim=0)
        
        # Pull loss
        pull = ((x_pos - centroid_pos)**2).sum(dim=1)
        pull_loss = (1.0 / (pull + epsilon)).mean()

        # Repel loss
        for other_lbl in unique_labels:
            if other_lbl == lbl:
                continue
            mask_other = labels == other_lbl
            centroid_neg = x[mask_other].mean(dim=0)
            repel = ((x_pos - centroid_neg)**2).sum(dim=1)
            repel_loss = (1.0 / (repel + epsilon)).mean()
            loss += repel_loss
        
        loss += pull_loss
        
    return loss / len(unique_labels)
