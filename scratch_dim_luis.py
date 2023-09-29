import torch
import torch.nn.functional as F

# Example 2: Cosine similarity between two 2D tensors
tensor2d_a = torch.tensor([[1.0, 2.0, 6.0], [3.0, 4.0, 7.0]])
tensor2d_b = torch.tensor([[5.0, 6.0, 3.0], [7.0, 8.0, 9.0]])

# Compute cosine similarity between the two 2D tensors along dimension 1 (columns)
cos_sim_2d = F.cosine_similarity(tensor2d_a, tensor2d_b, dim=1)
print("Cosine Similarity (2D):", cos_sim_2d.tolist())