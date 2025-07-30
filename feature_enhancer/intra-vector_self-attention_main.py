import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D

def load_snapshots(folder_path):
    files = sorted(Path(folder_path).glob("epoch*_val.pt"))
    all_embeddings = []
    all_labels = []

    for file in files:
        data = torch.load(file)
        all_embeddings.append(data['embeddings'])
        all_labels.append(data['labels'])

    return all_embeddings, all_labels, [f.stem for f in files]

def create_embedding_animation(
    embeddings_list,
    labels_list,
    titles,
    method='umap',
    save_path='embedding_animation_markers.mp4',
    max_classes=10,
):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    reducer.fit(embeddings_list[0].numpy())
    projected = [reducer.transform(e.numpy()) for e in embeddings_list]

    all_labels_flat = torch.cat(labels_list).unique(sorted=True).tolist()
    n_classes = min(len(all_labels_flat), max_classes)
    label_to_index = {label: idx for idx, label in enumerate(all_labels_flat[:n_classes])}

    colors = plt.cm.tab10.colors
    markers = ['o', 'v', 's', '^', '<', '>', 'D', 'P', '*', 'X']  # Extend if needed

    fig, ax = plt.subplots(figsize=(8, 6))
    scatters = []
    legend_elements = []

    for i in range(n_classes):
        sc = ax.scatter([], [], s=30, label=f"Speaker {i}",
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)])
        scatters.append(sc)
        legend_elements.append(Line2D([0], [0], marker=markers[i % len(markers)],
                                      color='w', label=f'Speaker {i}',
                                      markerfacecolor=colors[i % len(colors)], markersize=8))

    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)
    ax.legend(handles=legend_elements, loc='upper right', title="Speakers")

    def init():
        for sc in scatters:
            sc.set_offsets([])
        title.set_text('')
        return scatters + [title]

    def update(frame):
        coords = projected[frame]
        labels = labels_list[frame].numpy()

        for i in range(n_classes):
            idxs = np.where(labels == all_labels_flat[i])[0]
            if len(idxs) > 0:
                scatters[i].set_offsets(coords[idxs])
            else:
                scatters[i].set_offsets([])

        title.set_text(f"Epoch: {titles[frame]}")
        return scatters + [title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(projected),
        init_func=init, blit=False, interval=1000, repeat=False
    )

    ani.save(save_path, writer='ffmpeg', dpi=150)
    print(f"Saved animation with markers to {save_path}")


def visualize_embeddings(embeddings, labels, method='tsne', title='Embedding Space'):
    """
    embeddings: Tensor of shape (N, D)
    labels: Tensor of shape (N,)
    method: 'tsne' or 'umap'
    """
    X = embeddings.numpy()
    y = labels.numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random')
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.colorbar(scatter, label='Speaker Label')
    plt.grid(True)
    plt.show()


class MemoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.embeddings = []
        self.labels = []

    def add_samples(self, new_embeddings, new_labels):
        # Append new samples
        self.embeddings.extend(new_embeddings)
        self.labels.extend(new_labels)

        # Limit size if necessary
        if len(self.embeddings) > self.max_size:
            zipped = list(zip(self.embeddings, self.labels))
            random.shuffle(zipped)
            zipped = zipped[:self.max_size]
            self.embeddings, self.labels = zip(*zipped)
            self.embeddings = list(self.embeddings)
            self.labels = list(self.labels)

    def get_buffer_loader(self, batch_size=64):
        if len(self.embeddings) == 0:
            return None
        x = torch.stack(self.embeddings)
        y = torch.tensor(self.labels)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_refined_embeddings(model, dataloader, device, output_path):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            refined = model(x)
            all_embeddings.append(refined.cpu())
            all_labels.append(y)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    torch.save({'embeddings': all_embeddings, 'labels': all_labels}, output_path)
    print(f"Saved embeddings to {output_path}")


def update_dataloader(existing_embeddings, existing_labels, new_embeddings, new_labels, batch_size=64):
    combined_x = torch.cat([existing_embeddings, new_embeddings], dim=0)
    combined_y = torch.cat([existing_labels, new_labels], dim=0)

    new_dataset = TensorDataset(combined_x, combined_y)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
    return new_loader


class IntraVectorSelfAttention(nn.Module):
    def __init__(self, embedding_dim=256, num_tokens=32, token_dim=8,
                 num_heads=4, dropout=0.1, use_residual=True):
        super().__init__()
        assert embedding_dim == num_tokens * token_dim, "embedding_dim must equal num_tokens * token_dim"
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.use_residual = use_residual

        self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads,
                                               batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        B = x.size(0)

        # Reshape: (B, 256) -> (B, 32, 8)
        x_tokens = x.view(B, self.num_tokens, self.token_dim)
        x_norm = self.layernorm(x_tokens)

        # Self-attention per sample (intra-vector)
        attended, _ = self.attention(x_norm, x_norm, x_norm)
        attended = self.dropout(attended)

        out = attended.reshape(B, -1)
        out = self.output_proj(out)

        if self.use_residual:
            return x + out
        else:
            return out


def gravitational_loss_normalized(x, labels, lambda_repel=1.0, epsilon=1e-6):
    x = F.normalize(x, p=2, dim=1)
    unique_labels = labels.unique()
    
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        class_embeddings = x[mask]
        centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        centroids[lbl.item()] = centroid

    pull_loss = 0.0
    repel_loss = 0.0

    for i in range(x.size(0)):
        xi = x[i].unsqueeze(0)
        yi = labels[i].item()

        centroid_pos = centroids[yi]
        dist_sq = torch.sum((xi - centroid_pos) ** 2)
        pull_loss += 1.0 / (dist_sq + epsilon)

        for yj, centroid_neg in centroids.items():
            if yj == yi:
                continue
            dist = torch.norm(xi - centroid_neg, p=2)
            repel_loss += 1.0 / ((dist + epsilon) ** 2)

    pull_loss /= x.size(0)
    repel_loss /= x.size(0)

    return pull_loss + lambda_repel * repel_loss


import torch
from torch.utils.data import DataLoader, TensorDataset

def train(model, dataloader, optimizer, device='cuda', lambda_repel=1.0):
    model.train()
    total_loss = 0.0

    for batch_x, batch_labels in dataloader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        refined_x = model(batch_x)
        loss = gravitational_loss_normalized(refined_x, batch_labels, lambda_repel)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


import os
from pathlib import Path

def train_with_validation(model, optimizer, train_loader, val_loader,
                          device='cuda', lambda_repel=1.0,
                          save_every=5, output_dir="saved_embeddings",
                          epoch_start=1, epoch_end=50):

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epoch_start, epoch_end + 1):
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_labels in train_loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            refined_x = model(batch_x)
            loss = gravitational_loss_normalized(refined_x, batch_labels, lambda_repel)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 🔍 Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_labels in val_loader:
                batch_x = batch_x.to(device)
                batch_labels = batch_labels.to(device)
                refined_x = model(batch_x)
                val_loss = gravitational_loss_normalized(refined_x, batch_labels, lambda_repel)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 💾 Save refined embeddings periodically
        if epoch % save_every == 0:
            save_refined_embeddings(model, train_loader, device,
                                    output_path=Path(output_dir) / f"epoch{epoch:02d}_train.pt")
            save_refined_embeddings(model, val_loader, device,
                                    output_path=Path(output_dir) / f"epoch{epoch:02d}_val.pt")

# {
#   'embeddings': torch.Tensor of shape (N, 256),
#   'labels': torch.LongTensor of shape (N,)
# }


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    embedding_dim = 256
    model = IntraVectorSelfAttention(
        embedding_dim=embedding_dim,
        num_tokens=32,
        token_dim=8,
        num_heads=4,
        dropout=0.1,
        use_residual=True
    ).to('cuda')



    # Initial training set
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)

    # Initialize model + optimizer
    model = IntraVectorSelfAttention(...).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run training
    train_with_validation(model, optimizer, train_loader, val_loader,
                        output_dir="saved_embeddings", save_every=5)
    
    # Load saved embeddings
    data = torch.load("saved_embeddings/epoch20_val.pt")
    visualize_embeddings(data['embeddings'], data['labels'], method='umap', title='Refined Embeddings (Epoch 20)')

    emb_list, label_list, epoch_titles = load_snapshots("saved_embeddings/")
    create_embedding_animation(emb_list, label_list, epoch_titles,
                            save_path='refinement_marker_animation.mp4')


    # buffer = MemoryBuffer(max_size=1000)

    # # After processing a batch
    # refined_batch = model(new_x.to('cuda')).detach().cpu()
    # buffer.add_samples([r for r in refined_batch], new_y.tolist())

    # # Add buffer loader to training
    # buffer_loader = buffer.get_buffer_loader()
    # if buffer_loader is not None:
    #     combined_loader = DataLoader(
    #         torch.utils.data.ConcatDataset([new_dataset, buffer_loader.dataset]),
    #         batch_size=64,
    #         shuffle=True
    #     )
