from __future__ import print_function
import os
import warnings
import constants
import numpy as np
import hdbscan
from pathlib import Path
import sys
import warnings
import argparse
import re
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import umap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, \
    store_probs, organize_samples_by_label,\
    plot_histograms, calculate_X_centroids, \
    run_pca, plot_clustering_dual, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

min_cluster_size = 25 
pca_elem = 0
hdb_mode = 'eom'
min_samples = 5

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

feats_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Liz/STG2_EXP001-SHAS-DV/TestAO-Liz_SHAS_DV_feats.pkl')
output_folder_path_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest')
run_params_ex = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}"
Exp_name_ex = 'TestAO-Liz_SHAS_DV'

pred_lbl_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest/TestAO-Liz_SHAS_DV_predlbl.pickle')
pred_reduced_feats_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest/TestAO-Liz_SHAS_DV_reduced_feats.pickle')


parser = argparse.ArgumentParser()

parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')

parser.add_argument('--pred_lbl_picke', default=pred_lbl_pickle_ex, help='Output path for clustering labels')
parser.add_argument('--pred_reduced_feats', default=pred_reduced_feats_ex, help='Output path for reduced features')

args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

pred_lbl_pickle = Path(args.pred_lbl_picke)
pred_reduced_feats = Path(args.pred_reduced_feats)

run_params = args.run_params
Exp_name = args.exp_name

print(f'run_params: {run_params}')

# #RUN_PARAMS="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"
# #example "pca0_mcs10_ms5_eom"

pattern = r"pca(\d+)_mcs(\d+)_ms(\d+)_(\w+)"
match = re.match(pattern, run_params)

if match:
    pca_elem = int(match.group(1))
    min_cluster_size = int(match.group(2))
    min_samples = int(match.group(3))
    hdb_mode = match.group(4)
else:
    sys.exit("Invalid run_name format")

# Print the extracted values
print(f"pca_elem: {pca_elem}")
print(f"min_cluster_size: {min_cluster_size}")
print(f"min_samples: {min_samples}")
print(f"hdb_mode: {hdb_mode}")


plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False
plot_mode = 'store' # 'show' or 'show_store'

with open(f'{feats_pickle_path}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels

current_run_id = f'{Exp_name}_{run_params}'

hdb_data_input = None
# if pca_elem == None or pca_elem == 0:
#     hdb_data_input = Mixed_X_data
# else:
#     hdb_data_input = run_pca(Mixed_X_data, pca_elem) 

# perplexity_val = 15 
# n_iter = 900


# ## Add t-sne as preprocessing
# data_standardized = StandardScaler().fit_transform(Mixed_X_data)

# # Numbers to try: 16, 75, 108
# pca_selected = PCA(n_components=170)
# x_low_dim = pca_selected.fit_transform(data_standardized)

# tsne = TSNE(n_components=4, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
# hdb_data_input = tsne.fit_transform(x_low_dim)

# # Print the shape of the t-sne results
# print(f"tsne_results.shape: {hdb_data_input.shape}")

# # Plot the t-sne 3D results
# tsne_results = hdb_data_input
# output_path = output_folder_path
# run_id = current_run_id

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels, cmap='viridis', marker='o')

# legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend1)

# ax.set_title(f't-SNE 3D - {run_id}')
# ax.set_xlabel('t-SNE 1')
# ax.set_ylabel('t-SNE 2')
# ax.set_zlabel('t-SNE 3')

# plt.savefig(output_path / f'{run_id}_tsne_3d.png')
# plt.show()
# plt.close()


n_components = 15
data_standardized = StandardScaler().fit_transform(Mixed_X_data)

# Apply UMAP
umap_reducer = umap.UMAP(
    n_neighbors=5,  # Adjust based on dataset size
    min_dist=0.1,    # Controls compactness of clusters
    n_components=n_components,  # Reduced dimensionality
    metric='cosine',  # Good default for many feature types
)
hdb_data_input = umap_reducer.fit_transform(data_standardized)


with open(str(pred_reduced_feats), 'wb') as handle:
    pickle.dump(hdb_data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        min_samples=min_samples,\
                    cluster_selection_method = hdb_mode).fit(hdb_data_input)

samples_outliers = hdb.outlier_scores_
samples_prob = hdb.probabilities_
samples_label = hdb.labels_

with open(pred_lbl_pickle, 'wb') as handle:
    pickle.dump(samples_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

if plot_hist_flag:
    plot_histograms(samples_outliers, bin_mode = 'std_mode', bin_val=100,
                        add_cdf = False,
                        title_text = f'Outliers',
                        run_id = current_run_id,
                        plot_mode = plot_mode,
                        output_path = output_folder_path)

if store_probs_flag:
    store_probs(samples_prob, samples_label, output_folder_path, run_id = current_run_id)

if check_0_clusters(samples_prob, samples_label, verbose = False):
    print(f'0 clusters: {current_run_id}')

if plot_hist_flag:
    plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
                        add_cdf = False,
                        title_text = f'probabilities ({np.count_nonzero(samples_prob)})',
                        run_id = current_run_id,
                        plot_mode = plot_mode,
                        output_path = output_folder_path)

df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

# Store x_tsne_2d for later use
with open(f'{output_folder_path}/{current_run_id}_xtsne2d.pickle', 'wb') as handle:
    pickle.dump(x_tsne_2d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store Mixed_X_paths and Mixed_y_labels on a similar way
with open(f'{output_folder_path}/{current_run_id}_Xpaths.pickle', 'wb') as handle:
    pickle.dump(Mixed_X_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)


plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)


# Divide the X_train data in a dictionary with the labels as keys
centroids_dict = calculate_X_centroids(Mixed_X_data, samples_label)

# Store the centroid_dict
with open(f'{output_folder_path}/{current_run_id}_centroids-mean.pickle', 'wb') as handle:
    pickle.dump(centroids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_distance_histograms(centroids, X_train, y_labels, output_path, run_id):
    """
    Plots and saves histograms of Euclidean distances from each element in X_train to its label centroid.
    
    Parameters:
    centroids (dict): Dictionary where keys are labels and values are centroids (numpy arrays).
    X_train (list of lists): Feature vectors.
    y_labels (list): Corresponding labels for each feature vector.
    output_path (str): Directory where the plots will be saved.
    run_id (str): Identifier to include in the plot filenames.
    """
    if len(X_train) != len(y_labels):
        raise ValueError("X_train and y_labels must have the same length.")
    
    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)
    
    # Compute distances
    distances = []
    unique_labels = sorted(centroids.keys())
    
    for label in unique_labels:
        label_distances = [
            np.linalg.norm(np.array(X_train[i]) - centroids[label]) 
            for i in range(len(y_labels)) if y_labels[i] == label
        ]
        distances.append((label, label_distances))
    
    # Plot histograms and save figures
    num_labels = len(unique_labels)
    num_subplots = 4
    num_figures = (num_labels + num_subplots - 1) // num_subplots  # Calculate the number of figures
    
    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for subplot_idx in range(num_subplots):
            global_idx = fig_idx * num_subplots + subplot_idx
            if global_idx < num_labels:
                label, label_distances = distances[global_idx]
                axes[subplot_idx].hist(label_distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
                axes[subplot_idx].set_title(f'Label {label}')
                axes[subplot_idx].set_xlabel('Distance')
                axes[subplot_idx].set_ylabel('Frequency')
            else:
                axes[subplot_idx].axis('off')  # Turn off unused subplots
        
        plt.tight_layout()
        # Save the figure
        figure_path = os.path.join(output_path, f"hist_{run_id}_{fig_idx + 1}.png")
        plt.savefig(figure_path)
        plt.close(fig)
    
    return distances


# lbl_distances = plot_distance_histograms(centroids_dict, Mixed_X_data, samples_label, output_folder_path, current_run_id)


# # Store the centroid_dict
# with open(f'{output_folder_path}/{current_run_id}_CM-dist.pickle', 'wb') as handle:
#     pickle.dump(lbl_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)


organize_samples_by_label(Mixed_X_paths, samples_label, samples_prob, output_folder_path)