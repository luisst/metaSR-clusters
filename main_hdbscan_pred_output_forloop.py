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
import time

from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import umap

warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, \
    store_probs, organize_samples_by_label,\
    plot_histograms, \
    run_pca, plot_clustering_dual, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

pca_elem = 0

min_cluster_size = 25 
min_samples = 5
hdb_mode = 'eom'

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

feats_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Irmadb/STG_2/STG2_EXP010C-SHAS-DV/TestAO-Irmadb_SHAS_DV_feats.pkl')

output_folder_path_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Irmadb/STG_3/STG3_EXP010C-SHAS-DV-t3sneH7/HDBSCAN_pred_output')
run_params_ex = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}"
Exp_name_ex = 'STG3_EXP010C-SHAS-DV-t3sneH7'

parser = argparse.ArgumentParser()

parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')

args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

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


plot_mode = 'store' # 'show' or 'show_store'

with open(f'{feats_pickle_path}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels


# U-map num_components
a_options = [5, 10, 15, 20, 25]

# U-map num_neighbors
b_options = [2, 5, 10, 20, 50]

# U-map min_dist
c_options = [0.1, 0.25, 0.5, 0.8, 0.99]

# U-map metric
d_options = ['euclidean', 'manhattan','cosine']


### -------------------------------- from pickle file -----------------------
for_idx = 0
for a, b, c, d in product(a_options, b_options, c_options, d_options):
    # You can perform some action or function with a, b, and c here
    print(f"\n\n N_comp a: {a}, N_neighb b: {b}, min_dist c: {c}, Metric d: {d}")

    umap_N_comp = a
    umap_N_neighs = b
    umap_min_dist = c
    umap_metric = d

    run_params = f"Ncomp{umap_N_comp}_Nneigh{umap_N_neighs}_md{umap_min_dist}_{umap_metric}" 

    # Last substring '-' from exp_name
    Exp_name_last = Exp_name.split('-')[-1]


    current_run_id = f'{Exp_name_last}_{run_params}'

    tot_start = time.time()

    hdb_data_input = None

    data_standardized = StandardScaler().fit_transform(Mixed_X_data)

    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=umap_N_neighs,  # Adjust based on dataset size
        min_dist=umap_min_dist,    # Controls compactness of clusters
        n_components=umap_N_comp,  # Reduced dimensionality
        metric=umap_metric,  # Good default for many feature types
        random_state=42  # For reproducibility
    )
    hdb_data_input = umap_reducer.fit_transform(data_standardized)

    ### try cluster_selection_method = 'leaf' | default = 'eom'
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                            min_samples=min_samples,\
                        cluster_selection_method = hdb_mode).fit(hdb_data_input)

    samples_outliers = hdb.outlier_scores_
    samples_prob = hdb.probabilities_
    samples_label = hdb.labels_

    df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
    x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

    plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                            samples_label, samples_prob,
                            current_run_id, output_folder_path,
                            plot_mode)

    tot_end = time.time()
    print(f"{for_idx} - Clustering elapsed time : {(tot_end - tot_start):.1f}s")
    for_idx = for_idx + 1
