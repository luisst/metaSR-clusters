from __future__ import print_function
import os
import sys
import time
import warnings
import time
import constants
import numpy as np
import hdbscan
import pickle
import re
import argparse
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, \
    store_probs, \
    plot_histograms, \
    check_0_clusters, run_pca, estimate_pca_n, plot_clustering_dual


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

# min_cluster_size
a_options = [12, 18, 25, 30, 35]

# pca_elem
b_options = [None, 16, 170, 250]

# hdb_mode
c_options = ['eom', 'leaf']

# min_samples
d_options = [5, 7, 9]

percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

plot_mode = 'store' # 'show' or 'show_store'

feats_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Irma/STG_2/STG2_EXP003-SHASfilt-DV/TestAO-Irma_SHASfilt_DV_feats.pkl')
output_folder_path_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Irma/STG_3/EXP003_HDB-SCAN_forloop/')
Exp_name_ex = 'EXP003_TestAO-Irma_TDA'

parser = argparse.ArgumentParser()

parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')

args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

Exp_name = args.exp_name

plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False
plot_mode = 'store' # 'show' or 'show_store'

with open(f'{feats_pickle_path}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels



### -------------------------------- from pickle file -----------------------
for_idx = 0
for a, b, c, d in product(a_options, b_options, c_options, d_options):
    # You can perform some action or function with a, b, and c here
    print(f"\n\nParameter a: {a}, Parameter b: {b}, Parameter c: {c}, Parameter d: {d}")
    min_cluster_size = a
    pca_elem = b
    hdb_mode = c
    min_samples = d

    run_params = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}" 
    current_run_id = f'{Exp_name}_{run_params}'

    tot_start = time.time()

    if estimate_pca_flag:
        estimate_pca_n(Mixed_X_data)

    hdb_data_input = None
    if pca_elem != None:
        hdb_data_input = run_pca(Mixed_X_data, pca_elem)
    else:
        hdb_data_input = Mixed_X_data 


    ### try cluster_selection_method = 'leaf' | default = 'eom'
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                          min_samples=min_samples,\
                        cluster_selection_method = hdb_mode).fit(hdb_data_input)

    samples_outliers = hdb.outlier_scores_
    samples_prob = hdb.probabilities_
    samples_label = hdb.labels_

    if plot_hist_flag:
        plot_histograms(samples_outliers, bin_mode = 'std_mode', bin_val=100,
                            add_cdf = False,
                            title_text = f'Outliers',
                            run_id = current_run_id,
                            plot_mode = plot_mode,
                            output_path = output_folder_path)

    if store_probs_flag:
        store_probs(samples_prob, samples_label, output_folder_path, run_id = run_id)

    if check_0_clusters(samples_prob, samples_label, verbose = False):
        print(f'0 clusters: {current_run_id}')
        continue

    if plot_hist_flag:
        plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
                            add_cdf = False,
                            title_text = f'probabilities ({np.count_nonzero(samples_prob)})',
                            run_id = current_run_id,
                            plot_mode = plot_mode,
                            output_path = output_folder_path)

    df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
    x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

    plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                            samples_label, samples_prob,
                            current_run_id, output_folder_path,
                            plot_mode)

    tot_end = time.time()
    print(f"{for_idx} - Clustering elapsed time : {(tot_end - tot_start):.1f}s")
    for_idx = for_idx + 1

