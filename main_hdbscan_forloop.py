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
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    plot_styles, store_probs, \
    plot_clustering, concat_data, generate_prototype, plot_histograms, \
    run_pca, estimate_pca_n, plot_clustering_dual, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.CENTROID_FEAT_AOLME_TTS2_NOISE

# min_cluster_size
a_options = [3,5]

# pca_elem
b_options = [None, 16, 107, 170]

# hdb_mode
c_options = ['eom', 'leaf']
noise_type = 'TTS2-200'
dataset_type = 'GT_Irma'

percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

plot_mode = 'store' # 'show' or 'show_store'

dataset_dvectors = d_vectors_pretrained_model([test_feat_dir], percentage_test, 
                                            remove_outliers, use_cuda=True)

X_train = dataset_dvectors[0]
y_train = dataset_dvectors[1]
X_test = dataset_dvectors[2]
y_test = dataset_dvectors[3]
speaker_labels_dict_train = dataset_dvectors[4]

X_test = X_test.cpu().numpy()
X_train = X_train.cpu().numpy()

Mixed_X_data = np.concatenate((X_train, X_test), axis=0)
Mixed_y_labels = np.concatenate((y_train, y_test), axis=0)

# Store the data in a file using pickle
X_data_and_labels = [Mixed_X_data, Mixed_y_labels]
with open("X_data_and_labels.pickle", "wb") as file:
    pickle.dump(X_data_and_labels, file)

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)


### -------------------------------- from pickle file -----------------------
for_idx = 0
for a, b, c in product(a_options, b_options, c_options):
    # You can perform some action or function with a, b, and c here
    print(f"\n\nParameter a: {a}, Parameter b: {b}, Parameter c: {c}")
    min_cluster_size = a
    pca_elem = b
    hdb_mode = c

    run_id = f'{dataset_type}_min{min_cluster_size}_PCA{pca_elem}_{noise_type}_{hdb_mode}'

    tot_start = time.time()
    with open("X_data_and_labels.pickle", "rb") as file:
        X_data_and_labels = pickle.load(file)
    Mixed_X_data, Mixed_y_labels = X_data_and_labels

    if estimate_pca_flag:
        estimate_pca_n(Mixed_X_data)

    hdb_data_input = None
    if pca_elem != None:
        hdb_data_input = run_pca(Mixed_X_data, pca_elem)
    else:
        hdb_data_input = Mixed_X_data 


    ### try cluster_selection_method = 'leaf' | default = 'eom'
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        cluster_selection_method = hdb_mode).fit(hdb_data_input)

    samples_outliers = hdb.outlier_scores_
    samples_prob = hdb.probabilities_
    samples_label = hdb.labels_

    if plot_hist_flag:
        plot_histograms(samples_outliers, bin_mode = 'std_mode', bin_val=100,
                            add_cdf = False,
                            title_text = f'Outliers',
                            run_id = run_id,
                            plot_mode = plot_mode,
                            output_path = output_folder_path)

    if store_probs_flag:
        store_probs(samples_prob, samples_label, output_folder_path, run_id = run_id)

    if check_0_clusters(samples_prob, samples_label, verbose = False):
        print(f'0 clusters: {run_id}')
        continue

    if plot_hist_flag:
        plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
                            add_cdf = False,
                            title_text = f'probabilities ({np.count_nonzero(samples_prob)})',
                            run_id = run_id,
                            plot_mode = plot_mode,
                            output_path = output_folder_path)

    df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
    x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

    plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                            samples_label, samples_prob,
                            run_id, output_folder_path,
                            plot_mode)

    tot_end = time.time()
    print(f"{for_idx} - Clustering elapsed time : {(tot_end - tot_start):.1f}s")
    for_idx = for_idx + 1

