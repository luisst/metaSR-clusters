from __future__ import print_function
import os
import sys
import time
import warnings
import time
import constants
import numpy as np
import hdbscan
import re
import pickle
from pathlib import Path
from itertools import product

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    plot_styles, store_probs, \
    plot_clustering, concat_data, generate_prototype, plot_histograms, \
    run_pca, estimate_pca_n, plot_clustering_dual, check_0_clusters

def feature_selection_methods(method_n, X_data, y_labels):
    y_labels = y_labels - 1
    feature_selection, classification = method_n
    clf = Pipeline([
        ('feature_selection', feature_selection),
        ('classification', classification)
    ])
    clf.fit(X_data, y_labels)
    X_new = clf.named_steps['feature_selection'].transform(X_data)
    new_shape = X_new.shape[1]

    regex1 = r"(?<=estimator=)([a-zA-Z0-9]+)"
    matches = re.finditer(regex1, f'{feature_selection}')
    feat_sel_info_ft = None

    match_count = 0
    for match in matches:
        feat_sel_info_ft = match.group(1)
        match_count += 1

    if match_count != 1:
        print("Error: Expected exactly one match from re.finditer.")
    


    regex2 = r"([a-zA-Z0-9]+)\("
    matches = re.finditer(regex2, f'{classification}')
    feat_sel_info_cl = None

    match_count = 0
    for match in matches:
        feat_sel_info_cl = match.group(1)
        match_count += 1

    if match_count != 1:
        print("Error: Expected exactly one match from re.finditer.")

    feat_sel_info = f'{feat_sel_info_ft}_{feat_sel_info_cl}'

    print(f'New shape: {new_shape} with method: {feat_sel_info}')
    return X_new, feat_sel_info

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.CENTROID_FEAT_AOLME_NOISE

# min_cluster_size
a_options = [3,5,7]

# pca_elem
b_options = [None, 16, 50]

# hdb_mode
c_options = ['eom', 'leaf']

# Methods for feature selection
d_options = [
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), RandomForestClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), SVC()),
    (SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=5000)), RandomForestClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), DecisionTreeClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), KNeighborsClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), GradientBoostingClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), AdaBoostClassifier()),
    (SelectFromModel(LinearSVC(dual=False, penalty="l2")), XGBClassifier()),
]


noise_type = 'a'
dataset_type = 'NORMv2_AOLME_groups'

percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

plot_mode = 'store' # 'show' or 'show_store'

dataset_dvectors = d_vectors_pretrained_model([test_feat_dir], percentage_test,
                                            remove_outliers,
                                            return_paths_flag = True,
                                            norm_flag = True,
                                            samples_flag=False,
                                            use_cuda=True)

X_train = dataset_dvectors[0]
y_train = dataset_dvectors[1]
X_train_paths = dataset_dvectors[2]
X_test = dataset_dvectors[3]
y_test = dataset_dvectors[4]
X_test_paths = dataset_dvectors[5]
speaker_labels_dict_train = dataset_dvectors[6]

X_test = X_test.cpu().numpy()
X_train = X_train.cpu().numpy()

# Mixed_X_data = np.concatenate((X_train, X_test), axis=0)
Mixed_X_data = X_train

# Mixed_y_labels = np.concatenate((y_train, y_test), axis=0)
Mixed_y_labels = y_train

# Store the data in a file using pickle
# X_data_and_labels = [Mixed_X_data, Mixed_y_labels]

X_data_and_labels = [X_train, X_train_paths, y_train]
with open(f'dVectors_{dataset_type}_noise{noise_type}.pickle', "wb") as file:
    pickle.dump(X_data_and_labels, file)

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)

total_start = time.time()
### -------------------------------- from pickle file -----------------------
for_idx = 0
# for a, b, c, d in product(a_options, b_options, c_options, d_options):
for a, b, c in product(a_options, b_options, c_options):
    # You can perform some action or function with a, b, and c here
    print(f"\n\nParameter a: {a}, Parameter b: {b}, Parameter c: {c}")
    min_cluster_size = a
    pca_elem = b
    hdb_mode = c


    current_start = time.time()
    with open(f'dVectors_{dataset_type}_noise{noise_type}.pickle', "rb") as file:
        X_data_and_labels = pickle.load(file)
    Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels

    # Mixed_x_data, feat_sel_info = feature_selection_methods(d, Mixed_X_data, Mixed_y_labels)

    # run_id = f'{dataset_type}_min{min_cluster_size}_PCA{pca_elem}_{noise_type}_{hdb_mode}_{feat_sel_info}'
    run_id = f'{dataset_type}_min{min_cluster_size}_PCA{pca_elem}_{noise_type}_{hdb_mode}_noFeat'

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
        print(f'!!!! 0 clusters: {run_id}')
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

    current_end = time.time()
    print(f"{for_idx} - Clustering elapsed time : {(current_end - current_start):.1f}s")
    for_idx = for_idx + 1

total_end = time.time()
print(f"Total time : {(total_end - total_start):.1f}s")
