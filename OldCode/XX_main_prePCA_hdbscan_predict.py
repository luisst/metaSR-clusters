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
from utils_luis import gen_tsne_X, d_vectors_pretrained_model, \
    modify_predict_labels, \
    run_pca_inference, plot_clustering_dual_predict, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

train_feat_dir = constants.TTS2_FEAT_GROUPS_SPEECH
test_feat_dir = constants.TTS2_FEAT_GTTESTSIM

noise_type = 'TTS2'
dataset_type = 'pcaPRE_samples_TTS2'

percentage_test = 0.0
remove_outliers = 'None'

plot_mode = 'store' # 'show' or 'show_store'

dataset_dvectors_train = d_vectors_pretrained_model([train_feat_dir], 0.0, 
                                            'None', use_cuda=True)

X_train = dataset_dvectors_train[0]
y_train = dataset_dvectors_train[1]
speaker_labels_dict_train_train = dataset_dvectors_train[4]

dataset_dvectors_test = d_vectors_pretrained_model([test_feat_dir], 0.0, 
                                            'None', use_cuda=True)

X_test = dataset_dvectors_test[0]
y_test = dataset_dvectors_test[1]
speaker_labels_dict_train_test = dataset_dvectors_test[4]

X_test = X_test.cpu().numpy()
X_train = X_train.cpu().numpy()

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)

min_cluster_size = 5
pca_elem = 16
hdb_mode = 'leaf' 

run_id = f'{dataset_type}_min{min_cluster_size}_PCA{pca_elem}_{noise_type}_{hdb_mode}'

tot_start = time.time()

train_data_pca, test_data_pca = run_pca_inference(X_train, X_test, pca_elem)

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                    cluster_selection_method = hdb_mode,
                    prediction_data=True).fit(train_data_pca)

train_samples_outliers = hdb.outlier_scores_
train_prediction_prob = hdb.probabilities_
train_prediction_label = hdb.labels_


if check_0_clusters(train_prediction_prob, train_prediction_label, verbose = False):
    print(f'\n\n\n>>>>>>>>>>>>>>>>> 0 clusters: {run_id}')
tot_end = time.time()

###### -------------------------- Predict -----------------------------

## Create fake y_train labels
Mixed_X_data = np.concatenate((X_train, X_test), axis=0)
y_test = y_test + 100
Mixed_y_GT = np.concatenate((y_train, y_test), axis=0)

GT_labels, GT_strengths = hdbscan.approximate_predict(hdb, test_data_pca)

GT_labels = modify_predict_labels(GT_labels)

Mixed_y_prediction = np.concatenate((train_prediction_label, GT_labels), axis=0)

x_tsne_2d = gen_tsne_X(Mixed_X_data)

plot_clustering_dual_predict(x_tsne_2d, 
                             Mixed_y_GT,
                             Mixed_y_prediction,
                            run_id, output_folder_path,
                            plot_mode,
                            plot_minus1 = False)