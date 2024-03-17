from __future__ import print_function
import os
import time
import constants
import numpy as np
import hdbscan
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from utils_luis import gen_tsne, d_vectors_pretrained_model, \
     plot_clustering_dual


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.TTS2_FEAT_SAMPLES

noise_type = 'TTS2'
dataset_type = 'tsnePRE_samples_TTS2'

percentage_test = 0.0
remove_outliers = 'None'

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

df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

# Store the data in a file using pickle
X2d_data_and_labels = [x_tsne_2d, Mixed_y_labels]
with open("X2d_data_and_labels.pickle", "wb") as file:
    pickle.dump(X2d_data_and_labels, file)


output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)


min_cluster_size = 5 
hdb_mode = 'eom'

run_id = f'{dataset_type}_min{min_cluster_size}_{noise_type}_{hdb_mode}'

tot_start = time.time()
with open("X2d_data_and_labels.pickle", "rb") as file:
    X_data_and_labels = pickle.load(file)
x_tsne_2d, Mixed_y_labels = X_data_and_labels

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                    cluster_selection_method = hdb_mode).fit(x_tsne_2d)

samples_outliers = hdb.outlier_scores_
samples_prob = hdb.probabilities_
samples_label = hdb.labels_

plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        run_id, output_folder_path,
                        plot_mode)

tot_end = time.time()