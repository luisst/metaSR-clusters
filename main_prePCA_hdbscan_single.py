from __future__ import print_function
import os
import warnings
import constants
import numpy as np
import hdbscan
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    store_probs, organize_samples_by_label, \
    plot_histograms, \
    run_pca, plot_clustering_dual, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.CHUNKS_FEAT_STG2_AZURE_IRMA

noise_type = 'none'
dataset_type = 'run_chunks_azure_irma_stg2'

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

Mixed_X_data = X_train
Mixed_y_labels = y_train

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)


min_cluster_size = 25
pca_elem = None
hdb_mode = 'eom' 
min_samples = 5

run_id = f'{dataset_type}_minCL{min_cluster_size}_minSM{min_samples}_{noise_type}_{hdb_mode}'

hdb_data_input = None
if pca_elem != None or pca_elem == 0:
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
                        run_id = run_id,
                        plot_mode = plot_mode,
                        output_path = output_folder_path)

if store_probs_flag:
    store_probs(samples_prob, samples_label, output_folder_path, run_id = run_id)

if check_0_clusters(samples_prob, samples_label, verbose = False):
    print(f'0 clusters: {run_id}')

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


organize_samples_by_label(X_train_paths, samples_label, samples_prob, output_folder_path)