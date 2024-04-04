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

warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    store_probs, organize_samples_by_label, \
    plot_histograms, \
    run_pca, plot_clustering_dual, check_0_clusters


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

min_cluster_size = 0
pca_elem = 0
hdb_mode = None
min_samples = 0

# feats_folder = Path('/home/luis/Dropbox/DATASETS_AUDIO/VAD_aolme/EXP-001-Liz/TestSet_AOLME_SHAS/Testset_stage3/feats_files')
# wavs_folder = Path('/home/luis/Dropbox/DATASETS_AUDIO/VAD_aolme/EXP-001-Liz/TestSet_AOLME_SHAS/Testset_stage2/wav_chunks')
# output_folder_path = Path('/home/luis/Dropbox/DATASETS_AUDIO/VAD_aolme/EXP-001-Liz/TestSet_AOLME_SHAS/Testset_stage3/HDBSCAN_pred_output')

parser = argparse.ArgumentParser()

parser.add_argument('input_feats_folder', help='Path to the folder to load the extracted features')
parser.add_argument('wavs_folder', help='Path to the folder to input wavs paths')
parser.add_argument('output_pred_folder', help='Path to the folder to store the predictions')
parser.add_argument('run_name', help='')
parser.add_argument('exp_name', help='')

args = parser.parse_args()

feats_folder = Path(args.input_feats_folder)
wavs_folder = Path(args.wavs_folder)
output_folder_path = Path(args.output_pred_folder)

run_name = args.run_name
Exp_name = args.exp_name

print(f'run_name: {run_name}')

# #RUN_NAME="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"
# #example "pca0_mcs10_ms5_eom"

pattern = r"pca(\d+)_mcs(\d+)_ms(\d+)_(\w+)"
match = re.match(pattern, run_name)

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


percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

plot_mode = 'store' # 'show' or 'show_store'

dataset_dvectors = d_vectors_pretrained_model(feats_folder, percentage_test,
                                            remove_outliers,
                                            wavs_folder,
                                            return_paths_flag = True,
                                            norm_flag = True,
                                            use_cuda=True,
                                            samples_flag=True)

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


run_id = f'{Exp_name}_minCL{min_cluster_size}_minSM{min_samples}_{hdb_mode}'

hdb_data_input = None
if pca_elem == None or pca_elem == 0:
    hdb_data_input = Mixed_X_data
else:
    hdb_data_input = run_pca(Mixed_X_data, pca_elem) 


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