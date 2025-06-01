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

warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import gen_tsne, \
    store_probs, organize_samples_by_label, \
    plot_histograms, \
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

plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)


organize_samples_by_label(Mixed_X_paths, samples_label, samples_prob, output_folder_path)