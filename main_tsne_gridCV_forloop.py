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
                        plot_clustering_subfig


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.CHUNKS_FEAT_STG2_IRMA

# t-sne perplexity 
c_options = [5, 10, 15, 20, 30, 40, 50]

# t-sne n_iter
d_options = [500, 800, 1200, 5000]

noise_type = 'none'
dataset_type = 'tsneGrid_pcaNone_chunks'

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


X_data_and_labels = [Mixed_X_data, Mixed_y_labels]
with open(f'tsneGrid_{dataset_type}_noise{noise_type}.pickle', "wb") as file:
    pickle.dump(X_data_and_labels, file)

run_id = f'{dataset_type}_{noise_type}'

### -------------------------------- from pickle file -----------------------
for_idx = 0
tsne_labels_list = []
tsne_param_list = []
for c, d in product(c_options, d_options):
    # You can perform some action or function with a, b, and c here
    print(f"\n\nPerplexity: {c} \tN_iter:{d}")
    perplexity_val = c
    n_iter = d
    tsne_param_list.append({'per': c, 'n':d})

    tot_start = time.time()
    with open(f'tsneGrid_{dataset_type}_noise{noise_type}.pickle', "rb") as file:
        X_data_and_labels = pickle.load(file)
    Mixed_X_data, Mixed_y_labels = X_data_and_labels

    current_df = gen_tsne(Mixed_X_data, Mixed_y_labels,
                          perplexity_val = perplexity_val,
                          n_iter = n_iter,
                          n_comp = 0)
    tsne_labels_list.append(current_df)


### -------------------------------------------- Plot 8 figures ----------------
# Save output of tsne for future ref:

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{run_id}')
output_folder_path.mkdir(parents=True, exist_ok=True)


plot_clustering_subfig(tsne_labels_list, Mixed_y_labels,
                        1, 2,
                        tsne_param_list,
                        run_id, output_folder_path,
                        plot_mode)

tot_end = time.time()
print(f"{for_idx} - Clustering elapsed time : {(tot_end - tot_start):.1f}s")
for_idx = for_idx + 1
