# Load the pkl feats and plot a HDBSCAN clustering with U-MAP, based on the settings from this project
import warnings
import numpy as np
import hdbscan
from pathlib import Path
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
import umap

from generator.SR_Dataset import *
from utils_fbank_reduction import apply_spec_compression

from utils_luis import gen_tsne, plot_histograms, plot_clustering_dual 

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

def optional_PCA(data_scaled, n_components=50):
        
        pca_model = PCA(n_components=n_components)
        pca_model.fit(data_scaled)

        print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_[:5]}")
        print(f"Total variance explained: {pca_model.explained_variance_ratio_.sum():.3f}")

        compressed = pca_model.transform(data_scaled)

        print(f"PCA: {compressed.shape} (compression: {data_scaled.size/len(compressed):.1f}x)")
        return compressed


def normalize_frames(m, Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

min_cluster_size = 25 
min_samples = 5 
hdb_mode = 'eom'

# Method 0: Statistical features
# Method 1: Temporal pooling
# Method 2: Delta features
# Method 3: MFCC-inspired
# Method 4: Frequency bands pooling

plot_mode = 'store' # 'show' or 'show_store'


root_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Dvectors')                                          
wavs_folder = root_path / 'wavs_test_pairs' / 'aolme_fix_input_wavs'
feats_folder = root_path / 'wavs_test_pairs' / 'aolme_fix_feats'

output_folder_path = root_path / 'wavs_test_pairs' / 'aolme_fix_feats' / 'fbank_compression'

# Create output folder if it doesn't exist
output_folder_path.mkdir(parents=True, exist_ok=True)


list_of_feats = sorted(list(feats_folder.glob('*.pkl')))
list_of_wavs = sorted(list(wavs_folder.glob('*.wav')))


# Iterate through the methods
for compression_method in [0, 1, 2, 3, 4]:

    current_run_id = f'fbank_mn{min_cluster_size}_ms{min_samples}_{hdb_mode}_c{compression_method}'
    feats_array = []
    labels_list = []

    for path_idx, current_feat_path in enumerate(list_of_feats):

        # enroll_embedding, _ = get_d_vector_aolme(current_feat_path, model, norm_flag=norm_flag)
        with open(current_feat_path, 'rb') as f:
            feat_and_label = pickle.load(f)
            
        input = feat_and_label['feat'] # size : (n_frames, dim=40)
        label = feat_and_label['label']

        input = normalize_frames(input, Scale=False)
        current_comp_feat = apply_spec_compression(input, method_sel=compression_method)

        feats_array.append(current_comp_feat)
        labels_list.append(label)

    # Minimum number of frames from all the features
    min_frames = min([feat.shape[0] for feat in feats_array])

    # Trim all features to the minimum number of frames
    feats_array = [feat[:min_frames] for feat in feats_array]

    # Stack all features into a single numpy array
    Mixed_X_data = np.stack(feats_array, axis=0)

    n_components = 15
    data_standardized = StandardScaler().fit_transform(Mixed_X_data)

    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=5,  # Adjust based on dataset size
        min_dist=0.1,    # Controls compactness of clusters
        n_components=n_components,  # Reduced dimensionality
        metric='cosine',  # Good default for many feature types
    )
    hdb_data_input = umap_reducer.fit_transform(data_standardized)

    ### try cluster_selection_method = 'leaf' | default = 'eom'
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                            min_samples=min_samples,\
                        cluster_selection_method = hdb_mode).fit(hdb_data_input)

    samples_outliers = hdb.outlier_scores_
    samples_prob = hdb.probabilities_
    samples_label = hdb.labels_

    # plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
    #                     add_cdf = False,
    #                     title_text = f'probabilities ({np.count_nonzero(samples_prob)})',
    #                     run_id = current_run_id,
    #                     plot_mode = plot_mode,
    #                     output_path = output_folder_path)


    # alternative way to get labels from labels_list
    speaker_labels_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(labels_list)))])
    if 'noises' in speaker_labels_dict.keys():
        speaker_labels_dict['noises'] = 6

    y_lbls = [speaker_labels_dict[x] for x in labels_list]
    Mixed_y_labels = np.array(y_lbls)

    # Decide whether to apply PCA or not
    if Mixed_X_data.shape[1] < 108:
        tsne_pca_comp = 0

    df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels, n_comp=tsne_pca_comp)
    x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))   

    plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                            samples_label, samples_prob,
                            current_run_id, output_folder_path,
                            plot_mode)


    info_pred = [samples_label, x_tsne_2d, list_of_wavs]
    # Store Mixed_X_paths and Mixed_y_labels on a similar way
    with open(f'{output_folder_path}/{current_run_id}_predinfo.pickle', 'wb') as handle:
        pickle.dump(info_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Apply PCA as optional
if Mixed_X_data.shape[1] > 108:
    mixed_x_data_pca = optional_PCA(Mixed_X_data)
else:
    print(f"Skipping PCA as feature dimension is low. {Mixed_X_data.shape[1]} < 108")

# Apply UMAP
umap_reducer = umap.UMAP(
    n_neighbors=5,  # Adjust based on dataset size
    min_dist=0.1,    # Controls compactness of clusters
    n_components=n_components,  # Reduced dimensionality
    metric='cosine',  # Good default for many feature types
)
hdb_data_input = umap_reducer.fit_transform(data_standardized)

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        min_samples=min_samples,\
                    cluster_selection_method = hdb_mode).fit(hdb_data_input)

samples_outliers = hdb.outlier_scores_
samples_prob = hdb.probabilities_
samples_label = hdb.labels_

# plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
#                     add_cdf = False,
#                     title_text = f'probabilities ({np.count_nonzero(samples_prob)})',
#                     run_id = current_run_id,
#                     plot_mode = plot_mode,
#                     output_path = output_folder_path)


# alternative way to get labels from labels_list
speaker_labels_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(labels_list)))])
if 'noises' in speaker_labels_dict.keys():
    speaker_labels_dict['noises'] = 6

y_lbls = [speaker_labels_dict[x] for x in labels_list]
Mixed_y_labels = np.array(y_lbls)

# Decide whether to apply PCA or not
if Mixed_X_data.shape[1] < 108:
    tsne_pca_comp = 0

df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels, n_comp=tsne_pca_comp)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))   

plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)


info_pred = [samples_label, x_tsne_2d, list_of_wavs]
# Store Mixed_X_paths and Mixed_y_labels on a similar way
with open(f'{output_folder_path}/{current_run_id}_predinfo.pickle', 'wb') as handle:
    pickle.dump(info_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)