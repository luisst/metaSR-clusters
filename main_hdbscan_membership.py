from __future__ import print_function
import os
import sys
import time
import warnings
import time
import constants
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import pickle

from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    plot_styles, store_probs, \
    plot_clustering, concat_data, generate_prototype, plot_histograms, \
    run_pca, estimate_pca_n


warnings.filterwarnings("ignore", message="numpy.dtype size changed")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

percentage_test = 0.0
remove_outliers = 'None'
load_prev = True
pca_elem = 170
min_cluster_size = 7
run_id = f'GT_irma_min{min_cluster_size}_PCA{pca_elem}_Noise_leaf'

if not load_prev:
    test_feat_dir = [constants.CENTROID_FEAT_AOLME_NOISE]

    tot_start = time.time()

    dataset_dvectors = d_vectors_pretrained_model(test_feat_dir, percentage_test, 
                                                remove_outliers, use_cuda=True)

    X_train = dataset_dvectors[0]
    y_train = dataset_dvectors[1]
    X_test = dataset_dvectors[2]
    y_test = dataset_dvectors[3]
    speaker_labels_dict_train = dataset_dvectors[4]

    # prototype_tensor, prototypes_labels = generate_prototype(X_train, y_train, verbose=False)
    # prototypes_labels = prototypes_labels.cpu().numpy().astype(int)

    X_test = X_test.cpu().numpy()
    X_train = X_train.cpu().numpy()

    Mixed_X_data = np.concatenate((X_train, X_test), axis=0)
    Mixed_y_labels = np.concatenate((y_train, y_test), axis=0)


    ## ----------------------------------- Method --------------------------
    # Store the data in a file using pickle
    X_data_and_labels = [Mixed_X_data, Mixed_y_labels]
    with open("X_data_and_labels.pickle", "wb") as file:
        pickle.dump(X_data_and_labels, file)

    tot_end = time.time()
    print("Computing d-vectors elapsed time : %0.1fs" % (tot_end - tot_start))


tot_start = time.time()
with open("X_data_and_labels.pickle", "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_y_labels = X_data_and_labels

# estimate_pca_n(Mixed_X_data)
data_pca = run_pca(Mixed_X_data, pca_elem)

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                       cluster_selection_method = 'leaf').fit(data_pca)

samples_outliers = hdb.outlier_scores_
plot_histograms(samples_outliers, bin_mode = 'std_mode', bin_val=100,
                    add_cdf = False,
                    title_text = f'Outliers')

samples_prob = hdb.probabilities_
samples_label = hdb.labels_
non_zero_count =np.count_nonzero(samples_prob)

store_probs(samples_prob, samples_label, run_id = run_id)

# print number of clusters:
unique_labels = set(samples_label)
print(f'Number of clusters: {len(unique_labels) - 1}')

print(f'hdb probs \t min: {min(samples_prob)} \t max: {max(samples_prob)} \t non_zero: {non_zero_count}')

if non_zero_count == 0:
    sys.exit(f'All the probabilities from HDB-SCAN are zero. Skipped further computation.')

plot_histograms(samples_prob, bin_mode = 'std_mode', bin_val=100,
                    add_cdf = False,
                    title_text = f'probabilities ({non_zero_count})')

df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

plot_clustering(x_tsne_2d, labels=Mixed_y_labels, ground_truth=True, run_id = run_id)

plot_clustering(x_tsne_2d, samples_label, samples_prob, run_id = run_id)

plt.show()

tot_end = time.time()
print("Clustering elapsed time : %0.1fs" % (tot_end - tot_start))

