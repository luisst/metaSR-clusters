from __future__ import print_function
import os
import time
import warnings
import time
import constants
import numpy as np
import matplotlib.pyplot as plt
import hdbscan


from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    plot_styles, \
    plot_clustering, concat_data

warnings.filterwarnings("ignore", message="numpy.dtype size changed")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

percentage_test = 0.6
remove_outliers = 'None'
noise_flag =  True

if noise_flag:
    test_feat_dir = [constants.CENTROID_FEAT_AOLME_NOISE]
else:
    test_feat_dir = [constants.CENTROID_FEAT_AOLME]

tot_start = time.time()

dataset_dvectors = d_vectors_pretrained_model(test_feat_dir, percentage_test, 
                                              remove_outliers, use_cuda=True)

X_train = dataset_dvectors[0]
y_train = dataset_dvectors[1]
X_test = dataset_dvectors[2]
y_test = dataset_dvectors[3]
speaker_labels_dict_train = dataset_dvectors[4]

Mixed_X_data = concat_data(X_train, X_test)
Mixed_y_labels
## ----------------------------------- Method --------------------------
hdb = hdbscan.HDBSCAN(min_cluster_size=3).fit(Mixed_X_data)


df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

Mixed_y_labels = assign_labels(y_train, y_test)

plot_clustering(x_tsne_2d, labels=Mixed_y_labels, ground_truth=True)

plot_clustering(x_tsne_2d, hdb.labels_, hdb.probabilities_)

plt.show()
tot_end = time.time()

print("total elapsed time : %0.1fs" % (tot_end - tot_start))

