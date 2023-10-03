from __future__ import print_function
import os
import time
import warnings
import time
import constants
import numpy as np
import torch

from utils_luis import gen_tsne, d_vectors_pretrained_model, \
    generate_prototype, cos_sim_filter, plot_styles, assign_labels_prediction, \
    concat_data

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

prototype_tensor, prototypes_labels = generate_prototype(X_train, y_train, verbose=False)

prototypes_labels = prototypes_labels.cpu().numpy().astype(int)

Mixed_X_data = concat_data(X_test, data_prototypes=prototype_tensor)

#### ---------------------------(A) - Simple Approach --------------------------
y_labels_pred = cos_sim_filter(X_test, prototype_tensor, prototypes_labels, th=0.8)

Mixed_y_labels = assign_labels_prediction(y_test, y_labels_pred,
                                    prototypes_labels =  prototypes_labels)

print(f'Number of Train data: {len(y_train)} \t Number of Predicted data: {len(y_labels_pred)}')


df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)

plot_styles(df_mixed, speaker_labels_dict_train)

tot_end = time.time()

print("total elapsed time : %0.1fs" % (tot_end - tot_start))

