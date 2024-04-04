import os
import constants
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from utils_luis import d_vectors_pretrained_model



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

test_feat_dir = constants.CHUNKS_FEAT_STG2_AZURE_IRMA

noise_type = 'none'
dataset_type = 'chunks_azure_irma_stg2'

percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

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

# Print info
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
