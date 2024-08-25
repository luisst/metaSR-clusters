from __future__ import print_function
import os
import warnings
from pathlib import Path
import warnings
import pickle
import argparse

warnings.filterwarnings('ignore', category=FutureWarning)

from utils_luis import d_vectors_pretrained_model

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

min_cluster_size = 0
pca_elem = 0
hdb_mode = None
min_samples = 0

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

wavs_folder_ex = Path('')
mfcc_folder_ex = Path('')
feats_pickle_ex = Path('feats_files.pkl')

parser = argparse.ArgumentParser()

parser.add_argument('--wavs_folder', type=valid_path, default=wavs_folder_ex, help='Path to the folder to input chunks wavs paths')
parser.add_argument('--input_mfcc_folder', type=valid_path, default=mfcc_folder_ex, help='Path to the folder to load the mfcc feats')
parser.add_argument('--output_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')

args = parser.parse_args()

wavs_folder = Path(args.wavs_folder)
mfcc_folder_path = Path(args.input_mfcc_folder)
feats_pickle_path = Path(args.output_feats_pickle)

percentage_test = 0.0
remove_outliers = 'None'
plot_hist_flag = False
estimate_pca_flag = False
store_probs_flag = False

plot_mode = 'store'

dataset_dvectors = d_vectors_pretrained_model(mfcc_folder_path, percentage_test,
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

X_data_and_labels = [X_train, X_train_paths, y_train]
with open(f'{feats_pickle_path}.pickle', "wb") as file:
    pickle.dump(X_data_and_labels, file)