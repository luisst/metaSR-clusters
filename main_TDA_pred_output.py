from __future__ import print_function
import os
import time
import warnings
import time
import re
import constants
from pathlib import Path
import sklearn

import matplotlib.pyplot as plt
import kmapper as km
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
import hdbscan
import argparse
import sys
import pprint

warnings.filterwarnings('ignore', category=FutureWarning)
from utils_luis import d_vectors_pretrained_model, \
find_connected_nodes, get_groups_alt, copy_arrays_to_folder 

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

pca_elem = 16
min_cluster_size=5
hdb_mode = 'leaf'

n_cubes = 20
perc_overlap = 0.4

perplexity_val = 15 
n_iter = 900

compute_flag = True

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

feats_pickle_ex = Path('')
wavs_folder_ex = Path('')
output_folder_path_ex = Path('')

parser = argparse.ArgumentParser()

parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', help='string with the HDB-SCAN run name')
parser.add_argument('--exp_name', help='string with the experiment name')
# parser.add_argument('TDA_params', help='string with the Keppler mapper name')

args = parser.parse_args()

feats_pickle_path = Path(args.input_feats_pickle)
output_folder_path = Path(args.output_pred_folder)

run_params = args.run_params
Exp_name = args.exp_name

with open(f'{feats_pickle_path}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
X_train, X_train_paths, y_train = X_data_and_labels

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


run_id = f'{Exp_name}_TDA'
verbose =  True

data_standardized = StandardScaler().fit_transform(X_train)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

##Fit and transform data
projected_data = mapper.fit_transform(X_train,
                                      projection=[sklearn.decomposition.PCA(n_components=pca_elem),
                                                sklearn.manifold.TSNE(n_components=2,
                                                                      verbose=False,
                                                                      perplexity=perplexity_val,
                                                                      n_iter=n_iter)])

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        cluster_selection_method = hdb_mode),
    cover=km.Cover(n_cubes, perc_overlap),
)
 
current_nodes = graph['nodes']
if not current_nodes:
    sys.exit("Empty 'current_nodes' dictionary. Exiting program. \n Parameters produced no nodes")

mapper.visualize(
    graph,
    title=run_id,
    ## TO-DO: Change the path to the output folder
    path_html=f"{output_folder_path}/{run_id}_Hdbscan.html",
    color_values=y_train,
    color_function_name="labels",
)

my_nodes_dict = graph['links']
my_nodes_list = list(graph['nodes'].keys())
print(f'Node list len: {len(my_nodes_list)}')

my_representative_nodes = get_groups_alt(my_nodes_dict)
lbl_idx = 0
for current_unique_name, current_group_len in my_representative_nodes:

    connected_nodes = find_connected_nodes(current_unique_name, my_nodes_dict)
    # print(f'\n\nConnected nodes {connected_nodes} ')

    # if len(connected_nodes) != current_group_len:
    #     print(f'\t\tfrom groups:{current_group_len}\t len: {len(connected_nodes)}')

    # Skip the nodes that are connected to less than 4 nodes
    if current_group_len < 4:
        continue

    my_unique_nodes = []
    for idx in connected_nodes:
        my_unique_nodes.extend(graph['nodes'][idx])

    # print(f'Extended list {my_unique_nodes}')
    # Remove duplicates
    my_unique_nodes = list(set(my_unique_nodes))

    # print(f'Unique list {my_unique_nodes}')

    print(f'\n\nProcessing node {current_unique_name} - lbl: {lbl_idx}')

    folder_path = output_folder_path.joinpath(str(lbl_idx))
    lbl_idx += 1

    #Store the wavs from a given indexs
    copy_arrays_to_folder(X_train_paths, my_unique_nodes, folder_path)


# Define the path to save the chart
current_fig_path = output_folder_path.joinpath(f'{run_id}_chart.png')

# Get the list of subfolders in the output folder
subfolders = [folder for folder in output_folder_path.iterdir() if folder.is_dir()]
if verbose:
    print(f'Connected Components {len(subfolders)}: {[subfolder.name for subfolder in subfolders]}')
# Get the number of wav files in each subfolder
num_wav_files = [len(list(subfolder.glob("*.wav"))) for subfolder in subfolders]


# Add a percentage of stored wavs to the total
my_percentage = sum(num_wav_files) / len(X_train_paths) * 100
my_title = f'Memberships wavs from total wavs: {my_percentage:.2f}%\n{run_id}' 

# Create a figure and axis
my_fig, ax = plt.subplots(figsize=(12, 6))

# Create the pie chart
ax.pie(num_wav_files, labels=[subfolder.name for subfolder in subfolders], autopct='%1.1f%%')
ax.set_title(my_title)
my_fig.savefig(current_fig_path, dpi=300)