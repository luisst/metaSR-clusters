from __future__ import print_function
from pathlib import Path
# import warnings
import sys
import pickle
from shutil import copy

import kmapper as km
import matplotlib.pyplot as plt
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def find_connected_nodes(node, graph, visited=None):
    if visited is None:
        visited = set()

    # Check if the node is in the graph keys or values
    if node not in visited:
        visited.add(node)
        if node in graph:
            for connected_node in graph[node]:
                find_connected_nodes(connected_node, graph, visited)
        for key, value in graph.items():
            if node in value:
                find_connected_nodes(key, graph, visited)
    return list(visited)


def copy_arrays_to_folder(arrays, indices, folder_path):

    # Create the subfolder using pathlib
    folder_path.mkdir(parents=True, exist_ok=True)

    # Loop over the indices and copy each WAV file
    for idx in indices:
        if idx < len(arrays):
            file_path = arrays[idx]
            file_name = Path(file_path).name
            destination_path = folder_path / file_name
            copy(file_path, destination_path)
            print(f"{folder_path.name}: Copied {file_name}")
        else:
            print(f"\t{folder_path.name}: Index {idx} is out of range.")

            
def find_key_of_longest_list(input_dict):
    # Initialize variables to keep track of the longest list and its key
    max_length = -1
    key_of_longest_list = None

    # Iterate over all key-value pairs in the dictionary
    for key, value in input_dict.items():
        # Check if the current list is longer than the longest list found so far
        if len(value) > max_length:
            max_length = len(value)
            key_of_longest_list = key

    return key_of_longest_list, max_length


def get_groups(my_nodes_dict):
    def dfs(node):
        if node not in visited:
            visited.add(node)
            count[0] += 1
            for neighbor in my_nodes_dict[node]:
                dfs(neighbor)

    visited = set()
    groups = []
    # Create a copy of the dictionary keys before iterating over them
    nodes_copy = list(my_nodes_dict.keys())
    for node in nodes_copy:
        if node not in visited:
            count = [1]
            dfs(node)
            groups.append((node, count[0]))
    return groups


def remove_elements(list1, list2):
    return [item for item in list1 if item not in list2]


def get_groups_alt(my_nodes_dict, verbose = False):

    candidates_nodes = list(my_nodes_dict.keys())
    my_groups = []

    for current_node in my_nodes_dict.keys():

        if verbose:
            print(f'len: {len(candidates_nodes)}')

        if current_node not in candidates_nodes:
            if verbose:
                print(f'\tNode {current_node} already processed')
            continue

        connected_nodes = find_connected_nodes(current_node, my_nodes_dict)

        candidates_nodes = remove_elements(candidates_nodes, connected_nodes)   
        candidates_nodes.insert(0, current_node)

        my_groups.append((current_node, len(connected_nodes)))

    return my_groups


noise_type = 'none'
dataset_type = 'azure_chunks_stg3_Testset_Irma'

run_id = f'{dataset_type}_Kmapper_{noise_type}'
verbose =  True

with open(f'{run_id}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
X_train, X_train_paths, y_train = X_data_and_labels

output_folder_path = Path.cwd().joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)


# Load the graph object from the file
with open(f'mapper_graph_{run_id}.pkl', 'rb') as f:
    loaded_graph = pickle.load(f)

my_nodes_dict = loaded_graph['links']
my_nodes_list = list(loaded_graph['nodes'].keys())
print(f'Node list len: {len(my_nodes_list)}')

# mapper = km.KeplerMapper(verbose=2)
# current_cluster_data = mapper.data_from_cluster_id('cube246_cluster1', loaded_graph, X_train)
# print(current_cluster_data)

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
        my_unique_nodes.extend(loaded_graph['nodes'][idx])

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
my_fig.savefig(current_fig_path, dpi=300, overwrite=True)