from __future__ import print_function
import os
import time
import warnings
import time
import constants
import numpy as np
from pathlib import Path
import sklearn
import kmapper as km
import warnings
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan

warnings.filterwarnings('ignore', category=FutureWarning)
from utils_luis import d_vectors_pretrained_model

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



warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 


test_feat_dir = constants.CHUNKS_FEAT_STG2_AZURE_IRMA

noise_type = 'none'
dataset_type = 'azure_chunks_stg3_Testset_Irma'

pca_elem = 16
min_cluster_size=5
hdb_mode = 'leaf'
perplexity_val = 15 
n_iter = 900


compute_flag = True


run_id = f'{dataset_type}_Kmapper_{noise_type}'

if compute_flag:
    percentage_test = 0.0
    remove_outliers = 'None'
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


    X_data_and_labels = [X_train, X_train_paths, y_train]
    with open(f'{run_id}.pickle', "wb") as file:
        pickle.dump(X_data_and_labels, file)

output_folder_path = Path(test_feat_dir).parent.joinpath(f'{dataset_type}_{noise_type}')
output_folder_path.mkdir(parents=True, exist_ok=True)

with open(f'{run_id}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
X_train, X_train_paths, y_train = X_data_and_labels

tot_start = time.time()
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

# projected_data = mapper.fit_transform(Mixed_X_data,
#                                       projection= [sklearn.manifold.Isomap(n_components=100, 
#                                                                            n_jobs=-1), 
#                                                     umap.UMAP(n_components=2,
#                                                                       random_state=1)],
#                                       scaler=[None, 
#                                                sklearn.preprocessing.MinMaxScaler()])


# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    # clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    clusterer=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        cluster_selection_method = hdb_mode),
    cover=km.Cover(20, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member

# Print longest link
my_links = graph['links']
key_of_longest_list, max_length = find_key_of_longest_list(my_links)

print(f'Key of the longest list:{key_of_longest_list} \t Max length:{max_length}')

# Save the graph object to a file
with open(f'mapper_graph_{run_id}.pkl', 'wb') as f:
    pickle.dump(graph, f)

mapper.visualize(
    graph,
    title=run_id,
    path_html=f"output/{run_id}_PCA16_tsne_20Hdbscan04.html",
    color_values=y_train,
    color_function_name="labels",
)

tot_end = time.time()
