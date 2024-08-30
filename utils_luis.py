import torch
import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt
import random
import csv
import shutil
from sklearn.ensemble import IsolationForest
from pathlib import Path
from scipy import stats
import numpy as np
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

import matplotlib.lines as mlines


import torch.nn.functional as F
from utils_metaSR import load_model, get_d_vector_aolme, extract_label 


def filter_test_values(input_list):
    input_list_train = [x for x in input_list if x < 100]
    input_list_test = [x for x in input_list if x >= 100 ]
    
    for value in input_list_test:
        if not(value - 100 in input_list_train):
            input_list_train.append(value)
    
    return sorted(input_list_train)


def modify_predict_labels(GT_labels):

    modified_labels = []
    for elem in GT_labels:
        if elem == -1:
            modified_labels.append(elem)
        else:
            modified_labels.append(elem + 100)

    return modified_labels

def plot_styles(df_mixed, speakers_to_int_dict):
    marker_sizes = []
    for label in df_mixed['y']:
        if (30<=label<=40):
            marker_sizes.append(200)
        elif label == 6:
            marker_sizes.append(80)
        else:
            marker_sizes.append(60)

    marker_styles = {} 
    for current_label_y in set(df_mixed['y']):
        if 30 <= current_label_y <= 40:
            marker_styles[current_label_y] = 's'
        elif current_label_y == 6:
            marker_styles[current_label_y] = '^'
        else:
            marker_styles[current_label_y] = 'o'

    # Define a custom color mapping for each label
    # 0~9: Good Predictions   |   10~19: Bad Predictions
    # 20~29: Enroll (train)   |   30~39: Prototypes
    label_colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'purple',
        6: 'brown',
        11: 'red',
        12: 'blue',
        13: 'green',
        14: 'orange',
        15: 'purple',
        21: 'red',
        22: 'blue',
        23: 'green',
        24: 'orange',
        25: 'purple',
        31: 'red',
        32: 'blue',
        33: 'green',
        34: 'orange',
        35: 'purple',
        9: 'black'
    }

    # To print the centroids!
    alpha_values = [0.3 if label in [21,22,23,24,25] else 1.0 for label in df_mixed['y']]

    plt.figure(figsize=(20,14))
    scatter = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=label_colors,
        data=df_mixed,
        legend="full",
        alpha=alpha_values,
        style="y",
        s=marker_sizes,
        markers=marker_styles,
    )

    # Get the current legend handles and labels
    handles, labels = scatter.get_legend_handles_labels()

    # Define a list of labels to remove from the legend
    labels_to_remove = [x for x in set(df_mixed["y"]) if x > 10]
    labels_to_remove = [str(x) for x in labels_to_remove]

    # Create a new legend without the labels to remove
    new_handles = [h for i, h in enumerate(handles) if labels[i] not in labels_to_remove]
    new_labels = [l for l in labels if l not in labels_to_remove]

    # Update the legend with the new handles and labels
    scatter.legend(new_handles, new_labels, title="Total Numbers Dataset")

    # Get the current legend
    legend = plt.gca().get_legend()

    numbers_to_speakers_dict = {value: key for key, value in speakers_to_int_dict.items()}
    dict_speaker_stats = count_elements_and_create_dictionary(df_mixed['y'])

    # Update legend labels using the dictionary
    for text in legend.get_texts():
        original_label = int(text.get_text())
        if original_label < 10:
            if original_label in numbers_to_speakers_dict:
                new_label = numbers_to_speakers_dict[original_label]
                count_of_speaker = dict_speaker_stats[original_label]
                current_legend_text = f'Total: {new_label} - {count_of_speaker}'
                text.set_text(current_legend_text)

    plt.show()


def mean_and_print(filtered_data_np, data_np, verbose = False):

    if (filtered_data_np.shape[0] == 0) or (filtered_data_np.ndim < 2):
        return_np_data = np.mean(data_np, axis=0)
        return_np_data = torch.tensor(return_np_data.reshape(1, -1), dtype=torch.float32).cuda()
        if verbose:
            print(f'After filtering, no data left.')
    else:
        return_np_data = torch.tensor(filtered_data_np, dtype=torch.float32).cuda()

    if verbose:
        print("Original Data Shape:", data_np.shape)
        print("Filtered Data Shape:", return_np_data.shape)
    
    return return_np_data


def remove_outliers_z_score(data_np, verbose=False):
    # Calculate the Z-scores for each feature
    z_scores = stats.zscore(data_np, axis=0)

    # Set a threshold to identify outliers (e.g., 2 standard deviations)
    threshold = 2.0  # Adjust the threshold as needed

    # Identify outliers for each feature
    outliers = np.abs(z_scores) > threshold

    # Combine outlier flags across all features
    outliers_combined = np.any(outliers, axis=1)

    # Remove outliers and create a new tensor
    filtered_data_np = data_np[~outliers_combined]  # Keep inliers only

    return mean_and_print(filtered_data_np, data_np, verbose = True)

def remove_outliers_isolationforest(data_np, verbose=False):
    # contamination is how much of the data we expect to be outliers
    clf = IsolationForest(contamination=0.25)  # Adjust the contamination parameter

    # Fit the model on the data
    clf.fit(data_np)

    # Predict outliers
    outliers = clf.predict(data_np)

    filtered_data_np = data_np[outliers == 1]  # Keep inliers only

    return mean_and_print(filtered_data_np, data_np, verbose = True)

def remove_outliers_IQR(data_np, verbose = False):

    # Calculate the quartiles (Q1 and Q3) for each feature
    Q1 = np.percentile(data_np, 25, axis=0)
    Q3 = np.percentile(data_np, 75, axis=0)

    # Calculate the interquartile range (IQR) for each feature
    IQR = Q3 - Q1

    # Set a threshold to identify outliers
    threshold = 1.5  # Adjust the threshold as needed

    # Identify outliers for each feature
    outliers = ((data_np < Q1 - threshold * IQR) | (data_np > Q3 + threshold * IQR))

    # Combine outlier flags across all features
    outliers_combined = np.any(outliers, axis=1)

    # Remove outliers and create a new tensor
    filtered_data_np = data_np[~outliers_combined]  # Keep inliers only

    return mean_and_print(filtered_data_np, data_np, verbose = True)


def count_elements_and_create_dictionary(input_list):
    element_count = {}
    
    for item in input_list:
        if item in element_count:
            element_count[item] += 1
        else:
            element_count[item] = 1
            
    return element_count


def convert_dict_to_tensor(dict_data_input):
    # Initialize lists to store labels and concatenated data
    labels_list = []
    data_list = []
    path_list = []
    concatenated_tensor = torch.empty(0).cuda()

    # Iterate through the dictionary and concatenate tensors
    for label, tensor_and_path in dict_data_input.items():
        if len(tensor_and_path) != 0:
            # Repeat the label for each row in the tensor
            labels = [label] * len(tensor_and_path)
            
            # Append labels and data to the respective lists
            labels_list.extend(labels)
            for current_tuple in tensor_and_path:
                concatenated_tensor = torch.cat((concatenated_tensor, current_tuple[0]), dim=0)
                # data_list.append()
                path_list.append(current_tuple[1])

    # # Concatenate the tensors in data_list along the first dimension (rows)
    # if len(data_list) != 0:
    #     X_data = torch.cat(data_list, dim=0)
    # else:
    #     X_data = torch.empty(0)

    speaker_labels_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(labels_list)))])
    if 'noises' in speaker_labels_dict.keys():
        speaker_labels_dict['noises'] = 6

    y_lbls = [speaker_labels_dict[x] for x in labels_list]
    y_data = np.array(y_lbls)

    return concatenated_tensor, y_data, path_list, speaker_labels_dict 


def cos_sim_filter(X_test, prototype_tensor, prototypes_labels, th=0.6, verbose=False):
    cos_sim_matrix = F.linear(F.normalize(X_test), F.normalize(prototype_tensor))

    # for loop each row, print it as dict + std_dev
    for row_index, row in enumerate(cos_sim_matrix):
        std_dev_row = np.std(row.cpu().numpy())
        if verbose:
            print(f'{row_index}-{row}-{std_dev_row}')

    # calculate accuracy of predictions in the current episode
    max_values, max_indices = torch.max(cos_sim_matrix, 1)

    y_labels_pred = []
    # Assign the label based on the index
    for row_idx in range(cos_sim_matrix.size(0)):
        if max_values[row_idx] > th:
            current_new_label = prototypes_labels[max_indices[row_idx]]
            y_labels_pred.append(current_new_label)
        else:
            y_labels_pred.append(-1)
    
    y_labels_pred = np.array(y_labels_pred)
    return y_labels_pred

def separate_dict_embeddings(dict_embeddings, percentage_test,
                             remove_outliers='None',
                             return_paths = False,
                             verbose = False):
    # Calculate the total number of samples across all labels
    total_samples = sum(len(samples) for samples in dict_embeddings.values())
    
    # Calculate the label ratio based on the proportion of samples for each label
    labels_amounts = {label: len(samples) / total_samples for label, samples in dict_embeddings.items()}

    # Total number of samples in the Test
    test_samples = np.floor(percentage_test*total_samples)


    if verbose:
        print(f'\nTEST: Ratio of samples per class {labels_amounts}')

    for key in labels_amounts:
        labels_amounts[key] = int(np.floor(labels_amounts[key]*test_samples ))
    
    if verbose:
        print(f'TEST: Number of samples per class {labels_amounts}\n')


    # Initialize a dictionary to store the stratified samples
    dict_test_data = {}
    dict_train_data = {}

    # Perform stratified sampling for each label
    for label, samples in dict_embeddings.items():
        num_samples = len(samples)
        desired_num_samples = labels_amounts[label]



        # Check if there are enough samples for this label
        if num_samples >= desired_num_samples:
            test_indices = random.sample(range(num_samples), desired_num_samples)

            # List of indices that were not selected
            train_indices = [index for index in range(num_samples) if index not in test_indices]

            test_sampled_data = [samples[i] for i in test_indices]
            train_sampled_data = [samples[i] for i in train_indices]

        # Store the sampled data in the new dictionary
        dict_test_data[label] = test_sampled_data 

        # Define outliers
        if verbose:
            print(f'Key in prototypes: {label}')
        
        
        if remove_outliers == 'IQR':
            # Extract all the tensors from the tuple list   
            train_tensor_list = [t[0] for t in train_sampled_data]
            train_paths_list = [t[1] for t in train_sampled_data]

            train_filtered_data = remove_outliers_IQR(train_tensor_list.cpu().numpy())
            # Assuming your lists are named 'list1' and 'list2'
            train_tuples_list = list(zip(train_filtered_data, train_paths_list))

            dict_train_data[label] = train_tuples_list 
        elif remove_outliers == 'None':
            dict_train_data[label] = train_sampled_data 
        else: 
            sys.exit('Remove_outliers string not supported')


    X_test, y_test, X_test_path, speaker_labels_dict_test = convert_dict_to_tensor(dict_test_data)
    X_train, y_train, X_train_path, speaker_labels_dict_train = convert_dict_to_tensor(dict_train_data)

    if speaker_labels_dict_test != speaker_labels_dict_test:
        sys.error('speaker_labels_dict from Train and Test are not the same')

    if return_paths:
        return X_train, y_train, X_train_path, X_test, y_test, X_test_path, speaker_labels_dict_train
    else:
        return X_train, y_train, X_test, y_test, speaker_labels_dict_train


def generate_prototype(x_train, y_train, verbose=False):

    n_features = x_train.shape[1]

    # Get the number of unique labels using set
    unique_labels = sorted(list(set(y_train)))

    # If noise is present, delete it
    if -1 in unique_labels:
        unique_labels.remove(-1)

    k = len(unique_labels)  # Total unique labels

    # Initialize tensors to store means and counts
    prototype_tensor = torch.zeros(k, n_features).cuda()
    prototype_labels = torch.zeros(k).cuda()

    # Calculate means for each unique label

    for i, current_label in enumerate(unique_labels):
        mask = (y_train == current_label)  # Create a mask for rows with the current label
        tensors_with_label_ith = x_train[mask]  # Select rows with the current label
        prototype_tensor[i] = tensors_with_label_ith.mean(dim=0).reshape(1, n_features)
        prototype_labels[i] = current_label

    if verbose:
        # Check the shape of the concatenated tensor
        print("Shape of concatenated tensor:", prototype_tensor.shape)

    return prototype_tensor, prototype_labels


def d_vector_dict_lbls(list_of_feats, model, 
                       list_of_wavs,
                       norm_flag = False, samples_flag = True):
    
    if len(list_of_feats) != len(list_of_wavs):
        sys.exit('Error! Length of list_of_feats and wavs_paths are not the same')

    # Get enroll d-vector and test d-vector per utterance
    label_dict = {}
    with torch.no_grad():
        for path_idx, current_feat_path in enumerate(list_of_feats):
            enroll_embedding, _ = get_d_vector_aolme(current_feat_path, model, norm_flag=norm_flag)
            speakerID_clusters = extract_label(current_feat_path.name, samples_flag=samples_flag)

            # Get the current wav path
            current_wav_path = list_of_wavs[path_idx]
            if current_wav_path.stem != current_feat_path.stem:
                sys.exit('Error! Wav and Feat file names do not match')


            if speakerID_clusters in label_dict:
                label_dict[speakerID_clusters].append((enroll_embedding, current_wav_path))
            else:
                label_dict[speakerID_clusters] = [(enroll_embedding, current_wav_path)]

    return label_dict

def d_vectors_pretrained_model(feats_folder, percentage_test, remove_outliers,
                               wavs_paths,
                               return_paths_flag = False,
                               norm_flag = False,
                               use_cuda=True,
                               samples_flag=True , verbose = False):

    list_of_feats = sorted(list(feats_folder.glob('*.pkl')))
    list_of_wavs = sorted(list(wavs_paths.glob('*.wav')))
    n_classes = 5994 # from trained with vox1
    cp_num = 100

    log_dir = 'saved_model/baseline_000'
    pwd_path = Path.cwd()
    print(f'Current working directory: {pwd_path}')
    
    # load model from checkpoint
    model = load_model(log_dir, cp_num, n_classes, True)


    dict_embeddings = d_vector_dict_lbls(list_of_feats, model,
                                         list_of_wavs,
                                         norm_flag=norm_flag, samples_flag=samples_flag)


    return separate_dict_embeddings(dict_embeddings, 
                                    percentage_test,
                                    return_paths = return_paths_flag,
                                    remove_outliers=remove_outliers, 
                                    verbose = verbose)


def gen_tsne(Mixed_X_data, Mixed_y_labels,
             perplexity_val = 15, n_iter = 900,
             n_comp = 108):
    
    if n_comp == 0:
        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
        tsne_results = tsne.fit_transform(Mixed_X_data)
        print(f'PCA before t-snePRE skipped')
    else:
        data_standardized = StandardScaler().fit_transform(Mixed_X_data)
        # Numbers to try: 16, 75, 108
        pca_selected = PCA(n_components=108)
        x_low_dim = pca_selected.fit_transform(data_standardized)

        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
        tsne_results = tsne.fit_transform(x_low_dim)

    df_mixed = pd.DataFrame()
    df_mixed['y'] = Mixed_y_labels
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    return df_mixed


def gen_tsne_X(Mixed_X_data, perplexity_val = 15, n_iter = 900):

    data_standardized = StandardScaler().fit_transform(Mixed_X_data)
    # Numbers to try: 16, 75, 108
    pca_selected = PCA(n_components=108)
    x_low_dim = pca_selected.fit_transform(data_standardized)

    tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
    tsne_results = tsne.fit_transform(x_low_dim)

    return tsne_results

def assign_labels_prediction(gt_labels, pred_labels, 
                  prototypes_labels = None):


    if prototypes_labels is not None:
        # Function to map labels [1 ~ 6] -> prototype labels
        for idx, current_label in enumerate(prototypes_labels):
            prototypes_labels[idx] = current_label + 30 

    # Function to map labels [1 ~ 6] -> pred labels
    y_test_pred = np.zeros_like(gt_labels) 
    for idx, current_label in enumerate(gt_labels):
        if current_label == pred_labels[idx]:
            y_test_pred[idx] = current_label
        elif current_label == 9:
            y_test_pred[idx] = current_label
        else:
            # Failed prediction
            y_test_pred[idx] = current_label + 20 


    if prototypes_labels is not None:
        Mixed_y_labels = np.concatenate((prototypes_labels, y_test_pred), axis=0)
    else:
        Mixed_y_labels = y_test_pred 

    return Mixed_y_labels


def assign_labels_traintest(y_train, y_test, 
                  prototypes_labels = None):

    if prototypes_labels is not None:
        # Function to map labels [1 ~ 6] -> prototype labels
        for idx, current_label in enumerate(prototypes_labels):
            prototypes_labels[idx] = current_label + 30 

    # Function to map labels [1 ~ 6] -> train labels
    for idx, current_label in enumerate(y_train):
        y_train[idx] = current_label + 20 

    # Function to map labels [1 ~ 6] -> pred labels

    if prototypes_labels is not None:
        Mixed_y_labels = np.concatenate((prototypes_labels, y_train, y_test), axis=0)
    else:
        Mixed_y_labels = np.concatenate((y_train, y_test), axis=0)

    return Mixed_y_labels


def concat_data(x_test, x_train = None, data_prototypes=None):

    ## Convert test tensor into numpy array
    x_test = x_test.cpu().numpy()
    Mixed_X_data = x_test

    if x_train is not None:
        x_train = x_train.cpu().numpy()
        Mixed_X_data = np.concatenate((x_train, Mixed_X_data), axis=0)

    if data_prototypes is not None:
        prototype_np = data_prototypes.cpu().numpy()
        Mixed_X_data = np.concatenate((prototype_np, Mixed_X_data), axis=0)


    return Mixed_X_data


def plot_clustering(X, labels, probabilities=None, parameters=None, 
                    ground_truth=False, ax=None,
                    remove_outliers = False,
                    add_gt_prd_flag = True):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            if remove_outliers:
                continue

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if ground_truth:
        if n_clusters_ == 1:
            title = f"Unlabeled total samples: {len(labels)}"
        else:
            title = f"GT #n: {n_clusters_} T: {len(labels)}"
    else:
        non_outliers_percentage = (len(labels[labels != -1]) / len(labels)) * 100
        title = f"Prd #n: {n_clusters_} | Mem %: {non_outliers_percentage:.2f}%"

    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    title = title 
    ax.set_title(title)

    # # Add legend with the number of labels for each cluster
    # legend_labels = [f"C{k}: {list(labels).count(k)}" for k in unique_labels]
    legend_labels = [f": {list(labels).count(k)}" for k in unique_labels]

    # Customizing the legend to not display the marker
    legend_without_symbol = []
    for idx, label_id in enumerate(unique_labels):
        if label_id != -1:
            current_lbl_color = colors[idx]
            current_label = legend_labels[idx]
            legend_without_symbol.append(mlines.Line2D([], [], color=current_lbl_color,
                                                        marker='o', 
                                                        # linestyle='solid', 
                                                        label=current_label))
    plt.legend(handles=legend_without_symbol)

    # ax.legend(legend_labels, labelcolor=colors)
    plt.tight_layout()


def plot_clustering_predict(X, labels, parameters=None, 
                        ground_truth=False, ax=None,
                        remove_outliers = False,
                        add_gt_prd_flag = True):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = sorted(list(set(labels)))

    if ground_truth:
        # Create color palette only with 
        unique_train_test = filter_test_values(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_train_test))]
    else:
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            if remove_outliers:
                continue
        
        if ground_truth:
            if k in unique_train_test:
                idx_color = unique_train_test.index(k)
                current_col = colors[idx_color]
            else:
                idx_color = unique_train_test.index(k-100)
                current_col = colors[idx_color]
        else:
            idx_color = unique_labels.index(k)
            current_col = colors[idx_color]
        
        if k == -1:
            current_shape = "x"
        elif k > 100:
            current_shape = "^"
        else:
            current_shape = "o"


        if k == -1:
            current_marker_size = 4
        elif k > 100:
            current_marker_size = 10 
        else:
            current_marker_size = 6 

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                current_shape,
                markerfacecolor=tuple(current_col),
                markeredgecolor="k",
                markersize=current_marker_size,
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "GT" if ground_truth else "Prd"
    if add_gt_prd_flag:
        title = f"{preamble} #n: {n_clusters_}"
    else:
        title = ''
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    title = title 
    ax.set_title(title)
    plt.tight_layout()


def plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                         samples_label, samples_prob,
                         run_id, output_folder_path,
                         plot_mode):

    ## Available options: 
    ## 'show' : only plot
    ## 'store' : only store
    ## 'show_store' : plot and store fig

    combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering(x_tsne_2d, labels=Mixed_y_labels, ground_truth=True,
                    ax=axes[0])

    plot_clustering(x_tsne_2d, labels=samples_label,
                     probabilities = samples_prob,
                    remove_outliers = True, ax=axes[1])

    current_fig_path = output_folder_path.joinpath(f'{run_id}.png') 

    combined_fig.suptitle(f'{run_id}', fontsize=14)
    plt.tight_layout()


    if plot_mode == 'show':
        plt.show()
    elif plot_mode == 'store':
        combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
    elif plot_mode == 'show_store':
        combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
        plt.show()
    else:
        print(f'Error! plot_histogam plot_mode')


def plot_clustering_dual_predict(x_tsne_2d, 
                             Mixed_y_GT,
                             Mixed_y_prediction,
                            run_id, output_folder_path,
                            plot_mode,
                            plot_minus1 = True):

    ## Available options: 
    ## 'show' : only plot
    ## 'store' : only store
    ## 'show_store' : plot and store fig

    combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering_predict(x_tsne_2d, labels=Mixed_y_GT, ground_truth=True,
                    ax=axes[0])

    plot_clustering_predict(x_tsne_2d, labels=Mixed_y_prediction,
                    remove_outliers = not(plot_minus1), ax=axes[1])

    current_fig_path = output_folder_path.joinpath(f'{run_id}.png') 

    combined_fig.suptitle(f'{run_id}', fontsize=14)
    plt.tight_layout()


    if plot_mode == 'show':
        plt.show()
    elif plot_mode == 'store':
        combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
    elif plot_mode == 'show_store':
        combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
        plt.show()
    else:
        print(f'Error! plot_histogam plot_mode')


def divide_into_sublists(long_list, sublist_size):
    sublists = []
    for i in range(0, len(long_list), sublist_size):
        sublist = long_list[i:i + sublist_size]
        sublists.append(sublist)
    return sublists

def plot_clustering_subfig(list_dfs, Mixed_y_labels,
                           num_rows, num_cols,
                           params_list,
                            run_id, output_folder_path,
                            plot_mode):

    ## Available options: 
    ## 'show' : only plot
    ## 'store' : only store
    ## 'show_store' : plot and store fig

    num_arrays = len(list_dfs)
    num_subplots = num_rows * num_cols

    # Calculate the number of figures needed to display all arrays
    num_figures = int(np.ceil(num_arrays / num_subplots))

    for figure_num in range(num_figures):
        combined_fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        print(f'\nFigure {figure_num} / {num_figures}')

        for i in range(num_subplots):
            array_idx = figure_num * num_subplots + i

            if array_idx < num_arrays:
                row_idx = i // num_cols
                col_idx = i % num_cols

                current_2d_tsne = np.array(list(zip(list_dfs[array_idx]['tsne-2d-one'], 
                                                    list_dfs[array_idx]['tsne-2d-two'])))
                
                if num_rows == 1:
                    plot_clustering(current_2d_tsne, Mixed_y_labels,
                                    parameters=params_list[array_idx],
                                    ground_truth=True, ax=axes[col_idx],
                                    add_gt_prd_flag = False)
                else:
                    plot_clustering(current_2d_tsne, Mixed_y_labels,
                                    parameters=params_list[array_idx],
                                    ground_truth=True, ax=axes[row_idx, col_idx],
                                    add_gt_prd_flag = False)

        current_fig_path = output_folder_path.joinpath(f'{run_id}-{figure_num}.png') 

        combined_fig.suptitle(f'{run_id}', fontsize=14)
        plt.tight_layout()


        if plot_mode == 'show':
            plt.show()
        elif plot_mode == 'store':
            combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
        elif plot_mode == 'show_store':
            combined_fig.savefig(current_fig_path, dpi=300, overwrite=True)
            plt.show()
        else:
            print(f'Error! plot_histogam plot_mode')

def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    print(f'Min val {min_val} | max val {max_val}')
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins


def plot_histograms(input_list_all, bin_mode = 'std_mode', bin_val=100,
                    add_cdf = False,
                    title_text = 'Histogram',
                    run_id = 'histogram',
                    plot_mode = 'show',
                    output_path = '.'):
    '''
        modes:
            std_mode: 
    '''

    ## Available options: 
    ## 'show' : only plot
    ## 'store' : only store
    ## 'show_store' : plot and store fig

    if isinstance(input_list_all, np.ndarray):
        input_array = input_list_all
    else:
        input_array = np.array(input_list_all)

    if bin_mode == 'std_mode':
        bin_width = 3.5 * np.std(input_array) / len(input_array)**(1/3)
        bins = int((max(input_array) - min(input_array)) / bin_width)
    elif bin_mode == 'sturges_mode':
        bins = int(np.ceil(1 + np.log2(len(input_array))))
    elif bin_mode == 'iqr_mode':
        iqr = np.percentile(input_array, 75) - np.percentile(input_array, 25)
        bin_width = 2 * iqr / len(input_array)**(1/3)
        bins = int((max(input_array) - min(input_array)) / bin_width)
    elif bin_mode == 'bin_size':
        bins = compute_histogram_bins(input_array, bin_val)     


    plt.figure(figsize=(12, 6))    
    n, bins_hist, patches = plt.hist(x = input_array, bins=bins, color='#0504aa',
                                    alpha=0.7, rwidth=0.85)

    cum_exp_var = np.cumsum(n)
    step_axis = np.linspace(0, bins_hist[-1], bins)

    plt.step(step_axis, cum_exp_var, where='mid',
            label='Cumulative Hist', color='red')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{title_text} - {bin_mode} | bins:{bins}')

    if add_cdf:
        plt.figure(figsize=(12, 6))    
        n, bins, patches = plt.hist(x = input_array, bins=bins, color='g',
                                        alpha=0.7, rwidth=0.85, cumulative=True)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'CDF-{title_text}')

    if plot_mode == 'show':
        plt.show()
    elif plot_mode == 'store':
        current_fig_path = output_path.joinpath(f'{run_id}_{title_text}.png')
        plt.savefig(current_fig_path, dpi=300, overwrite=True)
    elif plot_mode == 'show_store':
        current_fig_path = output_path.joinpath(f'{run_id}_{title_text}.png')
        plt.savefig(current_fig_path, dpi=300, overwrite=True)
        plt.show()
    else:
        print(f'Error! plot_histogam plot_mode')

def estimate_pca_n(data):
    # Step 1: Standardize the data (mean=0, std=1)
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    data_standardized = (data - mean) / std_dev

    # Create a PCA object
    pca = PCA()

    # Fit the PCA model to the standardized data
    pca.fit(data_standardized)

    # Rule 1: Explained Variance Threshold
    threshold = 0.95  # You can adjust this threshold
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components_threshold = np.argmax(cumulative_variance_ratio >= threshold) + 1

    # Rule 2: Scree Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    fig.suptitle('Elbow Plot')
    fig.gca().grid(True)
    # fig.add_gridspec(True)
    plt.tight_layout()
    plt.show()

    # Rule 3: Kaiser's Rule
    num_components_kaiser = np.sum(pca.explained_variance_ > 1)

    print("Number of components selected by Explained Variance Threshold (Rule 1):", num_components_threshold)
    print("Number of components selected by Scree Plot (Rule 2):", 2)  # Manually choose the number of components from the plot
    print("Number of components selected by Kaiser's Rule (Rule 3):", num_components_kaiser)

def run_pca(data, n_components=16):

    # # Step 1: Standardize the data (mean=0, std=1)
    # mean = np.mean(data, axis=0)
    # std_dev = np.std(data, axis=0)
    # data_standardized = (data - mean) / std_dev

    data_standardized = StandardScaler().fit_transform(data)

    # Numbers to try: 16, 75, 108
    pca_selected = PCA(n_components=n_components)
    # Fit the PCA model to the standardized data with the selected number of components
    trained_data =  pca_selected.fit_transform(data_standardized)
    return trained_data


def run_pca_inference(train_data, test_data, n_components=16):

    data_standardized = StandardScaler().fit_transform(train_data)
    data_std_test = StandardScaler().fit_transform(test_data)

    pca_selected = PCA(n_components=n_components)

    # Fit the PCA model to the standardized data with the selected number of components
    train_data_pca =  pca_selected.fit_transform(data_standardized)

    # Predict
    test_data_pca = pca_selected.transform(data_std_test)

    return train_data_pca, test_data_pca


def store_probs(array1, array2, output_folder_path, run_id = 'current'):
    # Combine the arrays into a list of rows
    data = list(zip(array1, array2))

    # Specify the CSV file name
    csv_file = output_folder_path.joinpath(f"prob_labels_{run_id}.csv")

    # Write the data to the CSV file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        
        # Write the header row if needed
        writer.writerow(["Prob", "Label"])
        
        # Write the data rows
        writer.writerows(data)

    print(f"CSV file '{csv_file}' has been created.")

def check_0_clusters(samples_prob, samples_label, verbose = False):
    # print number of clusters:
    non_zero_count =np.count_nonzero(samples_prob)
    unique_labels = set(samples_label)
    if verbose:
        print(f'Number of clusters: {len(unique_labels) - 1}')
        print(f'hdb probs \t min: {min(samples_prob)} \t max: {max(samples_prob)} \t non_zero: {non_zero_count}')

    if non_zero_count == 0:
        return True
    else:
        return False


def organize_samples_by_label(X_test_paths, samples_label, samples_prob, wav_chunks_output_path):
    # Loop over all paths in X_test_paths
    for idx, path in enumerate(X_test_paths):
        # Create a Path object
        path_obj = Path(path)
        
        # Get the label for the current sample
        label = str(samples_label[idx])
        
        # Create a new directory path for the label if it doesn't exist
        new_dir = wav_chunks_output_path.joinpath(label)
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new name for the file with the prob appended
        new_name = f"{path_obj.stem}_{samples_prob[idx]:.2f}{path_obj.suffix}" 

        # Create a new path for the file in the new directory
        new_path = new_dir / new_name 
        
        # Copy the file to the new directory
        shutil.copy(path, new_path)


def copy_arrays_to_folder(arrays, indices, folder_path):

    # Create the subfolder using pathlib
    folder_path.mkdir(parents=True, exist_ok=True)

    # Loop over the indices and copy each WAV file
    for idx in indices:
        if idx < len(arrays):
            file_path = arrays[idx]
            new_filename = Path(file_path).stem + '_9.99.' + Path(file_path).suffix

            destination_path = folder_path / new_filename

            shutil.copy(file_path, destination_path)
            print(f"{folder_path.name}: Copied {new_filename}")
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