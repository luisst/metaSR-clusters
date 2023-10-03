import torch
import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import IsolationForest
from scipy import stats
import numpy as np
import sys
import time

from sklearn.manifold import TSNE
import pandas as pd

import torch.nn.functional as F
from utils_metaSR import get_DB_aolme, load_model, d_vector_dict_labels_aolme 

def plot_styles(df_mixed, speakers_to_int_dict):
    marker_sizes = []
    for label in df_mixed['y']:
        if (30<=label<=40):
            marker_sizes.append(200)
        elif label == -10:
            marker_sizes.append(80)
        else:
            marker_sizes.append(60)

    marker_styles = {} 
    for current_label_y in set(df_mixed['y']):
        if 30 <= current_label_y <= 40:
            marker_styles[current_label_y] = 's'
        elif current_label_y == -10:
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
        -10: 'black'
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

    # Iterate through the dictionary and concatenate tensors
    for label, tensor in dict_data_input.items():
        # Repeat the label for each row in the tensor
        labels = [label] * tensor.shape[0]
        
        # Append labels and data to the respective lists
        labels_list.extend(labels)
        data_list.append(tensor)

    # Concatenate the tensors in data_list along the first dimension (rows)
    X_data = torch.cat(data_list, dim=0)

    speaker_labels_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(labels_list)))])
    if 'noises' in speaker_labels_dict.keys():
        speaker_labels_dict['noises'] = -10

    y_lbls = [speaker_labels_dict[x] for x in labels_list]
    y_data = np.array(y_lbls)

    return X_data, y_data, speaker_labels_dict 


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

def separate_dict_embeddings(dict_embeddings, percentage_test, remove_outliers='None', verbose = False):
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

            test_sampled_data = samples[test_indices]
            train_sampled_data = samples[train_indices]

        # Store the sampled data in the new dictionary
        dict_test_data[label] = test_sampled_data 

        # Define outliers
        if verbose:
            print(f'Key in prototypes: {label}')
        
        
        if remove_outliers == 'IQR':
            filtered_data = remove_outliers_IQR(train_sampled_data.cpu().numpy())
            dict_train_data[label] = filtered_data 
        elif remove_outliers == 'None':
            dict_train_data[label] = train_sampled_data 
        else: 
            sys.error('Remove_outliers string not supported')


    X_test, y_test, speaker_labels_dict_test = convert_dict_to_tensor(dict_test_data)
    X_train, y_train, speaker_labels_dict_train = convert_dict_to_tensor(dict_train_data)

    if speaker_labels_dict_test != speaker_labels_dict_test:
        sys.error('speaker_labels_dict from Train and Test are not the same')

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

def d_vectors_pretrained_model(test_feat_dir, percentage_test, remove_outliers, use_cuda=True, verbose = False):

    test_db = get_DB_aolme(test_feat_dir)
    n_classes = 5994 # from trained with vox1
    cp_num = 100

    log_dir = 'saved_model/baseline_' + str(0).zfill(3)

    # print the experiment configuration
    print('\nNumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_db['labels']))))

    # load model from checkpoint
    model = load_model(log_dir, cp_num, n_classes, True)


    dict_embeddings = d_vector_dict_labels_aolme(test_db, model)


    return separate_dict_embeddings(dict_embeddings, 
                                    percentage_test, 
                                    remove_outliers=remove_outliers, 
                                    verbose = verbose)


def gen_tsne(Mixed_X_data, Mixed_y_labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=900)
    tsne_results = tsne.fit_transform(Mixed_X_data)

    df_mixed = pd.DataFrame()
    df_mixed['y'] = Mixed_y_labels
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    return df_mixed


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


def plot_clustering(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
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

        class_index = np.where(labels == k)[0]
        print(f'label: {k} \tcolor: {col} \tlen:{len(class_index)}')
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
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()