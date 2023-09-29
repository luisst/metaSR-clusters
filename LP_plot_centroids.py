
from __future__ import print_function
import os
import time
import argparse
import warnings
import pandas as pd
import seaborn as sns
# %matplotlib inline
from sklearn.manifold import TSNE
import sys

from scipy import stats
from numpy import concatenate
import time
import matplotlib.pyplot as plt
import re
import constants

import torch.nn.functional as F
from torch.autograd import Variable

from str2bool import str2bool
from generator.SR_Dataset import *
from generator.DB_wav_reader import read_feats_structure, read_feats_structure_aolme
from model.model import background_resnet
import pandas as pd
from sklearn.ensemble import IsolationForest


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

parser = argparse.ArgumentParser()
# Loading setting
parser.add_argument('--use_cuda', type=str2bool, default=True, help='Use cuda.')
parser.add_argument('--gpu', type=int, default=0, help='GPU device number.')
parser.add_argument('--n_folder', type=int, default=0, help='Number of folder.')
parser.add_argument('--cp_num', type=int, default=100, help='Number of checkpoint.')
parser.add_argument('--data_type', type=str, default='vox2', help='vox1 or vox2.')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
log_dir = 'saved_model/baseline_' + str(args.n_folder).zfill(3)


def plot_styles(df_mixed, speakers_to_int_dict):
    marker_sizes = []
    for label in df_mixed['y']:
        if (30<=label<=40):
            marker_sizes.append(200)
        elif label == 99:
            marker_sizes.append(80)
        else:
            marker_sizes.append(60)

    marker_styles = {} 
    for current_label_y in set(df_mixed['y']):
        if 30 <= current_label_y <= 40:
            marker_styles[current_label_y] = 's'
        elif current_label_y == 99:
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
        99: 'black'
    }

    # To print the centroids!
    alpha_values = [0.3 if label in [21,22,23,24,25] else 1.0 for label in df_mixed['y']]

    # alpha_values = []
    # for label in df_mixed['y']:
    #     if 10<label<20:
    #         alpha_values.append(0.3)
    #     else:
    #         alpha_values.append(1.0)

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

def extract_label(filename):
    match = re.search(r'(?<=segment_\d\d\d_)[A-Za-z0-9]+(?=_\d+\.\w\w\w)|(?<=group_background_)[A-Za-z0-9]+(?=_\d+\.\w\w\w)', filename)
    if match:
        return match.group()
    else:
        return None


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
            y_labels_pred.append(99)
    
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


def main_aolme():
    test_feat_dir = [constants.CENTROID_FEAT_AOLME]
    test_db = get_DB_aolme(test_feat_dir)
    n_classes = 5994 # from trained with vox1

    # print the experiment configuration
    print('\nNumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_db['labels']))))

    # load model from checkpoint
    model = load_model(args.use_cuda, log_dir, args.cp_num, n_classes)

    # enroll and test
    tot_start = time.time()

    dict_embeddings = d_vector_dict_labels_aolme(test_db, model)

    # parameter of percentage of test 
    percentage_test = 0.6

    X_train, y_train, X_test, y_test, \
    speaker_labels_dict_train = separate_dict_embeddings(dict_embeddings, 
                                                        percentage_test, 
                                                        remove_outliers='None', 
                                                        verbose = False)

    prototype_tensor, prototypes_labels = generate_prototype(X_train, y_train, verbose=False)

    #### ---------------------------(A) - Simple Approach --------------------------
    prototypes_labels = prototypes_labels.cpu().numpy().astype(int)
    y_labels_pred = cos_sim_filter(X_test, prototype_tensor, prototypes_labels, th=0.8)
    ### -----------------------------------------------------------------------------
    ## Convert test tensor into numpy array
    train_All_data = X_train.cpu().numpy()
    test_All_data = X_test.cpu().numpy()
    prototype_np = prototype_tensor.cpu().numpy()

    # Function to map labels [1 ~ 6] -> prototype labels
    for idx, current_label in enumerate(prototypes_labels):
        prototypes_labels[idx] = current_label + 30 

    # Function to map labels [1 ~ 6] -> train labels
    for idx, current_label in enumerate(y_train):
        y_train[idx] = current_label + 20 

    # Function to map labels [1 ~ 6] -> pred labels
    y_test_pred = np.zeros_like(y_labels_pred) 
    for idx, current_label in enumerate(y_labels_pred):
        if current_label == y_test[idx]:
            y_test_pred[idx] = current_label
        elif current_label == 99:
            y_test_pred[idx] = current_label
        else:
            y_test_pred[idx] = current_label + 10 

    ## Plot A: join prototypes and test labels
    Mixed_X_data = np.concatenate((prototype_np, train_All_data, test_All_data), axis=0)
    Mixed_y_labels = np.concatenate((prototypes_labels, y_train, y_labels_pred), axis=0)

    print(f'Number of Train data: {len(y_train)} \t Number of Predicted data: {len(y_labels_pred)}')


    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=900)
    tsne_results = tsne.fit_transform(Mixed_X_data)

    print('t-sne done! time elapsed: {} seconds'.format(time.time()-time_start))

    df_mixed = pd.DataFrame()

    df_mixed['y'] = Mixed_y_labels
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    plot_styles(df_mixed, speaker_labels_dict_train)

    tot_end = time.time()

    print("total elapsed time : %0.1fs" % (tot_end - tot_start))
    ### --------------------------------------------------------------------------------


def get_DB_aolme(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        # print(f'This is the ith in get_DB" {i}')
        tmp_DB, _, _ = read_feats_structure_aolme(i, idx)
        DB = DB.append(tmp_DB, ignore_index=True)

    return DB


def load_model(use_cuda, log_dir, cp_num, n_classes):
    model = background_resnet(num_classes=n_classes)

    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_d_vector(filename, model):
    input, label = test_input_load(filename)
    label = torch.tensor([1]).cuda()

    input = normalize_frames(input, Scale=c.USE_SCALE)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        if args.use_cuda:
            #load gpu
            input = input.cuda()
            label = label.cuda()

        activation = model(input) #scoring function is cosine similarity so, you don't need to normalization

    return activation, label


def normalize_frames(m, Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def test_input_load(filename):
    feat_name = filename.replace('.wav', '.pkl')
    mod_filename = os.path.join(c.TEST_FEAT_DIR, feat_name)

    file_loader = read_MFB
    input, label = file_loader(mod_filename)  # input size :(n_frames, dim), label:'id10309'

    return input, label

def d_vector_dict_labels_aolme(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    label_dict = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            tmp_dict_entry = test_DB['filename']
            # print(f'test_db: {tmp_dict_entry}')
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key_filename = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key_filename = os.path.splitext(key_filename)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            speakerID_clusters = extract_label(key_filename)

            if speakerID_clusters in label_dict:
                # Label already there, append tensor -> shape [1, 256]
                label_dict[speakerID_clusters] = torch.cat((label_dict[speakerID_clusters], enroll_embedding), dim=0)
            else:
                label_dict[speakerID_clusters] = enroll_embedding 

            # print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, speakerID_clusters))

    return label_dict


def d_vector_queries_aolme(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    dict_embeddings = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            tmp_dict_entry = test_DB['filename']
            print(f'test_db: {tmp_dict_entry}')
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            dict_embeddings[key] = enroll_embedding
            print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    return dict_embeddings



if __name__ == '__main__':
    main_aolme()