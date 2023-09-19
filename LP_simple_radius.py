
from __future__ import print_function
import os
import time
import argparse
import warnings
import pandas as pd
import seaborn as sns
# %matplotlib inline
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
from numpy import concatenate
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import constants

import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_curve

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

    d = dict([(y,x+1) for x,y in enumerate(sorted(set(labels_list)))])
    y_lbls = [d[x] for x in labels_list]
    y_data = np.array(y_lbls)

    return X_data, y_data

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

    # speaker_lbls = list(dict_embeddings.keys())
    

    # parameter of percentage of test 
    percentage_test = 0.2

    # Calculate the total number of samples across all labels
    total_samples = sum(len(samples) for samples in dict_embeddings.values())
    
    # Calculate the label ratio based on the proportion of samples for each label
    labels_amounts = {label: len(samples) / total_samples for label, samples in dict_embeddings.items()}

    # Total number of samples in the Test
    test_samples = np.floor(percentage_test*total_samples)

    print(f'\nTEST: Ratio of samples per class {labels_amounts}')

    for key in labels_amounts:
        labels_amounts[key] = int(np.floor(labels_amounts[key]*test_samples ))
    
    print(f'TEST: Number of samples per class {labels_amounts}\n')


    # Initialize a dictionary to store the stratified samples
    dict_test_data = {}
    dict_train_data = {}
    Total_labels_start = []

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
        dict_train_data[label] = train_sampled_data 
        Total_labels_start.extend([label]*num_samples)

    X_test, y_test = convert_dict_to_tensor(dict_test_data)
    X_train, y_train = convert_dict_to_tensor(dict_train_data)


    #### Create the Train Set along with labels
    prototype = None
    for key_label in dict_train_data:
        current_avg_tensor = dict_train_data[key_label].mean(dim=0).reshape(1, 256)
        if prototype is None:
            prototype = current_avg_tensor 
        else:
            prototype = torch.cat((prototype, current_avg_tensor), dim=0)

    # Check the shape of the concatenated tensor
    print("Shape of concatenated tensor:", prototype.shape)

    #####################################################################
    # # TODO: use norm or not?
    # angle_e = F.linear(X_test, F.normalize(prototype))
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long).cuda()

    # # calculate accuracy of predictions in the current episode
    # temp_angle = torch.max(angle_e, 1)[1].long()
    # temp_view_angle = temp_angle.view(y_test_tensor.size())

    # temp_ans = (torch.max(angle_e, 1)[1].long().view(y_test_tensor.size()) == y_test_tensor).sum().item()
    # current_acc = temp_ans/angle_e.size(0) * 100
    #####################################################################

    feat_cols = [ 'ft'+str(i) for i in range(0,256) ]

    ## Convert tensor dict into numpy array for tsne
    df_prototypes = pd.DataFrame(columns=feat_cols)
    i = 0
    prototype_labels = []
    ############ >>>>>>>>>>>>>>>>>>>>>>>>>> should be dict_train_data
    for emb_key, emb_data in dict_embeddings.items():
        df_prototypes.loc[i] = emb_data.cpu().numpy()[0,:]

        prototype_labels.append(emb_key)
        i = i + 1
    prototype_data = np.array(df_prototypes.values)

    d = dict([(y,x+1) for x,y in enumerate(sorted(set(prototype_labels)))])
    y_prototype = [d[x] for x in prototype_labels]
    y_prototype = np.array(y_prototype)

    ## Convert test tensor into numpy array
    test_All_data = X_test.cpu().numpy()
    train_All_data = X_train.cpu().numpy()


    ### ------------------------------ Remove outliers Isolation Forest ----------------------------
    # Convert the PyTorch tensor to a NumPy array
    data_np = train_All_data

    # Create an Isolation Forest model
    clf = IsolationForest(contamination=0.05)  # Adjust the contamination parameter

    # Fit the model on the data
    clf.fit(data_np)

    # Predict outliers
    outliers = clf.predict(data_np)

    # You can also get the indices of the outlier samples
    outlier_indices = np.where(outliers == -1)
    print("Outlier Indices:", outlier_indices[0])   

    filtered_data_np = data_np[outliers == 1]  # Keep inliers only
    filtered_tensor = torch.from_numpy(filtered_data_np)

    # Print the new tensor without outliers
    print("Original Data Shape:", train_All_data.shape)
    print("Filtered Data Shape:", filtered_tensor.shape)

    ### ------------------------------ Remove outliers IQR ---------------------------- 

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

    # Convert the combined outlier flags to a PyTorch tensor
    outliers_tensor = torch.from_numpy(outliers_combined)

    # Print the indices of the outlier samples
    outlier_indices = torch.nonzero(outliers_tensor).squeeze()
    print("Outlier Indices:", outlier_indices)

    # Remove outliers and create a new tensor
    filtered_data_np = data_np[~outliers_combined]  # Keep inliers only
    filtered_tensor = torch.from_numpy(filtered_data_np)

    # Print the new tensor without outliers
    print("Original Data Shape:", data_np.shape)
    print("Filtered Data Shape:", filtered_tensor.shape)

    # number_unlabeled = np.count_nonzero(tran_labels == -1)
    # print(f'number of unlabeled data: {number_unlabeled}')

    # # calculate score for test set
    # lp_score = accuracy_score(y_train_mixed_gt, tran_labels)
    # # summarize score
    # print('Accuracy: %.3f' % (lp_score*100))


    ### ------------------------------ Remove outliers Z-score ---------------------------- 
    # Calculate the Z-scores for each feature
    z_scores = stats.zscore(data_np, axis=0)

    # Set a threshold to identify outliers (e.g., 2 standard deviations)
    threshold = 2.0  # Adjust the threshold as needed

    # Identify outliers for each feature
    outliers = np.abs(z_scores) > threshold

    # Combine outlier flags across all features
    outliers_combined = np.any(outliers, axis=1)

    # Convert the combined outlier flags to a PyTorch tensor
    outliers_tensor = torch.from_numpy(outliers_combined)

    # Print the indices of the outlier samples
    outlier_indices = torch.nonzero(outliers_tensor).squeeze()
    print("Outlier Indices:", outlier_indices)

    # Remove outliers and create a new tensor
    filtered_data_np = data_np[~outliers_combined]  # Keep inliers only
    filtered_tensor = torch.from_numpy(filtered_data_np)

    # Print the new tensor without outliers
    print("Original Data Shape:", data_np.shape)
    print("Filtered Data Shape:", filtered_tensor.shape)

    
    ### ------------------------------ Remove outliers DB-SCAN ---------------------------- 
# Create a DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust the parameters as needed

# Fit the DBSCAN model on the data
labels = dbscan.fit_predict(data_np)

# Identify outliers (points with label -1)
outliers = labels == -1

# Convert the outlier flags to a PyTorch tensor
outliers_tensor = torch.from_numpy(outliers)

# Print the indices of the outlier samples
outlier_indices = torch.nonzero(outliers_tensor).squeeze()
print("Outlier Indices:", outlier_indices)

# Remove outliers and create a new tensor
filtered_data_np = data_np[~outliers]  # Keep inliers only
filtered_tensor = torch.from_numpy(filtered_data_np)

# Print the new tensor without outliers
print("Original Data Shape:", data.shape)
print("Filtered Data Shape:", filtered_tensor.shape)
    # -----------------------------------------------------------------------------

    # Function to map labels [1 ~ 6] -> prototype labels
    for idx, current_label in enumerate(y_prototype):
        y_prototype[idx] = current_label + 30 

    # Function to map labels [1 ~ 6] -> train labels
    for idx, current_label in enumerate(y_train):
        y_train[idx] = current_label + 20 

    ## Plot A: join prototypes and test labels
    Mixed_X_data = np.concatenate((prototype_data, train_All_data, test_All_data), axis=0)
    Mixed_y_labels = np.concatenate((y_prototype, y_train, y_test), axis=0)


    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=900)
    tsne_results = tsne.fit_transform(Mixed_X_data)

    print('t-sne done! time elapsed: {} seconds'.format(time.time()-time_start))

    df_mixed = pd.DataFrame()

    df_mixed['y'] = Mixed_y_labels
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    marker_sizes = [100 if label > 29 else 20 for label in df_mixed['y']]
    # marker_styles = ['s' if label < 29 else 'o' for label in df_mixed['y']]

    marker_styles = {} 
    for current_label_y in set(Mixed_y_labels):
        if current_label_y >= 30:
            marker_styles[current_label_y] = 's'
        else:
            marker_styles[current_label_y] = 'o'

    # Define a custom color mapping for each label
    label_colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'purple',
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
    }


    alpha_values = [0.3 if label in [21,22,23,24,25] else 1.0 for label in df_mixed['y']]

    plt.figure(figsize=(16,10))
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

    numbers_to_speakers_dict = {value: key for key, value in d.items()}
    dict_speaker_stats = count_elements_and_create_dictionary(Total_labels_start)

    # Update legend labels using the dictionary
    for text in legend.get_texts():
        original_label = int(text.get_text())
        if original_label < 10:
            if original_label in numbers_to_speakers_dict:
                new_label = numbers_to_speakers_dict[original_label]
                count_of_speaker = dict_speaker_stats[new_label]
                current_legend_text = f'Total: {new_label} - {count_of_speaker}'
                text.set_text(current_legend_text)

    plt.show()

    #--------------------------------------------------------------------------------

    # time_start = time.time()
    # tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=900)
    # tsne_results = tsne.fit_transform(X_train_mixed)

    # print('t-sne done! time elapsed: {} seconds'.format(time.time()-time_start))

    # df['y'] = tran_labels
    # df['tsne-2d-one'] = tsne_results[:,0]
    # df['tsne-2d-two'] = tsne_results[:,1]

    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("bright", 6),
    #     data=df,
    #     legend="full",
    #     alpha=1.0
    # )

    # # Get the current legend
    # legend = plt.gca().get_legend()

    # # Update legend labels using the dictionary
    # for text in legend.get_texts():
    #     original_label = int(text.get_text())
    #     if original_label in numbers_to_speakers_dict:
    #         new_label = numbers_to_speakers_dict[original_label]
    #         count_of_speaker = dict_speaker_stats[new_label]
    #         current_legend_text = f'{new_label} - {count_of_speaker}'
    #         text.set_text(current_legend_text)

    # plt.show()

    # tot_end = time.time()

    # print("total elapsed time : %0.1fs" % (tot_end - tot_start))



def get_DB_aolme(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        print(f'This is the ith in get_DB" {i}')
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
            print(f'test_db: {tmp_dict_entry}')
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key_filename = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key_filename = os.path.splitext(key_filename)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            speakerID_clusters = extract_label(key_filename)

            if speakerID_clusters in label_dict:
                # Label already there, append tensor -> shape [1, 256]
                label_dict[speakerID_clusters] = torch.cat((label_dict[speakerID_clusters], enroll_embedding), dim=0)
            else:
                label_dict[speakerID_clusters] = enroll_embedding 

            print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, speakerID_clusters))

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