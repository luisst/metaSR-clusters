
from __future__ import print_function
import os
import time
import argparse
import warnings
import pandas as pd
import seaborn as sns
# %matplotlib inline
from sklearn.manifold import TSNE
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

    dict_embeddings = d_vector_queries_aolme(test_db, model)

    feat_cols = [ 'ft'+str(i) for i in range(0,256) ]

    df = pd.DataFrame(columns=feat_cols)
    i = 0
    speaker_lbls = []
    for emb_key, emb_data in dict_embeddings.items():
        df.loc[i] = emb_data.cpu().numpy()[0,:]
        speakerID_clusters = extract_label(emb_key)
        # print(f'{emb_key} - {speakerID_clusters}')

        # current_speakerid = emb_key.split('/')[0]
        speaker_lbls.append(speakerID_clusters)
        i = i + 1
    
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(speaker_lbls)))])
    # d = dict([(x+1, y) for x,y in enumerate(sorted(set(speaker_lbls)))])
    numbers_to_speakers_dict = {value: key for key, value in d.items()}

    dict_speaker_stats = count_elements_and_create_dictionary(speaker_lbls)

    y_lbls = [d[x] for x in speaker_lbls]
    
    data_subset = df.values

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=900)
    # tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(data_subset)

    print('t-sne done! time elapsed: {} seconds'.format(time.time()-time_start))

    df['y'] = y_lbls
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # # Define colors:
    # colors = {
    #     'noise': 'black',
    #     'source1': 'red',
    #     'source2': 'blue',
    # }

    # dot_colors = [colors[name] for name in names]
    # num_entries = df.shape[0]
    # for idx_name in range(0, num_entries):
    #     # Add label into the dataframe in the code
    #     if 

    # xkcd_colors = plt.xkcd_palette(random.sample(list(plt.xkcd_rgb.keys()), len(string_list)))

    # # Dictionary to store strings and colors
    # string_color_dict = {}

    # # Iterate through the list of strings
    # for string in string_list:
    #     if string not in string_color_dict:
    #         # Find an unused color from the xkcd_colors list
    #         for color in xkcd_colors:
    #             if color not in string_color_dict.values():
    #                 string_color_dict[string] = color
    #                 break

    # # Print the resulting dictionary
    # for string, color in string_color_dict.items():
    #     print(f"{string}: {color}")

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("bright", 6),
        data=df,
        legend="full",
        alpha=1.0
    )

    # Get the current legend
    legend = plt.gca().get_legend()

    # Update legend labels using the dictionary
    for text in legend.get_texts():
        original_label = int(text.get_text())
        if original_label in numbers_to_speakers_dict:
            new_label = numbers_to_speakers_dict[original_label]
            count_of_speaker = dict_speaker_stats[new_label]
            current_legend_text = f'{new_label} - {count_of_speaker}'
            text.set_text(current_legend_text)

    plt.show()

    tot_end = time.time()

    print("total elapsed time : %0.1fs" % (tot_end - tot_start))



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


def d_vector_queries_aolme(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    dict_embeddings = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            tmp_dict_entry = test_DB['filename']
            # print(f'test_db: {tmp_dict_entry}')
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            dict_embeddings[key] = enroll_embedding
            # print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    return dict_embeddings



if __name__ == '__main__':
    main_aolme()