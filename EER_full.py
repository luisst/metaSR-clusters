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

def main_aolme():
    # load pair and test data
    veri_test_dir = 'lists/test_pair_aolme_interviews.txt' #original vox1 test set
    test_feat_dir = [constants.TEST_FEAT_AOLME]
    test_db = get_DB_aolme(test_feat_dir)
    n_classes = 5994 # from trained with vox1

    # print the experiment configuration
    print('\nnumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_db['labels']))))

    # load model from checkpoint
    model = load_model(args.use_cuda, log_dir, args.cp_num, n_classes)

    # enroll and test
    tot_start = time.time()
    dict_embeddings = enroll_per_utt_aolme(test_db, model)
    enroll_time = time.time() - tot_start

    feat_cols = [ 'ft'+str(i) for i in range(0,256) ]

    df = pd.DataFrame(columns=feat_cols)
    i = 0
    speaker_lbls = []
    for emb_key, emb_data in dict_embeddings.items():
        df.loc[i] = emb_data.cpu().numpy()[0,:]
        current_speakerid = emb_key.split('/')[0]
        speaker_lbls.append(current_speakerid)
        i = i + 1
    
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(speaker_lbls)))])
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

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("bright", 6),
        data=df,
        legend="full",
        alpha=1.0
    )

    plt.show()

    # perform verification
    verification_start = time.time()
    _ = perform_verification_aolme(veri_test_dir, dict_embeddings)
    tot_end = time.time()
    verification_time = tot_end - verification_start

    print("time elapsed for enroll : %0.1fs" % enroll_time)
    print("time elapsed for verification : %0.1fs" % verification_time)
    print("total elapsed time : %0.1fs" % (tot_end - tot_start))


def main():
    # load pair and test data
    veri_test_dir = 'lists/trial_pair_verification.txt' #original vox1 test set
    test_feat_dir = [c.test_feat_dir]
    test_db = get_DB(test_feat_dir)
    n_classes = 5994 if args.data_type == 'vox2' else 1211

    # print the experiment configuration
    print('\nnumber of classes (speakers) in test set:\n{}\n'.format(len(set(test_db['labels']))))

    # load model from checkpoint
    model = load_model(args.use_cuda, log_dir, args.cp_num, n_classes)

    # enroll and test
    tot_start = time.time()
    dict_embeddings = enroll_per_utt(test_db, model)
    enroll_time = time.time() - tot_start

    feat_cols = [ 'ft'+str(i) for i in range(0,256) ]

    df = pd.dataframe(columns=feat_cols)
    i = 0
    speaker_lbls = []
    for emb_key, emb_data in dict_embeddings.items():
        df.loc[i] = emb_data.cpu().numpy()[0,:]
        current_speakerid = emb_key.split('/')[0]
        speaker_lbls.append(current_speakerid)
        i = i + 1
    
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(speaker_lbls)))])
    y_lbls = [d[x] for x in speaker_lbls]
    
    data_subset = df.values

    time_start = time.time()
    tsne = tsne(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    print('t-sne done! time elapsed: {} seconds'.format(time.time()-time_start))

    df['y'] = y_lbls
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 40),
        data=df,
        legend="full",
        alpha=0.3
    )

    plt.show()

    # perform verification
    verification_start = time.time()
    _ = perform_verification(veri_test_dir, dict_embeddings)
    tot_end = time.time()
    verification_time = tot_end - verification_start

    print("time elapsed for enroll : %0.1fs" % enroll_time)
    print("time elapsed for verification : %0.1fs" % verification_time)
    print("total elapsed time : %0.1fs" % (tot_end - tot_start))


def get_DB(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        print(f'This is the ith in get_DB" {i}')
        tmp_DB, _, _ = read_feats_structure(i, idx)
        DB = DB.append(tmp_DB, ignore_index=True)

    return DB


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


def veri_test_parser(line):
    label = int(line.split(" ")[0])
    enroll_filename = line.split(" ")[1]
    test_filename = line.split(" ")[2].replace("\n", "")
    return label, enroll_filename, test_filename


def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    args.cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)))

    return eer, eer_threshold


def enroll_per_utt_aolme(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    dict_embeddings = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            dict_embeddings[key] = enroll_embedding
            print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    return dict_embeddings


def enroll_per_utt(test_DB, model):
    # Get enroll d-vector and test d-vector per utterance
    dict_embeddings = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            enroll_embedding, _ = get_d_vector(tmp_filename, model)
            key = os.sep.join(tmp_filename.split(os.sep)[-3:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key = os.path.splitext(key)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            dict_embeddings[key] = enroll_embedding
            print("[%s/%s] Embedding for \"%s\" is saved" % (str(i).zfill(len(str(total_len))), total_len, key))

    return dict_embeddings


def perform_verification(veri_test_dir, dict_embeddings):
    # Perform speaker verification using veri_test.txt
    f = open(veri_test_dir)
    score_list = []
    label_list = []
    num = 0

    while True:
        start = time.time()
        line = f.readline()
        if not line: break

        label, enroll_filename, test_filename = veri_test_parser(line)
        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_embedding = dict_embeddings[enroll_filename]
            test_embedding = dict_embeddings[test_filename]

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1
        end = time.time()
        print("%d) Score:%0.4f, Label:%s, Time:%0.4f" % (num, score, bool(label), end - start))

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer


def perform_verification_aolme(veri_test_dir, dict_embeddings):
    # Perform speaker verification using veri_test.txt
    f = open(veri_test_dir)
    score_list = []
    label_list = []
    num = 0

    while True:
        start = time.time()
        line = f.readline()
        if not line: break

        label, enroll_filename, test_filename = veri_test_parser(line)
        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_embedding = dict_embeddings[enroll_filename]
            test_embedding = dict_embeddings[test_filename]

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1
        end = time.time()
        print("%d) Score:%0.4f, Label:%s, Time:%0.4f" % (num, score, bool(label), end - start))

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer

if __name__ == '__main__':
    main_aolme()