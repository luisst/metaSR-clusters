import os
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import roc_curve
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from generator.DB_wav_reader import read_feats2
from generator.SR_Dataset import *

from model.model import background_resnet
from generator.meta_generator_test import metaGenerator_test

from utils_luis import d_vector_dict_lbls
from utils_metaSR import extract_label 
# from utils_metaSR import load_model


root_path = Path.home().joinpath('Dropbox','DATASETS_AUDIO')                                          
TEST_DATA_FOLDER = root_path / 'Dvectors/wavs_test_pairs'


USE_NORM = True  # Normalize sliced input
USE_SCALE = False
# Loading setting
use_cuda = True
gpu='0'
n_folder='0'
cp_num=100
# Episode setting
n_shot_test=5
n_query_test=2
nb_class_test=3
nb_episode_test=n_query_test + n_shot_test  # number of classes in each episode, total number of classes in each episode
# Test setting
enroll_length=400
test_length=100


TEST_FEAT_AOLME = TEST_DATA_FOLDER / 'input_feats'
TEST_WAV_AOLME = TEST_DATA_FOLDER / 'input_wavs'

veri_test_dir = TEST_FEAT_AOLME / 'test_pair_aolmeG.txt'
norm_flag = True
samples_flag = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

dataset_name = 'aolme'  # dataset name
params_name = 'pretrained'

run_id = f'TEST_{dataset_name}_{params_name}'
log_dir = 'saved_model/' + run_id
log_path = log_dir + f'/{run_id}_log.txt'

# Create log directory if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

finetrained_path = 'saved_model/checkpoint_100_original.pth'  # path to pre-trained model

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""
    message = " ".join(str(a) for a in args)
    print(message, **kwargs)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_model_test(finetrained_path, n_classes=5994):
    model = background_resnet(num_classes=n_classes)
    log_print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(finetrained_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def evaluation(test_generator, model, use_cuda, n_shot_test, n_query_test):

    total_acc = []
    ans_episode, n_episode = 0, 0
    log_interval = 100
    total_idx = 0

    # switch to test mode
    model.eval()
    with torch.no_grad():
        # for batch_idx, (data) in enumerate(test_loader):
        for t, (data) in test_generator:
            inputs, targets_g = data  # target size:(batch size), input size:(batch size, 1, n_filter, T)
            support, query = inputs

            #normalize sliced input
            if USE_NORM:
                support = support - torch.mean(support, dim=3, keepdim=True)
                query = query - torch.mean(query, dim=3, keepdim=True)
            current_sample = query.size(0)  # batch size

            if use_cuda:
                support = support.cuda(non_blocking=True)
                query = query.cuda(non_blocking=True)

            targets_e = tuple([i for i in range(nb_class_test)]) * (n_query_test)
            targets_e = torch.tensor(targets_e, dtype=torch.long).cuda()

            support = model(support)  # out size:(n_support * n_class, dim_embed)
            query = model(query)      # out size:(n_query   * n_class, dim_embed)

            support = support.reshape(n_shot_test, nb_class_test, -1)
            prototype = support.mean(dim=0)
            angle_e = F.linear(query, F.normalize(prototype))

            # calculate accuracy of predictions in the current episode
            temp_ans = (torch.max(angle_e, 1)[1].long().view(targets_e.size()) == targets_e).sum().item()
            total_acc.append(temp_ans/angle_e.size(0) * 100)

            ans_episode += temp_ans
            n_episode += current_sample
            acc_episode = 100. * ans_episode / n_episode

            log_print(f'\n{t}\tans_episode: {ans_episode} | temp_ans: {temp_ans}\tn_episode: {n_episode} | current_sample: {current_sample}')

            # if t % log_interval == 0:
            stds = np.std(total_acc, axis=0)
            ci95 = 1.96 * stds / np.sqrt(len(total_acc))
            log_print(('{}-Accuracy_test {}-shot ={:.2f}({:.2f})\n').format(total_idx, n_shot_test, acc_episode, ci95))
            total_idx = total_idx + 1



def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    log_print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)))


    return eer, eer_threshold


def perform_verification(veri_test_dir, dict_embeddings):
    # Perform speaker verification using veri_test.txt
    f = open(veri_test_dir)
    score_list = []
    label_list = []
    num = 0

    while True:
        line = f.readline()
        if not line: break

        label = int(line.split(" ")[0])
        enroll_filename = line.split(" ")[1]
        test_filename = line.split(" ")[2].replace("\n", "")

        # Remove the extension from the filename
        enroll_filename = enroll_filename.split('.')[0]
        test_filename = test_filename.split('.')[0]

        with torch.no_grad():
            # Get embeddings from dictionary
            enroll_speaker_id = extract_label(Path(enroll_filename), samples_flag=samples_flag, id_last=True)
            enroll_data_list = dict_embeddings[enroll_speaker_id]
            # Find the enroll embedding in the list of tuples (feature, filename)
            enroll_embedding = None
            for feat, filename in enroll_data_list:
                if filename.stem == enroll_filename:
                    enroll_embedding = feat
                    break

            test_speaker_id = extract_label(Path(test_filename), samples_flag=samples_flag, id_last=True)
            test_data_list = dict_embeddings[test_speaker_id]
            # Find the enroll embedding in the list of tuples (feature, filename)
            test_embedding = None
            for feat, filename in test_data_list:
                if filename.stem == test_filename:
                    test_embedding = feat
                    break
            
            # Verify if embeddings were found
            if enroll_embedding is None or test_embedding is None:
                sys.exit(f"Warning: Embedding not found for {enroll_filename} or {test_filename}. Skipping this pair.")
            

            score = F.cosine_similarity(enroll_embedding, test_embedding)
            score = score.data.cpu().numpy()[0]
            del enroll_embedding
            del test_embedding

        score_list.append(score)
        label_list.append(label)
        num += 1
        log_print("%d) Score:%0.4f, Label:%s" % (num, score, bool(label)))

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer


if __name__ == '__main__':
    # Load dataset
    test_DB, length_db, num_speakers = read_feats2(TEST_FEAT_AOLME, n_shot_test, n_query_test, dataset_id='aolme_tst')
    n_classes = 5994

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_test = load_model_test(finetrained_path, n_classes)
    if use_cuda:
        model_test.cuda()


    # make generator for unseen speaker identification
    test_generator = metaGenerator_test(test_DB, read_MFB, enroll_length=enroll_length, test_length=test_length,
                                   nb_classes=nb_class_test, n_support=n_shot_test, n_query=n_query_test,
                                   max_iter=nb_episode_test, xp=np)
    # evaluate
    evaluation(test_generator, model_test, use_cuda, n_shot_test, n_query_test)

    list_of_feats = sorted(list(TEST_FEAT_AOLME.glob('*.pkl')))
    list_of_wavs = sorted(list(TEST_WAV_AOLME.glob('*.wav')))

    # Enroll and test
    tot_start = time.time()

    dict_embeddings = d_vector_dict_lbls(list_of_feats, model_test,
                                         list_of_wavs,
                                         norm_flag=norm_flag, samples_flag=samples_flag)

    log_print("Keys in dict_embeddings: ", dict_embeddings.keys())

    enroll_time = time.time() - tot_start

    # Perform verification
    verification_start = time.time()
    _ = perform_verification(veri_test_dir, dict_embeddings)
    tot_end = time.time()
    verification_time = tot_end - verification_start

    log_print("Time elapsed for enroll : %0.1fs" % enroll_time)
    log_print("Time elapsed for verification : %0.1fs" % verification_time)
    log_print("Total elapsed time : %0.1fs" % (tot_end - tot_start))