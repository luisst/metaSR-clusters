import os
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import roc_curve
import sys
import pandas as pd

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
n_shot_test=1
n_query_test=3
nb_class_test=0
max_iter_test = 500  # number of episodes to run
# Test setting
enroll_length=400
test_length=100


TEST_FEAT_AOLME = TEST_DATA_FOLDER / 'input_feats_groups'
TEST_WAV_AOLME = TEST_DATA_FOLDER / 'input_wavs_groups'

veri_test_dir = TEST_FEAT_AOLME / 'test_pairs_groups_aolme.txt'
norm_flag = True
samples_flag = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

dataset_name = 'Aolme'  # dataset name
params_name = f's{n_shot_test}_q{n_query_test}_c15_{max_iter_test}_4'  # parameters name

run_id = f'TEST_{dataset_name}_{params_name}'
log_dir = 'saved_model/' + run_id
log_path = log_dir + f'/{run_id}_log.txt'
results_query_path = log_dir + f'/{run_id}_results_query.txt'
results_misslabeled_path = log_dir + f'/{run_id}_results_misslabeled.txt'

# Create log directory if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

finetrained_path = 'saved_model/checkpoint_100_original.pth'  # path to pre-trained model

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('log_path', 'log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_model_test(finetrained_path, n_classes=5994):
    model = background_resnet(num_classes=n_classes)
    log_print('=> loading checkpoint', log_path=log_path)
    # load pre-trained parameters
    checkpoint = torch.load(finetrained_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def evaluation(test_generator, model, use_cuda, n_shot_test, n_query_test, nb_class_test):

    total_acc = []
    ans_episode, accum_samples_epi = 0, 0
    log_interval = 100
    total_idx = 0
    misslabeled_files = []
    correct_files = []

    # switch to test mode
    model.eval()
    debug_idx = 0
    debug_total = 0
    with torch.no_grad():
        # for batch_idx, (data) in enumerate(test_loader):
        for t, (data) in test_generator:
            inputs, targets_g, filenames_batch = data  # target size:(batch size), input size:(batch size, 1, n_filter, T)
            support, query = inputs

            # Separate support and query filenames
            support_filenames = filenames_batch[:n_shot_test * nb_class_test]
            query_filenames = filenames_batch[n_shot_test * nb_class_test:]

            len_supp = support.size(0)  # number of support samples
            len_query = query.size(0)  # number of query samples
            debug_total += len_supp + len_query
            log_print(f'{debug_idx}, len_supp: {len_supp}, len_query: {len_query}| Acc: {debug_total}', log_path=log_path)
            debug_idx += 1

            #normalize sliced input
            if USE_NORM:
                support = support - torch.mean(support, dim=3, keepdim=True)
                query = query - torch.mean(query, dim=3, keepdim=True)
            n_samples = query.size(0)  # batch size

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

            log_print(f'\n\tBatch {t+1}/{max_iter_test}', log_path=results_query_path, print=False)

            # Calculate again the accuracy for each query sample and log it with the corresponding filenames
            query_preds = torch.max(angle_e, 1)[1].long().view(targets_e.size())
            for i in range(len(query_preds)):
                pred_label = query_preds[i].item()
                true_label = targets_e[i].item()
                query_filename = query_filenames[i]

                log_print(f'Query {i+1}/{len(query_preds)} Query: {query_filename}, '
                          f'Predicted: {pred_label}, True: {true_label}', log_path=results_query_path, print=False)
                
                if pred_label != true_label:
                    misslabeled_files.append(query_filename)
                
                if pred_label == true_label:
                    correct_files.append(query_filename)

            log_print(f'\tSupport filenames: {support_filenames}', log_path=results_query_path, print=False)

            ans_episode += temp_ans
            accum_samples_epi += n_samples
            acc_episode = 100. * ans_episode / accum_samples_epi

            # if t % log_interval == 0:
            stds = np.std(total_acc, axis=0)
            ci95 = 1.96 * stds / np.sqrt(len(total_acc))
            log_print(('{}-Overall Accuracy {}-shot = {:.2f}({:.2f})').format(total_idx, n_shot_test, acc_episode, ci95), log_path=log_path)
            log_print(f' \t OK: {temp_ans} | {n_samples} \t Acc_OK: {ans_episode} | {accum_samples_epi}', log_path=log_path)
            total_idx = total_idx + 1
    
    # Create a dictionary with misslabeled files and the count of occurrences
    misslabeled_count = {}
    for file in misslabeled_files:
        if file in misslabeled_count:
            misslabeled_count[file] += 1
        else:
            misslabeled_count[file] = 1
    
    # Create a dictionary with correct files and the count of occurrences
    correct_count = {}
    for file in correct_files:
        if file in correct_count:
            correct_count[file] += 1
        else:
            correct_count[file] = 1

    # Merge the dictionaries into a dataframe with 3 columns: filename, misslabeled_count, correct_count
    misslabeled_df = pd.DataFrame(list(misslabeled_count.items()), columns=['filename', 'misslabeled_count'])
    correct_df = pd.DataFrame(list(correct_count.items()), columns=['filename', 'correct_count'])
    results_df = pd.merge(misslabeled_df, correct_df, on='filename', how='outer').fillna(0)
    results_df['misslabeled_count'] = results_df['misslabeled_count'].astype(int)
    results_df['correct_count'] = results_df['correct_count'].astype(int)
    results_df.to_csv(results_misslabeled_path, index=True, header=True, sep='\t')




def get_eer(score_list, label_list):
    fpr, tpr, threshold = roc_curve(label_list, score_list, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    intersection = abs(1 - tpr - fpr)
    DCF2 = 100 * (0.01 * (1 - tpr) + 0.99 * fpr)
    DCF3 = 1000 * (0.001 * (1 - tpr) + 0.999 * fpr)
    log_print("Epoch=%d  EER= %.2f  Thres= %0.5f  DCF0.01= %.3f  DCF0.001= %.3f" % (
    cp_num, 100 * fpr[np.argmin(intersection)], eer_threshold, np.min(DCF2), np.min(DCF3)), log_path=log_path)


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
            enroll_speaker_id = extract_label(Path(enroll_filename), samples_flag=samples_flag)
            enroll_data_list = dict_embeddings[enroll_speaker_id]
            # Find the enroll embedding in the list of tuples (feature, filename)
            enroll_embedding = None
            for feat, filename in enroll_data_list:
                if filename.stem == enroll_filename:
                    enroll_embedding = feat
                    break

            test_speaker_id = extract_label(Path(test_filename), samples_flag=samples_flag)
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
        log_print("%d) Score:%0.4f, Label:%s" % (num, score, bool(label)), log_path=log_path)

    f.close()
    eer, eer_threshold = get_eer(score_list, label_list)
    return eer


if __name__ == '__main__':
    # Load dataset
    test_DB, length_db, num_speakers = read_feats2(TEST_FEAT_AOLME, n_shot_test, n_query_test, dataset_id='aolme_tst', log_path=log_path)
    n_classes = 5994

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_test = load_model_test(finetrained_path, n_classes)
    if use_cuda:
        model_test.cuda()

    nb_class_test=num_speakers
    # nb_class_test=15

    # Print number of speakers and length of the database
    log_print(f'Number of speakers in test set: {num_speakers}', log_path=log_path)
    log_print(f'Length of test set: {length_db}', log_path=log_path)


    # make generator for unseen speaker identification
    test_generator = metaGenerator_test(test_DB, read_MFB, enroll_length=enroll_length, test_length=test_length,
                                   nb_classes=nb_class_test, n_support=n_shot_test, n_query=n_query_test,
                                   max_iter=max_iter_test, xp=np)
    # evaluate
    evaluation(test_generator, model_test, use_cuda, n_shot_test, n_query_test, nb_class_test)

    list_of_feats = sorted(list(TEST_FEAT_AOLME.glob('*.pkl')))
    list_of_wavs = sorted(list(TEST_WAV_AOLME.glob('*.wav')))

    # Print separator
    log_print("\n" + "=" * 50, log_path=log_path)
    log_print("Starting enrollment and verification...", log_path=log_path)

    # Enroll and test
    tot_start = time.time()

    dict_embeddings = d_vector_dict_lbls(list_of_feats, model_test,
                                         list_of_wavs,
                                         norm_flag=norm_flag, samples_flag=samples_flag)

    log_print("Keys in dict_embeddings: ", dict_embeddings.keys(), log_path=log_path)

    enroll_time = time.time() - tot_start

    # Perform verification
    verification_start = time.time()
    _ = perform_verification(veri_test_dir, dict_embeddings)
    tot_end = time.time()
    verification_time = tot_end - verification_start

    log_print("Time elapsed for enroll : %0.1fs" % enroll_time, log_path=log_path)
    log_print("Time elapsed for verification : %0.1fs" % verification_time, log_path=log_path)
    log_print("Total elapsed time : %0.1fs" % (tot_end - tot_start), log_path=log_path)