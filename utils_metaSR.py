
import torch.nn.functional as F
from torch.autograd import Variable

import os
from generator.SR_Dataset import *
import pandas as pd
import re
import sys
from generator.DB_wav_reader import read_feats_structure_aolme
from model.model import background_resnet


def extract_label(filename):
    # match = re.search(r'(?<=segment_\d\d\d_)[A-Za-z0-9]+(?=_\d+\.\w\w\w)|(?<=group_background_)[A-Za-z0-9]+(?=_\d+\.\w\w\w)', filename)

    match = re.search(r'[a-zA-Z0-9]+?(?=_\d+\.\w\w\w)', filename)

    if match:
        return match.group()
    else:
        return 'sample'


def get_DB_aolme(feat_dir):
    DB = pd.DataFrame()
    for idx, i in enumerate(feat_dir):
        # print(f'This is the ith in get_DB" {i}')
        tmp_DB = read_feats_structure_aolme(i, idx)
        DB = DB.append(tmp_DB, ignore_index=True)

    return DB


def load_model(log_dir, cp_num, n_classes, use_cuda = True):
    model = background_resnet(num_classes=n_classes)

    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num).zfill(3) + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_d_vector(filename, model, use_cuda = True, norm_flag = False):
    input, label = test_input_load(filename)
    label = torch.tensor([1]).cuda()

    input = normalize_frames(input, Scale=c.USE_SCALE)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        if use_cuda:
            #load gpu
            input = input.cuda()
            label = label.cuda()

        activation = model(input) #scoring function is cosine similarity so, you don't need to normalization

        if norm_flag:
            result_tensor = F.normalize(activation, p=2.0, dim=-1)
            print(f'd-vector normalized')
        else:
            result_tensor = activation

    return result_tensor, label


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


def convert_feat_to_wav_path(filename):
    regex = r"\/test\/feat\/"

    subst = "/test/wav/"

    # Count the number of occurrences of the regex pattern
    occurrences_count = len(re.findall(regex, filename))
    # You can manually specify the number of replacements by changing the 4th argument

    if occurrences_count == 0:
        print('Error: No replacement')
        sys.exit(f'No replacement for {filename}')
    elif occurrences_count > 1:
        print('Error: More than one replacement')
        sys.exit(f'More than one replacement for {filename}')

    result = re.sub(regex, subst, filename)

    result = result.replace('.pkl', '.wav')

    return result

def d_vector_dict_labels_aolme(test_DB, model, norm_flag = False):
    # Get enroll d-vector and test d-vector per utterance
    label_dict = {}
    total_len = len(test_DB)
    with torch.no_grad():
        for i in range(len(test_DB)):
            tmp_filename = test_DB['filename'][i]
            enroll_embedding, _ = get_d_vector(tmp_filename, model, norm_flag=norm_flag)
            key_filename = os.sep.join(tmp_filename.split(os.sep)[-2:])  # ex) 'id10042/6D67SnCYY34/00001.pkl'
            key_filename = os.path.splitext(key_filename)[0] + '.wav'  # ex) 'id10042/6D67SnCYY34/00001.wav'
            speakerID_clusters = extract_label(key_filename)

            # Convert tmp_filename to wav path
            current_wav_path = convert_feat_to_wav_path(tmp_filename) 

            if speakerID_clusters in label_dict:
                label_dict[speakerID_clusters].append((enroll_embedding, current_wav_path))
            else:
                label_dict[speakerID_clusters] = [(enroll_embedding, current_wav_path)]

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
