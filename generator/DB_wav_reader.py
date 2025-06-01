import logging
import os
from glob import glob
import sys
from pathlib import Path

# import librosa
import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('log_path', 'log.txt')

    message = " ".join(str(a) for a in args)
    print(message, **kwargs)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def find_feats(directory, pattern='**/*.pkl'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_feats_structure(directory, test=False):
    DB = pd.DataFrame()
    DB['filename'] = find_feats(directory) # filename
    DB['filename'] = DB['filename'].unique().tolist()
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6]) # dataset folder name

    speaker_list = sorted(set(DB['speaker_id']))  # len(speaker_list) == n_speakers
    if test: spk_to_idx = {spk: i+1211 for i, spk in enumerate(speaker_list)}
    else: spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name

    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB, len(DB), num_speakers


def read_feats_structure_aolme(directory, test=False):
    DB = pd.DataFrame()
    DB['filename'] = find_feats(directory) # filename
    DB['filename'] = DB['filename'].unique().tolist()
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths

    return DB 

def read_feats2(feat_path_dir, n_shot, n_query, dataset_id='tts3', log_path=None):
    DB = pd.DataFrame()
    # List all files with *.pkl in directory, pathlib style
    DB['filename'] = list(feat_path_dir.glob('*.pkl'))

    DB['speaker_id'] = DB['filename'].apply(lambda x: x.stem.split('_')[1]) # speaker name
    DB['dataset_id'] = dataset_id # dataset name

    # Convert to string
    DB['filename'] = DB['filename'].astype(str)

    speaker_list = sorted(set(DB['speaker_id']))  # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name


    # Filter out speakers with less than n_shot + n_query samples
    speaker_counts = DB['labels'].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= (n_shot + n_query)].index
    DB = DB[DB['labels'].isin(valid_speakers)]

    # Print the deleted speakers
    deleted_speakers = speaker_counts[speaker_counts < (n_shot + n_query)].index
    if len(deleted_speakers) > 0:
        log_print('Deleted speakers with less than {} samples: {}'.format(n_shot + n_query, ', '.join(deleted_speakers.astype(str))), log_path=log_path)
    else:
        log_print('No speakers were deleted. All speakers have at least {} samples.'.format(n_shot + n_query), log_path=log_path)

    # Update speaker list after filtering
    valid_speakers = sorted(set(DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(valid_speakers)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    log_print('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)), log_path=log_path)


    log_print('Filtered DB to {} files with {} speakers having at least {} samples each.'.format(
        str(len(DB)).zfill(7), str(len(valid_speakers)).zfill(5), n_shot + n_query), log_path=log_path)


    return DB, len(DB), num_speakers