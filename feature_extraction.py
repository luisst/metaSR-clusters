import numpy as np
import constants as c
import os
import pickle # For python3
from pathlib import Path
from python_speech_features import *
import sys
import argparse

import scipy.io as sio
import scipy.io.wavfile

def extract_MFB_aolme(current_input_path, output_feats_folder):

    sr, audio = sio.wavfile.read(current_input_path)
    features, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025, winfunc=np.hamming)

    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))
        
    if c.USE_DELTA:
        delta_1 = delta(features, N=1)
        delta_2 = delta(delta_1, N=1)
        
        features = normalize_frames(features, Scale=c.USE_SCALE)
        delta_1 = normalize_frames(delta_1, Scale=c.USE_SCALE)
        delta_2 = normalize_frames(delta_2, Scale=c.USE_SCALE)
        features = np.hstack([features, delta_1, delta_2])

    if c.USE_NORM:
        features = normalize_frames(features, Scale=c.USE_SCALE)
        total_features = features

    else:
        total_features = features


    curent_output_path = output_feats_folder / (current_input_path.stem + '.pkl')

    ## TO-DO: Save only features, no label
    feat_and_label = {'feat':total_features, 'label':0}

    with open(curent_output_path, 'wb') as fp:
        pickle.dump(feat_and_label, fp)


def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


input_wavs_folder_ex = Path('data/processed_data/')
output_feats_folder_ex = Path('data/processed_data/feats/')

parser = argparse.ArgumentParser()

parser.add_argument('--wavs_folder', default=input_wavs_folder_ex , help='Path to the folder containing the WAV files')
parser.add_argument('--output_feats_folder', default=output_feats_folder_ex, help='Path to the folder to save the extracted features')

args = parser.parse_args()

wavs_folder = Path(args.wavs_folder)
output_feats_folder = Path(args.output_feats_folder)

list_of_wavs = sorted(list(wavs_folder.glob('*.wav')))

# Print the number of files to process
print(f'Number of files to process: {len(list_of_wavs)}')

if len(list_of_wavs) == 0:
    sys.exit("No files to process")

count = 0

for current_wav_path in list_of_wavs:
    extract_MFB_aolme(current_wav_path, output_feats_folder)
    count = count + 1
    print(f'{count} - feature extraction: {current_wav_path.name}')