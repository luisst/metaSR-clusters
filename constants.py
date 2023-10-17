from configure import save_path

# VoxCeleb2 train
TRAIN_AUDIO_VOX2 = save_path+'/voxceleb2/dev/wav'
TRAIN_FEAT_VOX2 = save_path+'/voxceleb2/dev/feat'

# VoxCeleb1 train
TRAIN_AUDIO_VOX1 = save_path+'/voxceleb1/dev/wav'
TRAIN_FEAT_VOX1 = save_path+'/voxceleb1/dev/feat'

# VoxCeleb1 test
TEST_AUDIO_VOX1 = save_path+'/voxceleb1/test/wav'
TEST_FEAT_VOX1= save_path+'/voxceleb1/test/feat'

TEST_AUDIO_TTS_GT='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS_voices/test/wav'
TEST_FEAT_TTS_GT='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS_voices/test/feat'


TEST_AUDIO_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/Groups4speakers/test/wav'
TEST_FEAT_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/Groups4speakers/test/feat'

CENTROID_AUDIO_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma/test/wav'
CENTROID_FEAT_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma/test/feat'

CENTROID_AUDIO_AOLME_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noises/test/wav'
CENTROID_FEAT_AOLME_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noises/test/feat'

GROUP_AUDIO_SAMPLES = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/group_samples/test/wav'
GROUP_FEAT_SAMPLES = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/group_samples/test/feat'

CENTROID_AUDIO_AOLME_MANUAL_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesManual/test/wav'
CENTROID_FEAT_AOLME_MANUAL_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesManual/test/feat'

CENTROID_AUDIO_AOLME_TTS2_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2/test/wav'
CENTROID_FEAT_AOLME_TTS2_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2/test/feat'

CENTROID_AUDIO_AOLME_TTS2ct_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2ct/test/wav'
CENTROID_FEAT_AOLME_TTS2ct_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2ct/test/feat'

USE_LOGSCALE = True
USE_NORM=  True
USE_DELTA = False
USE_SCALE = False

SAMPLE_RATE = 16000
FILTER_BANK = 40