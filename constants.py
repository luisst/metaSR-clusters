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

TTS2_AUDIO_SAMPLES = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_samples/test/wav'
TTS2_FEAT_SAMPLES = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_samples/test/feat'

TTS2_AUDIO_GROUPS = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_groups_noise/test/wav'
TTS2_FEAT_GROUPS = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_groups_noise/test/feat'

TTS2_AUDIO_GROUPS_SPEECH = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS_groups_onlyspeech/test/wav'
TTS2_FEAT_GROUPS_SPEECH = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS_groups_onlyspeech/test/feat'

TTS2_AUDIO_GTTESTSIM = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_GT_test_sim/test/wav'
TTS2_FEAT_GTTESTSIM = '/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/TTS2_GT_test_sim/test/feat'

CENTROID_AUDIO_AOLME_MANUAL_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesManual/test/wav'
CENTROID_FEAT_AOLME_MANUAL_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesManual/test/feat'

CENTROID_AUDIO_AOLME_TTS2_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2/test/wav'
CENTROID_FEAT_AOLME_TTS2_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2/test/feat'


CHUNKS_AUDIO_STG2_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/chunks_stg2_Irma/test/wav'
CHUNKS_FEAT_STG2_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/chunks_stg2_Irma/test/feat'


CHUNKS_AUDIO_STG2_SHAS_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Irma/test/wav'
CHUNKS_FEAT_STG2_SHAS_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Irma/test/feat'

CHUNKS_AUDIO_STG2_SHAS_ALLAN ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Allan/test/wav'
CHUNKS_FEAT_STG2_SHAS_ALLAN ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Allan/test/feat'

CHUNKS_AUDIO_STG2_SHAS_LIZ ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Liz/test/wav'
CHUNKS_FEAT_STG2_SHAS_LIZ ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/shas_chunks_stg3_Testset_Liz/test/feat'


CHUNKS_AUDIO_STG2_AZURE_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/azure_chunks_stg3_Testset_Irma/test/wav'
CHUNKS_FEAT_STG2_AZURE_IRMA ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/azure_chunks_stg3_Testset_Irma/test/feat'


CENTROID_AUDIO_AOLME_TTS2ct_NOISE ='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2ct/test/wav'
CENTROID_FEAT_AOLME_TTS2ct_NOISE='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/centroid_gamma_all_Irma_noisesTTS2ct/test/feat'

USE_LOGSCALE = True
USE_NORM=  True
USE_DELTA = False
USE_SCALE = False

SAMPLE_RATE = 16000
FILTER_BANK = 40