# Feature path
save_path = '/home/luis/Downloads'                                           # recommend SSD
TRAIN_FEAT_DIR_2 = save_path + '/voxceleb2/dev/feat/train_logfbank_nfilt40' # train_Vox2
TRAIN_FEAT_DIR_1 = save_path + '/voxceleb1/dev/feat/train_logfbank_nfilt40' # train_Vox1
TEST_FEAT_DIR = save_path + '/voxceleb1/test/feat/test_logfbank_nfilt40'    # test_Vox1

TEST_AUDIO_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/Only_interviews/test/wav'
TEST_FEAT_AOLME='/home/luis/Dropbox/DATASETS_AUDIO/SD_test_pairs/Only_interviews/test/feat'

# Training context window size
NUM_WIN_SIZE = 200 # 200ms == 2 seconds
SHORT_SIZE = 100   # 100ms == 1 seconds

# Settings for feature extraction
USE_NORM = True
USE_SCALE = False