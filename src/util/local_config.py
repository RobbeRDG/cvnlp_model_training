from os.path import join
import pandas as pd

# Set the base path
BASE_PATH = '/workspaces/cvnlp_speech_sentiment_recognition'

# Set the code mount data path
CODE_MOUNT_FOLDER_PATH = join(BASE_PATH, 'code_mount')

# Set the dataset folder paths
DATASET_FOLDER_PATH = join(CODE_MOUNT_FOLDER_PATH, 'meld_dataset')
TRAIN_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'train_data')
DEV_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'dev_data')
TEST_DATA_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'test_data')

# Set the labels path
LABELS_FOLDER_PATH = join(DATASET_FOLDER_PATH, 'labels')
TRAIN_LABELS = pd.read_csv(join(LABELS_FOLDER_PATH, 'train_sent_emo_transformed.csv'))
DEV_LABELS = pd.read_csv(join(LABELS_FOLDER_PATH, 'dev_sent_emo_transformed.csv'))
TEST_LABELS = pd.read_csv(join(LABELS_FOLDER_PATH, 'test_sent_emo_transformed.csv'))