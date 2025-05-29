# config_step_2.py - Feature Extraction Configuration

import torch

# --- DATA PATHS ---
DATA_DIR = "data"
FACES_FOLDER = "data/faces"
ACTORS_READY_FOLDER = "data/actors_ready"
AUGMENTED_FOLDER = "data/actors_ready_augmented"
GAZES_FOLDER = "data/gazes_csv"  # Input data folder
CSV_OUTPUTS = "data/csvs_outputs"  # For generated CSVs

#Face Recognition model embeddings constants
# Penultimate layer name of the VGG16 model (after RELU activation, before DROPOUT layer)
VGG_EMBEDDING_LAYER_NAME = 'classifier.4'
RESNET_EMBEDDING_LAYER_NAME = 'last_bn'

# From step 5 config
FACE_RECOGNITION_MODEL_PATH = "/home/ssd_storage/experiments/ng_ids_imgs_tradeoff/vgg_pre_trained_models/face_trained_vgg16_119.pth"

# --- FEATURE EXTRACTION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.75  # Similarity threshold for matching 