# config.py

import torch

# --- DATA PATHS ---
DATA_DIR = "data"
VIDEO_FOLDER = "data/videos"
JSON_FOLDER = "data/jsons"
FACES_FOLDER = "data/faces"
ACTORS_READY_FOLDER = "data/dataset"
AUGMENTED_FOLDER = "data/actors_ready_augmented"
DATASET_DIR = "data/dataset"

# --- MODEL FILES (EDIT THESE TO MATCH YOUR SYSTEM) ---
PROTOTXT_PATH = r"C:\Users\IsarShmueli\Korean_Modulate\deploy.prototxt.txt"
CAFFE_MODEL_PATH = r"C:\Users\IsarShmueli\Korean_Modulate\res10_300x300_ssd_iter_140000.caffemodel"

# --- FACE DETECTION THRESHOLDS ---
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
FRAME_SKIP = 60

# --- FEATURE EXTRACTION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.9  # Similarity threshold for matching

# --- TRAINING PARAMETERS ---
NUM_CLASSES = 48
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
