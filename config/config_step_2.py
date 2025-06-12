# config_step_2.py - Feature Extraction Configuration

import torch
from config.inference_model_config import InferenceModelConfig

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
# FACE_RECOGNITION_MODEL_PATH = "/home/ssd_storage/experiments/ng_ids_imgs_tradeoff/vgg_pre_trained_models/face_trained_vgg16_119.pth"
FACE_RECOGNITION_MODEL_PATH = "data/models/citizen4_resnet_vggface2pretrained_finetuned_frame_step=24_lr_sched.pth"

# --- FEATURE EXTRACTION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.75  # Similarity threshold for matching

FACE_RECOGNITION_MODEL_CONFIG = InferenceModelConfig(
        arch="resnet",
        pretrained=True,
        train_mode=False,
        device=DEVICE,
        checkpoint_path=None,
        freeze_grads=False,
        additional_trainable_keywords=None,
        data_parallel_patch=False,
        show_logs=True,
        copy_batchnorm_stats_from_pretrained=False,
        embedding_layer_name=RESNET_EMBEDDING_LAYER_NAME)