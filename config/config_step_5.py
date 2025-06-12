import torch
from config.model_config import ModelConfig

# Training data paths
ACTORS_READY_FOLDER = "data/actors_ready"  # folder containing processed actor images
FACES_FOLDER = "data/faces"  # folder containing extracted faces to be matched
MODELS_DIR = "data/models"  # directory to save the trained model
MODEL_OUTPUT_PATH = "data/models/citizen4_citizen4_LabPretrainedVGG16_unfreeze_last_3_layers.pth"  # where to save the trained model
PRETRAINED_VGGFACE_PATH = "/home/ssd_storage/experiments/ng_ids_imgs_tradeoff/vgg_pre_trained_models/face_trained_vgg16_119.pth"

# Training parameters
BATCH_SIZE = 32
NUM_CLASSES = 2  # adjust based on number of actors
LEARNING_RATE = 1e-4  # initial learning rate
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNEAL_STRATEGY = 'cos'

# Face matching parameters
SIMILARITY_THRESHOLD = 0.75  # threshold for face matching confidence 

#Face Recognition model embeddings constants
# Penultimate layer name of the VGG16 model (after RELU activation, before DROPOUT layer)
VGG_EMBEDDING_LAYER_NAME = 'classifier.4'
RESNET_EMBEDDING_LAYER_NAME = 'last_bn'

# # Tensorboard Logging
# LOG_BASE_DIR = "./logs"
# RUN_NAME = 'Citizen4_Lab_VGG16_FINE_TUNED_FREEZE_GRADS'

TRAIN_MODEL_CONFIG = ModelConfig(
    arch="resnet",
    num_classes=NUM_CLASSES,
    pretrained=True,
    train_mode=True,
    device=DEVICE,
    checkpoint_path=None,
    freeze_grads=True,
    additional_trainable_keywords=None,
    data_parallel_patch=False,
    show_logs=True
)