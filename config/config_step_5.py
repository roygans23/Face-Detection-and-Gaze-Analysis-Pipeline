import torch

# Training data paths
ACTORS_READY_FOLDER = "data/actors_ready"  # folder containing processed actor images
FACES_FOLDER = "data/faces"  # folder containing extracted faces to be matched
MODELS_DIR = "data/models"  # directory to save the trained model
MODEL_OUTPUT_PATH = "data/models/fine_tuned_face_trained_vgg16_119_frame_skip=24.pth"  # where to save the trained model
PRETRAINED_VGGFACE_PATH = "/home/ssd_storage/experiments/ng_ids_imgs_tradeoff/vgg_pre_trained_models/face_trained_vgg16_119.pth"

# Training parameters
BATCH_SIZE = 32
NUM_CLASSES = 48  # adjust based on number of actors
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Face matching parameters
SIMILARITY_THRESHOLD = 0.75  # threshold for face matching confidence 