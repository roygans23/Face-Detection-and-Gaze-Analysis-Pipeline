import torch

# Training data paths
ACTORS_READY_FOLDER = "data/actors_ready"  # folder containing processed actor images
FACES_FOLDER = "data/faces"  # folder containing extracted faces to be matched
MODEL_OUTPUT_PATH = "models/fine_tuned_vgg16.pth"  # where to save the trained model

# Training parameters
BATCH_SIZE = 32
NUM_CLASSES = 48  # adjust based on number of actors
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Face matching parameters
SIMILARITY_THRESHOLD = 0.75  # threshold for face matching confidence 