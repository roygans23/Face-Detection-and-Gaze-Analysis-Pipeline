# config_step_2.py - Feature Extraction Configuration

import torch

# --- DATA PATHS ---
DATA_DIR = "data"
FACES_FOLDER = "data/faces"
ACTORS_READY_FOLDER = "data/actors_ready"
AUGMENTED_FOLDER = "data/actors_ready_augmented"
GAZES_FOLDER = "data/gazes_csv"  # Input data folder
CSV_OUTPUTS = "data/csvs_outputs"  # For generated CSVs

# --- FEATURE EXTRACTION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.75  # Similarity threshold for matching 