# config_step_1.py - Face Detection Configuration

# --- DATA PATHS ---
DATA_DIR = "data"
VIDEO_FOLDER = "data/videos"
FACES_FOLDER = "data/faces"
CSV_OUTPUTS = "data/csvs_outputs"  # For generated CSVs

# --- MODEL FILES (EDIT THESE TO MATCH YOUR SYSTEM) ---
PROTOTXT_PATH = r"utils/deploy.prototxt.txt"
CAFFE_MODEL_PATH = r"utils/res10_300x300_ssd_iter_140000.caffemodel"

# --- FACE DETECTION THRESHOLDS ---
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
FRAME_SKIP = 24 # FPS * TR = 24 * 1 = 24