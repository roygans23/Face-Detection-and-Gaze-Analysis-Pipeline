"""
Step 1: Face Detection and Extraction Pipeline

This script performs the initial processing of video files to detect and extract faces:

1. Face Detection:
   - Uses a pre-trained Caffe model to detect faces in video frames
   - Processes videos at specified intervals (frame_skip)
   - Returns face detection results as Python dictionaries

2. Data Processing:
   - Converts detection results to pandas DataFrames
   - Extracts detected faces from original videos
   - Saves face images to organized directories
   - Creates CSV files with detection metadata

Required folder structure:
- videos/: Input video files
- faces/: Output extracted face images
- csv/: Output metadata CSV files

Configuration parameters are loaded from config.py
"""

import torch
from config.config_step_1 import *
import numpy as np
import pandas as pd
import os
import warnings
import logging

logging.getLogger('opencv-python').setLevel(logging.ERROR)
os.environ["OPENCV_LOG_LEVEL"]="ERROR"
os.environ["OPENCV_FFMPEG_DEBUG"]="0"

from modules.face_detection import load_caffe_model, process_videos_in_folder
from modules.detection_to_dataframe import create_dataframes_from_dict
from modules.face_extraction import process_dataframes

def ensure_folders_exist():
    """Create necessary folders if they don't exist"""
    folders = [
        VIDEO_FOLDER,
        FACES_FOLDER,
        CSV_OUTPUTS
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}")

def main():
    print("\n=== Starting Step 1: Face Detection and Extraction ===\n")
    
    # Ensure folders exist
    ensure_folders_exist()
    
    # Debug information
    print("\nChecking paths and files:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Video folder: {VIDEO_FOLDER}")
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found videos: {videos}")
    print(f"Prototxt exists: {os.path.exists(PROTOTXT_PATH)}")
    print(f"Caffe model exists: {os.path.exists(CAFFE_MODEL_PATH)}")

    # 1. Load face detection model
    print("\nLoading face detection model...")
    try:
        net = load_caffe_model(PROTOTXT_PATH, CAFFE_MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 2. Process videos and get face detection results
    print("\nProcessing videos for face detection...")
    try:
        face_detection_results = process_videos_in_folder(
            folder_path=VIDEO_FOLDER,
            net=net,
            frame_skip=FRAME_SKIP,
            conf_threshold=CONF_THRESHOLD,
            nms_threshold=NMS_THRESHOLD
        )
        print("Video processing completed!")
    except Exception as e:
        print(f"Error processing videos: {str(e)}")
        return

    # 3. Convert detection results to DataFrames
    print("\nConverting detection results to DataFrames...")
    try:
        dataframes = create_dataframes_from_dict(face_detection_results)
        print(f"Created {len(dataframes)} dataframes")
    except Exception as e:
        print(f"Error creating dataframes: {str(e)}")
        return

    # 4. Extract faces from the videos
    print("\nExtracting faces from videos...")
    try:
        process_dataframes(
            dataframes,
            video_folder=VIDEO_FOLDER,
            output_base_folder=FACES_FOLDER
        )
        print("Face extraction completed!")
    except Exception as e:
        print(f"Error extracting faces: {str(e)}")
        return
    
    # 5. Save DataFrames as CSV files
    print("\nCreating CSV files...")
    try:
        for video_name, df in dataframes.items():
            csv_path = os.path.join(CSV_OUTPUTS, f"{video_name}.csv")
            df.to_csv(csv_path)
            print(f"Saved CSV for {video_name}")
    except Exception as e:
        print(f"Error saving CSVs: {str(e)}")
        return

    print("\n=== Step 1 completed successfully! ===")

if __name__ == "__main__":
    main()