import torch
import config
import numpy as np
import pandas as pd
import os
import warnings
import logging

logging.getLogger('opencv-python').setLevel(logging.ERROR)
os.environ["OPENCV_LOG_LEVEL"]="ERROR"
os.environ["OPENCV_FFMPEG_DEBUG"]="0"

from modules.face_detection import load_caffe_model, process_videos_in_folder
from modules.json_to_dataframe import create_dataframes_from_jsons
from modules.face_extraction import process_dataframes

def ensure_folders_exist():
    """Create necessary folders if they don't exist"""
    folders = [
        config.VIDEO_FOLDER,
        config.JSON_FOLDER,
        config.FACES_FOLDER,
        config.CSV_FOLDER
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
    print(f"Video folder: {config.VIDEO_FOLDER}")
    videos = [f for f in os.listdir(config.VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found videos: {videos}")
    print(f"Prototxt exists: {os.path.exists(config.PROTOTXT_PATH)}")
    print(f"Caffe model exists: {os.path.exists(config.CAFFE_MODEL_PATH)}")

    # 1. Load face detection model
    print("\nLoading face detection model...")
    try:
        net = load_caffe_model(config.PROTOTXT_PATH, config.CAFFE_MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 2. Process videos to JSON
    print("\nProcessing videos for face detection...")
    try:
        process_videos_in_folder(
            folder_path=config.VIDEO_FOLDER,
            output_folder=config.JSON_FOLDER,
            net=net,
            frame_skip=config.FRAME_SKIP,
            conf_threshold=config.CONF_THRESHOLD,
            nms_threshold=config.NMS_THRESHOLD
        )
        print("Video processing completed!")
    except Exception as e:
        print(f"Error processing videos: {str(e)}")
        return

    # 3. Convert JSONs -> DataFrames
    print("\nConverting JSON files to DataFrames...")
    try:
        dataframes = create_dataframes_from_jsons(config.JSON_FOLDER)
        print(f"Created {len(dataframes)} dataframes")
    except Exception as e:
        print(f"Error creating dataframes: {str(e)}")
        return

    # 4. Extract faces from the videos
    print("\nExtracting faces from videos...")
    try:
        process_dataframes(
            dataframes,
            video_folder=config.VIDEO_FOLDER,
            output_base_folder=config.FACES_FOLDER
        )
        print("Face extraction completed!")
    except Exception as e:
        print(f"Error extracting faces: {str(e)}")
        return
    
    # 5. Save DataFrames as CSV files
    print("\nCreating CSV files...")
    try:
        for video_name, df in dataframes.items():
            csv_path = os.path.join(config.CSV_FOLDER, f"{video_name}.csv")
            df.to_csv(csv_path)
            print(f"Saved CSV for {video_name}")
    except Exception as e:
        print(f"Error saving CSVs: {str(e)}")
        return

    print("\n=== Step 1 completed successfully! ===")

if __name__ == "__main__":
    main()