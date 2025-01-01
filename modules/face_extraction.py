# face_extraction.py

import os
import cv2
import pandas as pd 
import numpy as np


def save_faces_from_dataframe(video_path, face_data, output_folder="faces"):
    """
    Extracts and saves face images from a video based on bounding box coordinates
    provided in a DataFrame. Skips invalid or problematic crops.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    for frame_idx in sorted(face_data.index.unique()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Get all rows for this frame
        rows_for_frame = face_data.loc[[frame_idx]] if frame_idx in face_data.index else None
        if rows_for_frame is None:
            continue

        # If only one row (single face), convert to DataFrame
        if isinstance(rows_for_frame, pd.Series):
            rows_for_frame = rows_for_frame.to_frame().T

        for i, row in rows_for_frame.iterrows():
            index_in_frame = row['index_in_frame']
            top_left = row['top_left']
            bottom_right = row['bottom_right']

            if pd.isnull(index_in_frame) or top_left is None or bottom_right is None:
                continue

            (x1, y1) = top_left
            (x2, y2) = bottom_right

            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                out_name = f"frame_{frame_idx}_face_{index_in_frame}.png"
                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, face_crop)

    cap.release()


def process_dataframes(dataframes, video_folder, output_base_folder):
    """
    Iterates through DataFrames and extracts faces from their corresponding videos.
    """
    for video_name, df in dataframes.items():
        video_path = os.path.join(video_folder, f"{video_name}.mp4")  # Adjust if other extension
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}, skipping...")
            continue

        output_folder = os.path.join(output_base_folder, video_name)
        print(f"Processing video: {video_name}")
        save_faces_from_dataframe(video_path, df, output_folder)
