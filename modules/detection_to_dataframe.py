# detection_to_dataframe.py

import os
import pandas as pd
import numpy as np


def create_dataframes_from_dict(face_detection_results):
    """
    Creates DataFrames from face detection results dictionary.
    Returns: {video_name: DataFrame}
    """
    if not face_detection_results:
        print("No face detections found")
        return {}

    dataframes = {}

    for video_name, face_data in face_detection_results.items():
        data = []
        for frame_idx, faces in face_data.items():
            if not faces:
                continue
                
            for idx, face in enumerate(faces):
                data.append({
                    "frame": int(frame_idx),
                    "index_in_frame": idx + 1,
                    "top_left": face[0],
                    "bottom_right": face[1]
                })

        if data:  # Only create DataFrame if we have data
            df = pd.DataFrame(data)
            df.set_index("frame", inplace=True)
            dataframes[video_name] = df

    return dataframes
