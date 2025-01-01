# json_to_dataframe.py

import os
import json
import pandas as pd
import numpy as np


def create_dataframes_from_jsons(json_folder):
    """
    Reads all JSON files in a folder and creates DataFrames for each JSON file.
    Returns: {json_filename (without .json): DataFrame}
    """
    dataframes = {}

    for filename in os.listdir(json_folder):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_folder, filename)

            with open(json_path, "r") as f:
                face_data = json.load(f)

            data = []
            for frame, faces in face_data.items():
                if not faces:
                    data.append({
                        "frame": int(frame),
                        "index_in_frame": None,
                        "top_left": None,
                        "bottom_right": None
                    })
                else:
                    for index, face in enumerate(faces, start=1):
                        data.append({
                            "frame": int(frame),
                            "index_in_frame": int(index),
                            "top_left": face[0],
                            "bottom_right": face[1]
                        })

            df = pd.DataFrame(data)
            df.set_index("frame", inplace=True)

            # Use the filename without extension as the key
            base_name, _ = os.path.splitext(filename)
            dataframes[base_name] = df

    return dataframes
