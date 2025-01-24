"""
Step 3: Eye Gaze Integration Pipeline

This script integrates eye gaze data with face detection and recognition results:

1. Gaze Integration:
   - Uses previously merged face detection and recognition data
   - Maps each gaze point to detected faces in the same frame
   - Determines if gaze points fall within face bounding boxes
   - Associates gazes with recognized actors when applicable

2. Data Processing:
   - Processes each video's gaze data separately
   - Combines gaze coordinates with face recognition results
   - Handles missing or invalid gaze data points
   - Creates integrated datasets with gaze-face associations

3. Results Storage:
   - Generates CSV files containing integrated gaze and face data
   - Stores frame numbers, gaze coordinates, and matched actor information
   - Includes similarity scores for matched faces

Required folder structure:
- gazes/: Input eye gaze data files
- merged_csv/: Input merged face detection and recognition data
- integrated_gazes/: Output integrated gaze and face data

Configuration parameters are loaded from config.py
"""

import pandas as pd
from pathlib import Path
import os
from config.config_step_3 import (
    GAZES_FOLDER,
    CSV_OUTPUTS
)
from modules.merge_face_locations import get_video_name

def get_video_name_from_gaze(filename):
    """Extract video name from gaze filename by removing 'gazes_' prefix and '.csv' suffix."""
    name = filename.replace('gazes_', '')  # Remove gazes_ prefix
    name = name.replace('.csv', '')  # Remove .csv extension
    return name

def integrate_gazes_with_faces():
    gazes_dir = Path(GAZES_FOLDER)
    output_dir = Path(CSV_OUTPUTS)
    
    # Get all gaze CSV files
    gaze_files = list(gazes_dir.glob('*.csv'))
    print(f"\nFound {len(gaze_files)} gaze files")
    
    for gaze_file in gaze_files:
        # Get video name from gaze file
        video_name = get_video_name_from_gaze(gaze_file.name)
        merged_file = output_dir / f"{video_name}_merged.csv"
        
        if not merged_file.exists():
            print(f"Warning: No corresponding merged file found for {video_name}")
            continue
            
        print(f"\nProcessing gazes for {video_name}...")
        
        # Read the CSVs
        gaze_df = pd.read_csv(gaze_file)
        face_df = pd.read_csv(merged_file)  # This already has the coordinates split
        
        # Create result DataFrame
        result_df = gaze_df.copy()
        result_df['best_match'] = None
        result_df['similarity'] = None
        
        # For each gaze entry
        for idx, gaze_row in result_df.iterrows():
            frame = gaze_row['frame']
            x, y = gaze_row['Gaze point X'], gaze_row['Gaze point Y']
            
            # Get faces in the same frame
            frame_faces = face_df[face_df['frame'] == frame]
            
            # Find which face contains the gaze point
            for _, face_row in frame_faces.iterrows():
                if (face_row['top_left_x'] <= x <= face_row['bottom_right_x'] and
                    face_row['top_left_y'] <= y <= face_row['bottom_right_y']):
                    result_df.at[idx, 'best_match'] = face_row['best_match']
                    result_df.at[idx, 'similarity'] = face_row['similarity']
                    break
        
        # Save the result
        output_file = output_dir / f"{video_name}_with_gazes.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Saved integrated file to: {output_file}")
        
        # Print statistics
        matches = result_df['best_match'].notna().sum()
        print(f"Total gaze points: {len(result_df)}")
        print(f"Gaze points matched to faces: {matches}")

if __name__ == "__main__":
    integrate_gazes_with_faces()
