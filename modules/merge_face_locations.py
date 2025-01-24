"""
This script merges face detection data with face similarity matches.

It processes pairs of CSV files:
1. Face matches files (from data/similarity_to_actor/) containing similarity scores
2. Face locations files (from data/videos_csv/) containing bounding box coordinates

For each video, it:
- Matches faces across files using frame number and face index
- Splits coordinate data into separate columns
- Creates a new CSV combining similarity scores with face locations
- Saves results as '*_face_matches_with_locations.csv'

The script handles coordinate parsing and provides warnings for unmatched entries.
"""

import pandas as pd
import re
import os
from pathlib import Path
import ast  # for safely evaluating string literals
from config.config_step_3 import (
    CSV_OUTPUTS,
    DATA_DIR
)

# All files are now in CSV_OUTPUTS
INPUT_OUTPUT_DIR = CSV_OUTPUTS

def extract_frame_and_index(image_path):
    """Extract frame number and face index from image path."""
    match = re.search(r'frame_(\d+)_face_(\d+)', image_path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_video_name(filename):
    """Extract video name from filename."""
    # Remove _face_matches from similarity files
    name = filename.replace('_face_matches', '')
    # Remove .csv extension
    name = name.replace('.csv', '')
    return name

def split_coordinates(df):
    """Split top_left and bottom_right coordinates into separate integer columns."""
    # Convert string representations of lists to actual lists
    df['top_left'] = df['top_left'].apply(ast.literal_eval)
    df['bottom_right'] = df['bottom_right'].apply(ast.literal_eval)
    
    # Extract coordinates into separate columns
    df['top_left_x'] = df['top_left'].apply(lambda x: int(x[0]))
    df['top_left_y'] = df['top_left'].apply(lambda x: int(x[1]))
    df['bottom_right_x'] = df['bottom_right'].apply(lambda x: int(x[0]))
    df['bottom_right_y'] = df['bottom_right'].apply(lambda x: int(x[1]))
    
    # Drop the original coordinate columns
    df = df.drop(['top_left', 'bottom_right'], axis=1)
    
    return df

def merge_face_matches_with_locations():
    # Define directory using config path
    io_dir = Path(INPUT_OUTPUT_DIR)
    
    # Add debug prints
    print(f"Looking for files in: {io_dir}")
    print(f"Output directory will be: {io_dir}")
    
    # Create output directory if it doesn't exist
    io_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all face matches CSV files and add debug print
    face_matches_files = list(io_dir.glob('*_face_matches.csv'))
    print(f"Found {len(face_matches_files)} face matches files")
    
    if len(face_matches_files) == 0:
        print("No face matches files found! Please check if the path is correct:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected path: {io_dir.absolute()}")
        return
    
    for face_matches_file in face_matches_files:
        # Get video name
        video_name = get_video_name(face_matches_file.name)
        face_locations_file = io_dir / f"{video_name}.csv"
        
        # Check if corresponding face locations file exists
        if not face_locations_file.exists():
            print(f"Warning: No corresponding face locations file found for {video_name}")
            continue
            
        print(f"\nProcessing {video_name}...")
        
        # Read the CSVs
        matches_df = pd.read_csv(face_matches_file)
        face_locations_df = pd.read_csv(face_locations_file)
        
        # Extract frame and index from image_path
        matches_df[['frame', 'index_in_frame']] = pd.DataFrame(
            matches_df['image_path'].apply(extract_frame_and_index).tolist(),
            columns=['frame', 'index_in_frame']
        )
        
        # Merge the dataframes
        merged_df = pd.merge(
            matches_df,
            face_locations_df,
            on=['frame', 'index_in_frame'],
            how='left'
        )
        
        # Split coordinates into separate columns
        merged_df = split_coordinates(merged_df)
        
        # Save the merged result with new path and naming convention
        output_path = io_dir / f"{video_name}_merged.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Saved merged file to: {output_path}")
        print(f"Total matches: {len(merged_df)}")
        
        # Print any unmatched entries
        unmatched = merged_df[merged_df['top_left_x'].isna()]
        if len(unmatched) > 0:
            print(f"Warning: {len(unmatched)} entries had no matching face locations")

if __name__ == "__main__":
    merge_face_matches_with_locations() 