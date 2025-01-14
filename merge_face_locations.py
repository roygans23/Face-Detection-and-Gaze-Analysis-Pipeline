import pandas as pd
import re
import os
from pathlib import Path

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

def merge_face_matches_with_locations():
    # Define directories
    similarity_dir = Path('data/similarity_to_actor')
    face_locations_dir = Path('data/videos_csv')
    
    # Get all face matches CSV files
    face_matches_files = list(similarity_dir.glob('*_face_matches.csv'))
    
    for face_matches_file in face_matches_files:
        # Get video name
        video_name = get_video_name(face_matches_file.name)
        face_locations_file = face_locations_dir / f"{video_name}.csv"
        
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
        
        # Save the merged result
        output_path = similarity_dir / f"{video_name}_face_matches_with_locations.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Saved merged file to: {output_path}")
        print(f"Total matches: {len(merged_df)}")
        
        # Print any unmatched entries
        unmatched = merged_df[merged_df['top_left'].isna()]
        if len(unmatched) > 0:
            print(f"Warning: {len(unmatched)} entries had no matching face locations")

if __name__ == "__main__":
    merge_face_matches_with_locations() 