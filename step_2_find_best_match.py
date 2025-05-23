from config.config_step_2 import *
import os
import numpy as np
import pandas as pd
from modules.feature_extraction import FeatureExtraction

"""
Step 2: Face Matching Pipeline

This script performs face matching between extracted faces and actor reference images:

1. Actor Embedding Generation:
   - Generates mean embeddings for each actor from reference images
   - Uses a pre-trained face recognition model for embedding generation

2. Face Matching:
   - Processes each video's extracted faces
   - Generates embeddings for detected faces
   - Finds best matching actor based on embedding similarity
   - Applies confidence threshold for matching

3. Results Storage:
   - Creates CSV files for each video containing match results
   - Stores image paths, best matching actor, and similarity scores

Required folder structure:
- actors_ready/: Preprocessed actor reference images
- faces/: Input extracted face images from Step 1
- similarity/: Output CSV files with matching results

Configuration parameters are loaded from config.py
"""

def process_face_folder(folder_path, actor_embeddings, feature_extraction):
    """Process all faces in a folder and find best matches"""
    results = []
    
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(folder_path, image_name)
        
        # Extract and align face
        face = feature_extraction.extract_aligned_face(image_path)
        if face is None:
            print(f"Failed to extract face from {image_name}")
            continue
        print(f"Generating embedding for {image_name}...")
            
        # Generate embedding
        face_embedding = feature_extraction.generate_embedding(face)
        if face_embedding is None:
            continue
            
        # Find best match
        best_actor, similarity = feature_extraction.find_best_match(face_embedding, actor_embeddings)
        
        results.append({
            'image_path': image_name,
            'best_match': best_actor,
            'similarity': similarity
        })
    
    return results

def main():
    """
    Main function to:
    1. Generate mean embeddings for each actor
    2. Process all faces and find best matches for each video subfolder
    3. Save results to CSV for each video
    4. Merge face matches with locations
    """
    print("\n=== Starting Step 2: Finding Best Matches ===\n")

    feature_extraction = FeatureExtraction(
        face_recognition_model_checkpoint=MODEL_OUTPUT_PATH,
        device=DEVICE
    )
    
    # 1. Generate actor embeddings
    print("Generating actor embeddings...")
    actor_embeddings = feature_extraction.generate_actor_embeddings(ACTORS_READY_FOLDER)
    print(f"Generated embeddings for {len(actor_embeddings)} actors")
    
    # Get all video subfolders
    video_folders = [f.path for f in os.scandir(FACES_FOLDER) if f.is_dir()]
    
    # Process each video folder separately
    for video_folder in video_folders:
        results = []
        video_name = os.path.basename(video_folder)
        print(f"\nProcessing faces for video: {video_name}")
        
        # Process faces in current video folder
        results = process_face_folder(video_folder, actor_embeddings, feature_extraction)
        
        # Save results for current video to CSV
        print(f"\nSaving results for {video_name}...")
        
        # Check if CSV_OUTPUTS folder exists, if not create it
        if not os.path.exists(CSV_OUTPUTS):
            os.makedirs(CSV_OUTPUTS)
        
        # Create CSV with video name directly in CSV_OUTPUTS
        csv_path = os.path.join(CSV_OUTPUTS, f"{video_name}_face_matches.csv")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to: {csv_path}")
        print(f"Total images processed for {video_name}: {len(results)}")
    
    # Add merging step
    print("\n=== Merging face matches with locations ===")
    from modules.merge_face_locations import merge_face_matches_with_locations
    merge_face_matches_with_locations()
    
    print("\n=== Step 2 completed successfully! ===")

if __name__ == "__main__":
    main() 