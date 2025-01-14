import os
import torch
import pandas as pd
from modules.feature_extraction import (
    extract_aligned_face,
    generate_embedding,
    generate_actor_embeddings,
    find_best_match
)

def main():
    """
    Main function to:
    1. Generate mean embeddings for each actor
    2. Process all faces and find best matches for each video subfolder
    3. Save results to CSV for each video
    """
    import config
    
    print("\n=== Starting Step 2: Finding Best Matches ===\n")
    
    # 1. Generate actor embeddings
    print("Generating actor embeddings...")
    actor_embeddings = generate_actor_embeddings(config.ACTORS_READY_FOLDER)
    print(f"Generated embeddings for {len(actor_embeddings)} actors")
    
    # Get all video subfolders
    video_folders = [f.path for f in os.scandir(config.FACES_FOLDER) if f.is_dir()]
    
    # Process each video folder separately
    for video_folder in video_folders:
        results = []
        video_name = os.path.basename(video_folder)
        print(f"\nProcessing faces for video: {video_name}")
        
        # Process each face in the current video folder
        for root, _, files in os.walk(video_folder):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(root, file)
                
                # Extract and generate embedding for the face
                face = extract_aligned_face(image_path)
                if face is None:
                    print(f"No face detected in {image_path}")
                    continue
                    
                face_embedding = generate_embedding(face)
                
                # Find best matching actor
                best_actor, similarity = find_best_match(
                    face_embedding, 
                    actor_embeddings, 
                    threshold=config.THRESHOLD
                )
                
                # Store result with full path
                results.append({
                    'image_path': image_path,
                    'best_match_actor': best_actor if best_actor else "unknown",
                    'similarity_score': float(similarity)
                })
        
        # Save results for current video to CSV
        print(f"\nSaving results for {video_name}...")
        
        # Define similarity folder path
        similarity_folder = os.path.join(config.DATA_DIR, "similarity_to_actor")
        
        # Check if folder exists, if not create it
        if not os.path.exists(similarity_folder):
            os.makedirs(similarity_folder)
        
        # Create CSV with video name
        csv_path = os.path.join(similarity_folder, f"{video_name}_face_matches.csv")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to: {csv_path}")
        print(f"Total images processed for {video_name}: {len(results)}")
    
    print("\n=== Step 2 completed successfully! ===")

if __name__ == "__main__":
    main() 