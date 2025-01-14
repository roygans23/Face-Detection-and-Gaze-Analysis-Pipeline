# feature_extraction.py

import os
import shutil
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import time


# Initialize device, FaceNet, MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)


def extract_aligned_face(image_path):
    """
    Extract and align a face from an image file using MTCNN.
    Returns a torch.Tensor or None if extraction fails.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')

            # Optionally resize if extremely large
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))

            face = mtcnn(img)
            if face is not None:
                return face
            else:
                return None

    except (OSError, IOError) as e:
        print(f"File error with {image_path}: {e}")
        return None
    except MemoryError:
        print(f"MemoryError: Unable to process {image_path}. Skipping file.")
        return None
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
        return None


def generate_embedding(face):
    """
    Generates a 1D embedding tensor from an aligned face using FaceNet.
    """
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face)
    return embedding.squeeze(0)


def generate_actor_embeddings(actors_ready_folder):
    """
    Generates face embeddings for each actor in `actors_ready_folder`.
    Returns a dictionary: {actor_name: mean_embedding_tensor}
    The embedding for each actor is the mean of embeddings from all their images.
    """
    actor_embeddings = {}
    
    for actor_name in os.listdir(actors_ready_folder):
        actor_dir = os.path.join(actors_ready_folder, actor_name)
        if not os.path.isdir(actor_dir):
            continue

        # Get all images for this actor
        images = [f for f in os.listdir(actor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"No images found for actor: {actor_name}")
            continue

        print(f"Processing {len(images)} images for {actor_name}")
        embeddings_list = []

        # Generate embedding for each image
        for image_name in images:
            image_path = os.path.join(actor_dir, image_name)
            face = extract_aligned_face(image_path)
            
            if face is not None:
                embedding = generate_embedding(face)
                embeddings_list.append(embedding)

        if embeddings_list:
            # Calculate mean embedding
            mean_embedding = torch.stack(embeddings_list).mean(dim=0)
            actor_embeddings[actor_name] = mean_embedding
            print(f"Generated mean embedding for {actor_name} from {len(embeddings_list)} images")
        else:
            print(f"No valid embeddings generated for {actor_name}")

    return actor_embeddings


def find_best_match(face_embedding, actor_embeddings, threshold=0.9):
    """
    Finds the best-matching actor. Returns (best_actor, best_similarity) or (None, best_similarity) if below threshold.
    """
    best_similarity = -1.0
    best_actor = None

    for actor_name, embedding in actor_embeddings.items():
        sim = cosine_similarity(
            face_embedding.unsqueeze(0),
            embedding.unsqueeze(0),
            dim=1
        ).item()

        if sim > best_similarity:
            best_similarity = sim
            best_actor = actor_name

    if best_similarity >= threshold:
        return best_actor, best_similarity
    else:
        return None, best_similarity

def match_and_move_faces(faces_folder, actors_ready_folder, actor_embeddings, threshold=0.9):
    """
    Recursively matches each face in `faces_folder` (including subfolders) to the best actor (above `threshold`)
    and moves it to that actor's folder.
    If no match is found, moves the file to an "unknown" folder in the same directory as faces_folder.
    """
    
    # Create the "unknown" folder in the same directory as faces_folder
    unknown_dir = os.path.join(faces_folder, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    print(f"Unknown directory ensured at: {unknown_dir}")
    
    # Traverse all subdirectories and files within faces_folder
    for root, dirs, files in os.walk(faces_folder):
        # Skip the 'unknown' directory to prevent re-processing moved files
        dirs[:] = [d for d in dirs if d.lower() != 'unknown']
        
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {file}")
                continue  # Skip non-image files
            
            face_image_path = os.path.join(root, file)
            
            # Extract face from the image
            face = extract_aligned_face(face_image_path)
            if face is None:
                print(f"No face extracted from: {face_image_path}. Skipping.")
                continue  # Skip if no face is detected
            
            # Generate face embedding
            face_embedding = generate_embedding(face)
            if face_embedding is None:
                print(f"Failed to generate embedding for: {face_image_path}. Skipping.")
                continue  # Skip if embedding generation fails
            
            # Find best match from actor embeddings
            best_actor, best_similarity = find_best_match(face_embedding, actor_embeddings, threshold)
            
            if best_actor is not None:
                # We have a match above the threshold; move the file to best_actor's folder
                actor_dir = os.path.join(actors_ready_folder, best_actor)
                os.makedirs(actor_dir, exist_ok=True)
                
                # To prevent filename conflicts, append a timestamp to the filename
                timestamp = int(time.time())
                new_filename = f"{os.path.splitext(file)[0]}_{timestamp}{os.path.splitext(file)[1]}"
                destination_path = os.path.join(actor_dir, new_filename)
                
                shutil.move(face_image_path, destination_path)
            else:
                # No match found above threshold, move image to "unknown" folder
                # To prevent filename conflicts, append a timestamp to the filename
                timestamp = int(time.time())
                new_filename = f"{os.path.splitext(file)[0]}_{timestamp}{os.path.splitext(file)[1]}"
                destination_path = os.path.join(unknown_dir, new_filename)
                
                shutil.move(face_image_path, destination_path)
    
    print("[INFO] Completed matching and moving faces.")